"""Gate 3: unpredictable late questions after interfering writes.

Loads Gate 2's frozen pretrained transformer and its frozen learned lift.
Only the 4 x 64 (1,024-byte) resident memory changes during each stream.

Run: python -m live_mode.hf_gate3 --dry-run
Then: python -m live_mode.hf_gate3 --trials 20
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence
import numpy as np
from .hf_gate2 import COLORS, PROMPTS, TEST_NAMES, FrozenLMAdapter
from .memory import LiveMemory, Settings
from .stream import VectorCodec

CODEC_SEED = 20261017
DEFAULT_SEEDS = (281828, 314159)
ARMS = ('live', 'erased', 'stale', 'wrong_address', 'last_only',
        'oracle_code', 'dictionary_context')


@dataclass
class StreamEpisode:
    prompt: str
    address: str
    truth: str
    previous: str
    response: np.ndarray
    stale_response: np.ndarray
    distractor_response: np.ndarray
    last_value_response: np.ndarray
    events: tuple[tuple[str, str], ...]
    protected_names: tuple[str, ...]
    final_truth: dict[str, str]
    selected_name: str
    latest_name: str
    last_target_write_index: int
    selected_rewrite_index: int
    slots: int
    delay: int
    memory_bytes: int
    query_selected_after_stream: bool = True


def available_names(slots: int) -> tuple[str, ...]:
    if slots < 4 or slots > 256:
        raise ValueError('slots must be between 4 and 256')
    # Gate 2's evaluation names, then further names absent from training.
    return tuple(TEST_NAMES[:slots]) if slots <= len(TEST_NAMES) else (
        *TEST_NAMES, *(f'Novice{i:03d}' for i in range(slots - len(TEST_NAMES))))


def new_memory(codec: VectorCodec) -> LiveMemory:
    return LiveMemory(Settings(key_dim=codec.key_dim, value_dim=codec.value_dim,
                               learning_rate=0.95, max_update_norm=3.0))


def stream_episode(rng: random.Random, codec: VectorCodec, slots: int,
                   delay: int) -> StreamEpisode:
    """Create a complete observation stream BEFORE drawing a late query.

    Four candidates are rewritten and kept untouched during the interfering
    suffix. The actual query selects one candidate only after the suffix.
    The stream history is retained by the *evaluator*, never by LiveMemory.
    """
    if delay < 1:
        raise ValueError('delay must be positive')
    names = available_names(slots)
    protected = tuple(rng.sample(list(names), 4))
    eligible = [name for name in names if name not in protected]
    if not eligible:
        raise ValueError('need at least one interfering address')
    keys = {name: codec.key(name + '.color') for name in names}
    values = {color: codec.value(color) for color in COLORS}
    observed: list[tuple[str, str]] = []
    truth: dict[str, str] = {}
    memory = new_memory(codec)

    def observe(name: str, color: str) -> None:
        observed.append((name, color))
        truth[name] = color
        memory.step(keys[name], values[color])

    shuffled = list(names)
    rng.shuffle(shuffled)
    for name in shuffled:
        observe(name, rng.choice(COLORS))

    # Every candidate has both an old and a new value.
    candidate_info: dict[str, tuple[str, int]] = {}
    order = list(protected)
    rng.shuffle(order)
    for name in order:
        previous = truth[name]
        replacement = rng.choice([c for c in COLORS if c != previous])
        candidate_info[name] = (previous, len(observed))
        observe(name, replacement)

    # Continue writing without allowing a protected fact to be overwritten.
    for _ in range(delay):
        observe(rng.choice(eligible), rng.choice(COLORS))

    # No question was supplied to the writer while the stream progressed.
    name = rng.choice(list(protected))
    previous, rewritten_at = candidate_info[name]
    truth_color = truth[name]
    readout = memory.read(keys[name])

    # Exact counterfactual: all identical events, omit ONLY this rewrite.
    stale = new_memory(codec)
    for index, (seen_name, color) in enumerate(observed):
        if index != rewritten_at:
            stale.step(keys[seen_name], values[color])
    stale_read = stale.read(keys[name])

    # Guarantee the wrong-address control has a different correct value.
    other = rng.choice([n for n in names if n != name and truth[n] != truth_color])
    wrong_read = memory.read(keys[other])
    latest_name, latest_color = observed[-1]
    assert latest_name != name
    assert len(observed) - 1 - rewritten_at >= delay
    return StreamEpisode(
        prompt=rng.choice(PROMPTS).format(name=name),
        address=name + '.color', truth=truth_color, previous=previous,
        response=readout, stale_response=stale_read,
        distractor_response=wrong_read,
        last_value_response=values[latest_color].copy(),
        events=tuple(observed), protected_names=protected,
        final_truth=truth.copy(), selected_name=name, latest_name=latest_name,
        last_target_write_index=rewritten_at, selected_rewrite_index=rewritten_at,
        slots=slots, delay=delay, memory_bytes=memory.state_bytes)


def score_logits(logits, label_ids: dict[str, int], truth: str) -> dict:
    import torch
    ids = torch.as_tensor([label_ids[color] for color in COLORS],
                          device=logits.device)
    candidates = logits[ids]
    correct = COLORS.index(truth)
    others = torch.cat((candidates[:correct], candidates[correct+1:]))
    return {
        'strict': int(int(torch.argmax(logits)) == label_ids[truth]),
        'choice': int(int(torch.argmax(candidates)) == correct),
        'margin': float((candidates[correct] - torch.max(others)).cpu()),
        'prediction': COLORS[int(torch.argmax(candidates))]
    }


def summarize(rows: list[dict], arm_names: Sequence[str] = ARMS) -> dict:
    n = len(rows)
    assert n > 0
    return {
        'trials': n,
        'scores': {arm: {
            'strict_next_token_accuracy': round(sum(r[arm]['strict'] for r in rows) / n, 4),
            'four_color_accuracy': round(sum(r[arm]['choice'] for r in rows) / n, 4),
            'mean_true_color_margin': round(sum(r[arm]['margin'] for r in rows) / n, 4)
        } for arm in arm_names},
        'live_correct_stale_wrong_rate': round(sum(
            r['live']['strict'] and not r['stale']['strict'] for r in rows) / n, 4),
        'live_correct_last_only_wrong_rate': round(sum(
            r['live']['strict'] and not r['last_only']['strict'] for r in rows) / n, 4)
    }


def gate_report(cells: list[dict]) -> dict:
    # Fixed engineering criteria before any Gate 3 Qwen evaluation.
    indexed = {(item['slots'], item['delay']): item['summary'] for item in cells}
    report = {}
    for slots, delay, threshold in ((16, 100, 0.8), (16, 500, 0.6)):
        label = f'slots_{slots}_delay_{delay}'
        summary = indexed.get((slots, delay))
        if summary is None:
            report[label] = 'not_run'
            continue
        scores = summary['scores']
        live = scores['live']['strict_next_token_accuracy']
        last = scores['last_only']['strict_next_token_accuracy']
        report[label] = {
            'pass': live >= threshold and live >= last + 0.3,
            'live_accuracy': live, 'last_only_accuracy': last,
            'required_live': threshold, 'required_advantage': 0.3
        }
    return report


def evaluate(model: FrozenLMAdapter, slots: tuple[int, ...],
             delays: tuple[int, ...], trials: int, seeds: tuple[int, ...]) -> dict:
    import torch
    codec = VectorCodec(COLORS, key_dim=64, seed=CODEC_SEED)
    cells, examples = [], []
    model.model.eval()
    with torch.no_grad():
        for size in slots:
            for delay in delays:
                rows = []
                for seed in seeds:
                    rng = random.Random(seed * 1000003 + size * 997 + delay * 17)
                    for _ in range(trials):
                        ep = stream_episode(rng, codec, size, delay)
                        readouts = {
                            'live': ep.response, 'erased': None,
                            'stale': ep.stale_response,
                            'wrong_address': ep.distractor_response,
                            'last_only': ep.last_value_response,
                            'oracle_code': codec.value(ep.truth),
                            'dictionary_context': None
                        }
                        row = {}
                        for arm in ARMS:
                            prompt = ep.prompt
                            if arm == 'dictionary_context':
                                prompt = (f'Fact: {ep.selected_name} has the color {ep.truth}.\n'
                                          + ep.prompt)
                            logits = model._logits(prompt, readouts[arm])
                            row[arm] = score_logits(logits, model.label_ids, ep.truth)
                        rows.append(row)
                        if len(examples) < 24 and len(rows) <= 2:
                            examples.append({
                                'slots': size, 'delay': delay, 'prompt': ep.prompt,
                                'previous': ep.previous, 'correct': ep.truth,
                                'last_event': ep.events[-1],
                                'color_guesses': {arm: row[arm]['prediction']
                                                  for arm in ARMS}
                            })
                summary = summarize(rows)
                cells.append({'slots': size, 'delay': delay, 'summary': summary})
                print(f'Gate3 slots={size:>3} delay={delay:>4}: '
                      f'live={summary["scores"]["live"]["strict_next_token_accuracy"]:.3f} '
                      f'last_only={summary["scores"]["last_only"]["strict_next_token_accuracy"]:.3f} '
                      f'dictionary={summary["scores"]["dictionary_context"]["four_color_accuracy"]:.3f}',
                      flush=True)
    return {
        'gate': 3,
        'description': 'Unpredictable late read after multiple independent overwrites',
        'configuration': {
            'slots': list(slots), 'interfering_writes': list(delays),
            'trials_per_seed_cell': trials, 'seeds': list(seeds),
            'trained_adapter_parameters': int(model.lift.weight.numel()),
            'online_memory_bytes': 1024, 'pretrained_model_frozen': True,
            'adapter_frozen': True, 'key_binder_supplied': True,
            'four_color_codebook_supplied': True,
            'query_selected_after_all_observations': True
        },
        'cells': cells, 'engineering_gates': gate_report(cells), 'examples': examples,
        'limitations': (
            'Small artificial four-color vocabulary and explicit hashed addresses. '
            'Gate 2 adapter is fixed; no raw-language semantic observation parser '
            'or evidence of efficiency over a dictionary. Each episode resets '
            'memory. Evaluator retains events only for control replay; online '
            'LiveMemory stores no event list. Additional slot names are synthetic.')
    }


def parse_positive_ints(value: str, name: str, minval: int = 1) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(','))
    except ValueError as exc:
        raise ValueError(f'--{name} expects comma-separated integers') from exc
    if not parsed or len(set(parsed)) != len(parsed) or any(x < minval for x in parsed):
        raise ValueError(f'--{name} must contain distinct integers >= {minval}')
    return parsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--checkpoint', default='runs/hf_gate2_adapter.pt')
    p.add_argument('--report', default='runs/hf_gate3_result.json')
    p.add_argument('--slots', default='8,16,32')
    p.add_argument('--delays', default='10,50,100,500')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--seeds', default='281828,314159')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        slots = parse_positive_ints(args.slots, 'slots', minval=5)
        delays = parse_positive_ints(args.delays, 'delays')
        seeds = parse_positive_ints(args.seeds, 'seeds')
        if args.trials < 1 or any(x > 256 for x in slots):
            raise ValueError('--trials must be positive and slots <= 256')
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        rng = random.Random(seeds[0])
        codec = VectorCodec(COLORS, key_dim=64, seed=CODEC_SEED)
        print('Gate 3 synthetic smoke check (not Qwen results):')
        for size in slots:
            for delay in delays:
                ep = stream_episode(rng, codec, size, delay)
                assert ep.selected_name != ep.latest_name and ep.previous != ep.truth
                print(f'slots={size:3} delay={delay:4}: {ep.selected_name} '
                      f'{ep.previous}->{ep.truth}, events={len(ep.events)}, '
                      f'operator={ep.memory_bytes} bytes')
        return
    model = FrozenLMAdapter(args.model, args.layer, device=args.device, gain=args.gain)
    state = model.torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (state['model'], state['layer'], state['gain']) != (
            args.model, args.layer, args.gain):
        raise ValueError('checkpoint model/layer/gain mismatch')
    model.lift.load_state_dict(state['adapter'])
    model.lift.requires_grad_(False)
    report = evaluate(model, slots, delays, args.trials, seeds)
    report['configuration'].update({
        'model': args.model, 'layer': args.layer, 'gain': args.gain,
        'checkpoint': args.checkpoint
    })
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Gate 3 engineering criteria:', json.dumps(
        report['engineering_gates'], indent=2))
    print('Saved:', path)


if __name__ == '__main__':
    main()
