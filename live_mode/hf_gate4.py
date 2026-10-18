"""Gate 4: can the reader learn its memory's response geometry?

Two separate questions, neither claiming native J-space:
 A: after offline calibration on train-only entity names and noisy streams,
    does an equally-sized frozen adapter read the same 1024-byte state better?
 B: do raw or mean-centered Qwen middle-layer name features provide useful
    associative *addresses* compared with the SHA baseline (direct memory
    decoding, no learned semantic binder and no gradient optimization)?

Qwen weights frozen throughout; live episodes change W only. Every question is
selected after the stream. Run --dry-run without Qwen, then GPU experiment.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import numpy as np

from .hf_gate2 import COLORS, PROMPTS, TRAIN_NAMES, TEST_NAMES, FrozenLMAdapter, episode
from .hf_gate3 import available_names, CODEC_SEED, score_logits
from .memory import LiveMemory, Settings
from .stream import VectorCodec

TRAIN_SEED = 729011
TEST_SEEDS = (813887, 997001)  # disjoint from Gate 3 and training
SIZES = (16, 32)
DELAYS = (100, 500)
CODEC = VectorCodec(COLORS, key_dim=64, seed=CODEC_SEED)


@dataclass(frozen=True)
class Sequence:
    events: tuple[tuple[str, str], ...]
    selected: str
    truth: str
    previous: str
    selected_rewrite: int
    wrong: str
    prompt: str
    last_color: str
    suffix_length: int


def names_for_training(slots: int) -> tuple[str, ...]:
    if not 5 <= slots <= 128:
        raise ValueError('training slots must lie in 5..128')
    return (*TRAIN_NAMES[:min(slots, len(TRAIN_NAMES))],
            *(f'CalibName{i:03}' for i in range(max(0, slots - len(TRAIN_NAMES)))))


def generate(rng: random.Random, names: tuple[str, ...], delay: int) -> Sequence:
    """Produce the full observation stream without choosing a query until afterward."""
    if len(names) < 5 or len(set(names)) != len(names) or delay < 1:
        raise ValueError('need >=5 unique entities and at least 1 intervening write')
    chosen = tuple(rng.sample(list(names), 4))
    other = [n for n in names if n not in chosen]
    order = list(names)
    rng.shuffle(order)
    known: dict[str, str] = {}
    events: list[tuple[str, str]] = []
    for n in order:
        color = rng.choice(COLORS)
        known[n] = color
        events.append((n, color))
    prev: dict[str, tuple[str, int]] = {}
    rewrites = list(chosen)
    rng.shuffle(rewrites)
    for n in rewrites:
        before = known[n]
        color = rng.choice([x for x in COLORS if x != before])
        prev[n] = (before, len(events))
        known[n] = color
        events.append((n, color))
    for _ in range(delay):
        n = rng.choice(other)
        color = rng.choice(COLORS)
        known[n] = color
        events.append((n, color))
    # This random draw occurs only AFTER observations are complete.
    target = rng.choice(chosen)
    wrong_candidates = [n for n in names if n != target and known[n] != known[target]]
    if not wrong_candidates:
        raise AssertionError('no different-valued wrong-address control')
    stale_value, stale_index = prev[target]
    assert events[-1][0] != target
    return Sequence(tuple(events), target, known[target], stale_value,
                    stale_index, rng.choice(wrong_candidates),
                    rng.choice(PROMPTS).format(name=target), events[-1][1], delay)


def replay(spec: Sequence, keys: dict[str, np.ndarray],
           exclude_write: int | None = None) -> LiveMemory:
    memory = LiveMemory(Settings(key_dim=64, value_dim=4, learning_rate=0.95,
                                 max_update_norm=3.0))
    for i, (n, label) in enumerate(spec.events):
        if i != exclude_write:
            memory.step(keys[n], CODEC.value(label))
    return memory


def hashed_keys(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: CODEC.key(name + '.color') for name in names}


def make_reads(spec: Sequence, keys: dict[str, np.ndarray]) -> dict[str, np.ndarray | None]:
    memory = replay(spec, keys)
    return {'live': memory.read(keys[spec.selected]),
            'stale': replay(spec, keys, spec.selected_rewrite).read(keys[spec.selected]),
            'wrong_address': memory.read(keys[spec.wrong]),
            'last_only': CODEC.value(spec.last_color),
            'oracle_code': CODEC.value(spec.truth),
            'erased': None}


def direct_score(reads: dict, expected: str) -> int:
    return int(CODEC.decode(reads['live']) == expected)


def pairwise_cosine(keys: dict[str, np.ndarray]) -> dict[str, float]:
    matrix = np.stack(list(keys.values()))
    dots = matrix @ matrix.T
    values = np.abs(dots[np.triu_indices(len(keys), k=1)])
    return {'mean_abs': round(float(np.mean(values)), 5),
            'max_abs': round(float(np.max(values)), 5)}


class NativeKeyer:
    """Fixed projection of frozen model activations; no J-lens, no learned keys.

    Raw and centered keys use the same frozen projector. The centering mean is
    estimated solely from TRAIN_NAMES; evaluation names are never calibration.
    Per-name Qwen forward passes and the projection are fixed scaffold costs
    separate from the 1024-byte mutable W. This still receives explicit names.
    """
    def __init__(self, adapter: FrozenLMAdapter, seed: int = 9173):
        torch = adapter.torch
        self.adapter = adapter
        hidden = adapter.lift.out_features
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((64, hidden)).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=1, keepdims=True)
        self.raw_cache: dict[str, np.ndarray] = {}
        self.center = np.mean([self.embedding(name) for name in TRAIN_NAMES], axis=0)

    def embedding(self, name: str) -> np.ndarray:
        if name not in self.raw_cache:
            adapter = self.adapter
            torch = adapter.torch
            prompt = f'Entity: {name}\nAttribute: color'
            inputs = {k: v.to(adapter.device) for k, v in
                      adapter.tokenizer(prompt, return_tensors='pt').items()}
            with torch.inference_mode():
                r = adapter.model(**inputs, output_hidden_states=True, use_cache=False)
                h = r.hidden_states[adapter.layer_index + 1][0, -1].float().cpu().numpy()
            self.raw_cache[name] = np.asarray(self.proj @ h, dtype=np.float32)
        return self.raw_cache[name].copy()

    def keys(self, names: tuple[str, ...], centered: bool = False) -> dict[str, np.ndarray]:
        keys = {}
        for name in names:
            v = self.embedding(name) - (self.center if centered else 0)
            norm = float(np.linalg.norm(v))
            if norm < 1e-8:
                raise ValueError('degenerate native representation')
            keys[name] = v / norm
        return keys


def load_gate2(adapter: FrozenLMAdapter, path: str) -> None:
    ckpt = adapter.torch.load(path, map_location='cpu', weights_only=True)
    if (ckpt['model'], ckpt['layer'], ckpt['gain']) != (
            adapter.model.config._name_or_path, adapter.layer_index, adapter.gain):
        # A local HF alias may differ from the checkpoint's public name.
        if (ckpt['layer'], ckpt['gain']) != (adapter.layer_index, adapter.gain):
            raise ValueError('wrong checkpoint layer/gain')
    adapter.lift.load_state_dict(ckpt['adapter'])


def train_reader(adapter: FrozenLMAdapter, steps: int, seed: int, lr: float,
                 curriculum: str) -> list[float]:
    """Matched offline steps for noisy-stream vs clean-memory reader, no test names."""
    if curriculum not in ("interference", "clean"):
        raise ValueError("unknown reader curriculum")
    rng = random.Random(seed)
    torch = adapter.torch
    adapter.lift.requires_grad_(True)
    optimizer = torch.optim.AdamW(adapter.lift.parameters(), lr=lr, weight_decay=0)
    losses = []
    for step in range(steps):
        if curriculum == 'interference':
            size = rng.choice((8, 16, 32))
            spec = generate(rng, names_for_training(size), rng.choice((10, 100, 500)))
            reads = make_reads(spec, hashed_keys(names_for_training(size)))
            prompt, response, truth = spec.prompt, reads['live'], spec.truth
        else:
            ep = episode(rng, TRAIN_NAMES, CODEC, rewrite=(step % 3 != 0))
            prompt, response, truth = ep.prompt, ep.response, ep.truth
        optimizer.zero_grad(set_to_none=True)
        logits = adapter._logits(prompt, response)
        label = torch.as_tensor([adapter.label_ids[truth]], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.lift.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            print(f'{curriculum} {step + 1}/{steps}: recent CE={np.mean(losses[-50:]):.4f}', flush=True)
    adapter.lift.requires_grad_(False)
    return losses


def eval_pair(adapter: FrozenLMAdapter, originals: dict, cleaned: dict,
              specs: list[Sequence]) -> tuple[dict, list[dict]]:
    """Paired baseline/robust evaluation; model and memory never update."""
    torch = adapter.torch
    updated = {k: v.detach().clone() for k, v in adapter.lift.state_dict().items()}
    scores = {arm: {'strict': [], 'choice': [], 'margin': []} for arm in
              ('gate2_reader', 'clean_extra_steps', 'robust_reader', 'erased', 'stale', 'wrong_address',
               'last_only', 'oracle_code', 'dictionary_context')}
    memory_correct = []
    samples = []
    with torch.inference_mode():
        for spec in specs:
            k = hashed_keys(tuple({n for n, _ in spec.events}))
            reads = make_reads(spec, k)
            memory_correct.append(direct_score(reads, spec.truth))
            for arm in scores:
                if arm == 'gate2_reader':
                    adapter.lift.load_state_dict(originals)
                    read = reads['live']
                elif arm == 'clean_extra_steps':
                    adapter.lift.load_state_dict(cleaned)
                    read = reads['live']
                else:
                    adapter.lift.load_state_dict(updated)
                    read = (reads['live'] if arm == 'robust_reader'
                            else reads.get(arm))
                prompt = spec.prompt
                if arm == 'dictionary_context':
                    prompt = f'Fact: {spec.selected} has the color {spec.truth}.\n' + prompt
                result = score_logits(adapter._logits(prompt, read),
                                      adapter.label_ids, spec.truth)
                for metric in ('strict', 'choice', 'margin'):
                    scores[arm][metric].append(result[metric])
                if len(samples) < 6 and arm in ('gate2_reader', 'robust_reader'):
                    if arm == 'gate2_reader':
                        samples.append({'prompt': spec.prompt, 'truth': spec.truth,
                                        'original': result['prediction']})
                    else:
                        samples[-1]['robust'] = result['prediction']
    adapter.lift.load_state_dict(updated)
    summary = {arm: {metric: round(float(np.mean(seq)), 4) for metric, seq in group.items()}
               for arm, group in scores.items()}
    summary['direct_memory_four_color_accuracy'] = round(float(np.mean(memory_correct)), 4)
    return summary, samples


def run(args) -> dict:
    model = FrozenLMAdapter(args.model, args.layer, args.device, args.gain)
    torch = model.torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint['model'], checkpoint['layer'], checkpoint['gain']) != (
            args.model, args.layer, args.gain):
        raise ValueError('original adapter model/layer/gain differs from requested')
    originals = checkpoint['adapter']
    model.lift.load_state_dict(originals)
    losses = []
    clean_losses = []
    if args.eval_only:
        state = torch.load(args.reader_checkpoint, map_location='cpu', weights_only=True)
        if (state['model'], state['layer'], state['gain']) != (args.model, args.layer, args.gain):
            raise ValueError('robust reader checkpoint model/layer/gain mismatch')
        model.lift.load_state_dict(state['adapter'])
        model.lift.requires_grad_(False)
        clean_state = torch.load(args.clean_checkpoint, map_location='cpu', weights_only=True)
        if (clean_state['model'], clean_state['layer'], clean_state['gain']) != (args.model, args.layer, args.gain):
            raise ValueError('clean checkpoint model/layer/gain mismatch')
        cleaned = clean_state['adapter']
    else:
        losses = train_reader(model, args.steps, args.train_seed, args.lr, 'interference')
        Path(args.reader_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': args.model, 'layer': args.layer, 'gain': args.gain,
                    'adapter': model.lift.state_dict()}, args.reader_checkpoint)
        robust = {k: v.detach().clone() for k, v in model.lift.state_dict().items()}
        model.lift.load_state_dict(originals)
        clean_losses = train_reader(model, args.steps, args.train_seed, args.lr, 'clean')
        cleaned = {k: v.detach().clone() for k, v in model.lift.state_dict().items()}
        Path(args.clean_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': args.model, 'layer': args.layer, 'gain': args.gain,
                    'adapter': model.lift.state_dict()}, args.clean_checkpoint)
        model.lift.load_state_dict(robust)

    report = {'gate': 4, 'hypothesis': 'reader calibration can rescue interference-damaged state',
              'configuration': {'model': args.model, 'layer': args.layer, 'gain': args.gain,
                                'adapter_train_names_only': list(TRAIN_NAMES),
                                'test_names_only': list(TEST_NAMES),
                                'train_steps': None if args.eval_only else args.steps,
                                'trained_reader_parameters': int(model.lift.weight.numel()),
                                'original_reader_parameters': int(model.lift.weight.numel()),
                                'persistent_memory_bytes': 1024,
                                'frozen_transformer': True,
                                'explicit_address_binding': True,
                                'no_jacobian_lens': True},
              'train_last50_cross_entropy': round(float(np.mean(losses[-50:])), 5) if losses else None,
              'clean_control_last50_cross_entropy': round(float(np.mean(clean_losses[-50:])), 5) if clean_losses else None,
              'cells': [], 'native_diagnostic': {}, 'engineering_gates': {}}
    specs_by_cell = {}
    for slots in args.slots:
        for delay in args.delays:
            names = available_names(slots)
            specs = []
            for seed in args.seeds:
                rng = random.Random(seed * 100003 + slots * 991 + delay * 71)
                specs.extend(generate(rng, names, delay) for _ in range(args.trials))
            specs_by_cell[(slots, delay)] = specs
            summary, examples = eval_pair(model, originals, cleaned, specs)
            report['cells'].append({'slots': slots, 'delay': delay, 'trials': len(specs),
                                    'scores': summary, 'examples': examples})
            print(f'Gate4 slots={slots:>2} delay={delay:>3}: '
                  f'original={summary["gate2_reader"]["strict"]:.3f} '
                  f'clean_more={summary["clean_extra_steps"]["strict"]:.3f} '
                  f'calibrated={summary["robust_reader"]["strict"]:.3f} '
                  f'direct_memory={summary["direct_memory_four_color_accuracy"]:.3f}',
                  flush=True)
    idx = {(x['slots'], x['delay']): x['scores'] for x in report['cells']}
    if (32, 500) in idx and (16, 100) in idx:
        hard = idx[32, 500]
        easy = idx[16, 100]
        report['engineering_gates'] = {
            '32x500_calibrated_beats_clean_extra_steps_by_10pp': bool(
                hard['robust_reader']['strict'] >= hard['clean_extra_steps']['strict'] + 0.10),
            '32x500_above_erased_by_30pp': bool(
                hard['robust_reader']['strict'] >= hard['erased']['strict'] + 0.30),
            '16x100_does_not_regress_over_5pp': bool(
                easy['robust_reader']['strict'] >= easy['gate2_reader']['strict'] - 0.05)}

    if args.native_keys:
        native = NativeKeyer(model)
        for slots in args.slots:
            names = available_names(slots)
            modes = {'hash': hashed_keys(names),
                     'native_raw': native.keys(names),
                     'native_centered': native.keys(names, centered=True)}
            for delay in args.delays:
                specs = specs_by_cell[(slots, delay)]
                for mode, keys in modes.items():
                    acc = np.mean([direct_score(make_reads(s, keys), s.truth) for s in specs])
                    report['native_diagnostic'][f'{slots}x{delay}/{mode}'] = {
                        'direct_color_accuracy': round(float(acc), 4),
                        'cosine_geometry': pairwise_cosine(keys)}
                print(f'Native keys slots={slots} delay={delay}: '
                      + ' '.join(f'{m}={report["native_diagnostic"][f"{slots}x{delay}/{m}"]["direct_color_accuracy"]:.3f}'
                                 for m in modes), flush=True)
    return report


def parse_ints(value: str, name: str) -> tuple[int, ...]:
    result = tuple(int(x.strip()) for x in value.split(','))
    if not result or any(x < 1 for x in result) or len(set(result)) != len(result):
        raise ValueError(f'invalid {name} list')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--gain', type=float, default=1.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--checkpoint', default='runs/hf_gate2_adapter.pt')
    p.add_argument('--reader-checkpoint', default='runs/hf_gate4_reader.pt')
    p.add_argument('--clean-checkpoint', default='runs/hf_gate4_clean_control.pt')
    p.add_argument('--report', default='runs/hf_gate4_result.json')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--train-seed', type=int, default=TRAIN_SEED)
    p.add_argument('--lr', type=float, default=0.0007)
    p.add_argument('--slots', default='16,32')
    p.add_argument('--delays', default='100,500')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--seeds', default='813887,997001')
    p.add_argument('--native-keys', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        args.slots = parse_ints(args.slots, 'slots')
        args.delays = parse_ints(args.delays, 'delays')
        args.seeds = parse_ints(args.seeds, 'seeds')
        if any(s < 5 or s > 128 for s in args.slots) or args.trials < 1 or args.steps < 1:
            raise ValueError('slots 5..128; trials and training steps must be positive')
    except ValueError as err:
        p.error(str(err))
    if args.dry_run:
        rng = random.Random(TRAIN_SEED)
        for size in args.slots:
            for delay in args.delays:
                a = generate(rng, names_for_training(size), delay)
                b = generate(rng, available_names(size), delay)
                reads = make_reads(b, hashed_keys(available_names(size)))
                print(f'{size} slots / {delay} writes: train={a.selected}: {a.previous}->{a.truth}; '
                      f'test={b.selected}: {b.previous}->{b.truth}; '
                      f'direct_memory={CODEC.decode(reads["live"])}; bytes=1024')
        return
    report = run(args)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Gate4 engineering gates:', json.dumps(report['engineering_gates'], indent=2))
    print('Saved:', path)


if __name__ == '__main__':
    main()