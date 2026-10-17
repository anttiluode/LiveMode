"""Gate 2: train a small readout of an online-updated matrix into a frozen LM.

The external address binder and four color codes are supplied. The pretrained
model never changes. This is a controlled answer-changing gate, not an agent.
Run on local GPU: python -m live_mode.hf_gate2 --steps 300 --trials 80
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import numpy as np
from .memory import LiveMemory, Settings
from .stream import VectorCodec

COLORS = ('red', 'blue', 'green', 'yellow')
TRAIN_NAMES = ('Alice', 'Bob', 'Carol', 'Dave', 'Eva', 'Frank', 'Grace', 'Henry',
               'Ingrid', 'Jack', 'Kira', 'Liam', 'Mary', 'Nora', 'Oscar', 'Pia')
TEST_NAMES = ('Quinn', 'Ruth', 'Sven', 'Tessa', 'Uma', 'Vera', 'Wendy', 'Xavier',
              'Yara', 'Zane', 'Aino', 'Bela', 'Cleo', 'Dara', 'Eero', 'Faye')
PROMPTS = ('What color is {name}? Answer:',
           'Question: What is the color of {name}? Answer:',
           'Give only the color of {name}. Answer:')


@dataclass
class Episode:
    prompt: str
    address: str
    truth: str
    previous: str
    response: np.ndarray
    stale_response: np.ndarray
    distractor_response: np.ndarray


def episode(rng, names, codec, rewrite=True):
    """Four experiences, optionally one overwrite, and an unpredictable late query."""
    if len(names) < 4:
        raise ValueError('need four distinct names')
    chosen = rng.sample(list(names), k=4)
    values = rng.sample(list(COLORS), k=4)
    memory = LiveMemory(Settings(key_dim=codec.key_dim, value_dim=codec.value_dim,
                                 learning_rate=.95, max_update_norm=3.0))
    for name, value in zip(chosen, values):
        memory.step(codec.key(name + '.color'), codec.value(value))
    i = rng.randrange(4)
    name, old = chosen[i], values[i]
    address = name + '.color'
    stale = memory.read(codec.key(address))
    new = rng.choice([v for v in COLORS if v != old]) if rewrite else old
    if rewrite:
        memory.step(codec.key(address), codec.value(new))
    response = memory.read(codec.key(address))
    alternatives = [j for j in range(4) if j != i and values[j] != new]
    wrong = memory.read(codec.key(chosen[rng.choice(alternatives)] + '.color'))
    return Episode(rng.choice(PROMPTS).format(name=name), address, new, old,
                   response, stale, wrong)


class FrozenLMAdapter:
    """Only lift.weight is trainable. The base model is completely frozen."""

    def __init__(self, model_name, layer_index, device='auto', gain=1., seed=17):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        options = {'dtype': 'auto'}
        if device == 'auto':
            options['device_map'] = 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **options).eval()
        self.model.requires_grad_(False)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.layers = self.model.transformer.h
        else:
            raise ValueError('unsupported transformer layer layout')
        if not 0 <= layer_index < len(self.layers):
            raise ValueError('layer out of range')
        self.layer_index = layer_index
        self.device = self.model.get_input_embeddings().weight.device
        self.gain = gain
        hidden = (self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size')
                  else self.model.config.n_embd)
        self.lift = torch.nn.Linear(len(COLORS), int(hidden), bias=False,
                                    device=self.device, dtype=torch.float32)
        torch.nn.init.zeros_(self.lift.weight)
        self.label_ids = {}
        for color in COLORS:
            tokens = self.tokenizer.encode(' ' + color, add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f'{color!r} needs more than one next-token ID: {tokens!r}')
            self.label_ids[color] = tokens[0]
        if len(set(self.label_ids.values())) != len(COLORS):
            raise ValueError('non-unique color tokens')

    def _logits(self, prompt, readout):
        torch = self.torch
        inputs = {k: v.to(self.device) for k, v in
                  self.tokenizer(prompt, return_tensors='pt').items()}
        handle = None
        if readout is not None:
            value = torch.as_tensor(readout, dtype=torch.float32, device=self.device)
            delta = self.lift(value) * self.gain
            def hook(_module, _args, result):
                h = result[0] if isinstance(result, tuple) else result
                mask = torch.zeros((1, h.shape[1], 1), device=h.device, dtype=h.dtype)
                mask[:, -1, :] = 1
                updated = h + mask * delta.to(h.dtype).view(1, 1, -1)
                return (updated, *result[1:]) if isinstance(result, tuple) else updated
            handle = self.layers[self.layer_index].register_forward_hook(hook)
        try:
            return self.model(**inputs, use_cache=False).logits[0, -1].float()
        finally:
            if handle is not None:
                handle.remove()

    def train(self, steps, lr=.005, seed=1776, print_every=50):
        torch = self.torch
        rng = random.Random(seed)
        codec = VectorCodec(COLORS, key_dim=64, seed=20261017)
        optimizer = torch.optim.AdamW(self.lift.parameters(), lr=lr, weight_decay=0)
        losses = []
        for i in range(steps):
            ep = episode(rng, TRAIN_NAMES, codec, rewrite=(i % 3 != 0))
            optimizer.zero_grad(set_to_none=True)
            logits = self._logits(ep.prompt, ep.response)
            label = torch.tensor([self.label_ids[ep.truth]], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lift.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if print_every and (i + 1) % print_every == 0:
                print(f'train {i+1}/{steps}: recent cross entropy {np.mean(losses[-print_every:]):.3f}', flush=True)
        return losses

    def evaluate(self, trials, seed=191919):
        torch = self.torch
        rng = random.Random(seed)
        codec = VectorCodec(COLORS, key_dim=64, seed=20261017)
        arms = ('live', 'erased', 'stale', 'wrong_address', 'oracle_code', 'dictionary_context')
        results = {a: {'strict': 0, 'choice': 0, 'margins': []} for a in arms}
        changed = []
        examples = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(trials):
                ep = episode(rng, TEST_NAMES, codec, rewrite=True)
                reads = {'live': ep.response, 'erased': None, 'stale': ep.stale_response,
                         'wrong_address': ep.distractor_response,
                         'oracle_code': codec.value(ep.truth), 'dictionary_context': None}
                guesses = {}
                for arm, read in reads.items():
                    prompt = ep.prompt
                    if arm == 'dictionary_context':
                        name = ep.address.removesuffix('.color')
                        prompt = f'Fact: {name} has the color {ep.truth}.\n' + prompt
                    logits = self._logits(prompt, read)
                    ids = [self.label_ids[v] for v in COLORS]
                    candidates = logits[torch.as_tensor(ids, device=logits.device)]
                    ix = COLORS.index(ep.truth)
                    guesses[arm] = COLORS[int(candidates.argmax())]
                    results[arm]['strict'] += int(int(logits.argmax()) == self.label_ids[ep.truth])
                    results[arm]['choice'] += int(guesses[arm] == ep.truth)
                    others = torch.cat((candidates[:ix], candidates[ix+1:]))
                    results[arm]['margins'].append(float((candidates[ix] - others.max()).cpu()))
                changed.append(int(guesses['live'] == ep.truth and guesses['stale'] != ep.truth))
                if len(examples) < 8:
                    examples.append({'prompt': ep.prompt, 'previous': ep.previous,
                                     'new': ep.truth, 'color_predictions': guesses})
        return {'trials': trials, 'all_rewrites': True,
                'heldout_names': list(TEST_NAMES),
                'scores': {a: {
                    'strict_next_token_accuracy': round(d['strict'] / trials, 4),
                    'four_color_accuracy': round(d['choice'] / trials, 4),
                    'mean_true_color_margin': round(float(np.mean(d['margins'])), 4)}
                    for a, d in results.items()},
                'live_correct_stale_wrong_rate': round(float(np.mean(changed)), 4),
                'examples': examples,
                'warning': 'Four-color choice is an easier diagnostic. Dictionary context explicitly includes the fact.'}


def main():
    p = argparse.ArgumentParser(description='Frozen Qwen learned LiveMode bridge')
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--device', default='auto')
    p.add_argument('--gain', type=float, default=1.)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--trials', type=int, default=80)
    p.add_argument('--checkpoint', default='runs/hf_gate2_adapter.pt')
    p.add_argument('--report', default='runs/hf_gate2_result.json')
    p.add_argument('--eval-only', action='store_true')
    args = p.parse_args()
    model = FrozenLMAdapter(args.model, args.layer, args.device, args.gain)
    path = Path(args.checkpoint)
    if args.eval_only:
        state = model.torch.load(path, map_location='cpu', weights_only=True)
        if (state['model'], state['layer'], state['gain']) != (args.model, args.layer, args.gain):
            raise ValueError('checkpoint belongs to different model/layer/gain')
        model.lift.load_state_dict(state['adapter'])
    else:
        if args.steps <= 0:
            p.error('steps must be positive except with --eval-only')
        model.train(args.steps, args.lr)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.torch.save({'model': args.model, 'layer': args.layer,
                          'gain': args.gain, 'adapter': model.lift.state_dict()}, path)
    report = model.evaluate(args.trials)
    report['configuration'] = {'model': args.model, 'layer': args.layer, 'gain': args.gain,
                               'train_steps': None if args.eval_only else args.steps,
                               'trained_adapter_parameters': int(model.lift.weight.numel()),
                               'resident_bytes': 4 * 64 * 4, 'frozen_pretrained_model': True,
                               'explicit_address_binder': True, 'values': list(COLORS)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved report:', output)


if __name__ == '__main__':
    main()
