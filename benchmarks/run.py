"""Predeclared late-query synthetic experiment. Run: python benchmarks/run.py."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_mode import VectorCodec, LiveStream

SEEDS = (41, 97, 127, 223, 509)
KEYS = tuple(f'sensor_{i}/state' for i in range(12))
VALUES = tuple(f'value_{i}' for i in range(16))
N_UPDATES = 96
N_TRIALS = 40


def trial(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    codec = VectorCodec(VALUES, key_dim=64, seed=seed)
    live = LiveStream(codec, learning_rate=0.85)
    hebb = np.zeros_like(live.memory.W)
    truth: dict[str, str] = {}
    last: tuple[str, str] | None = None
    # The reader does not know which address will be queried later.
    for i in range(N_UPDATES):
        key = KEYS[int(rng.integers(len(KEYS)))]
        val = VALUES[int(rng.integers(len(VALUES)))]
        live.observe(key, val)
        k = codec.key(key)
        v = codec.value(val)
        hebb += 0.85 * np.outer(v, k)
        truth[key] = val
        last = key, val
    questions = list(truth.keys())
    rng.shuffle(questions)
    results = {'delta': 0, 'hebb': 0, 'frozen': 0, 'last_only': 0, 'dictionary': 0}
    for key in questions:
        expected = truth[key]
        delta, _ = live.ask(key)
        guess_hebb = codec.decode(hebb @ codec.key(key))
        results['delta'] += int(delta == expected)
        results['hebb'] += int(guess_hebb == expected)
        results['frozen'] += int(VALUES[0] == expected)
        results['last_only'] += int(last == (key, expected))
        results['dictionary'] += 1
    return {name: value / len(questions) for name, value in results.items()}


def main() -> None:
    rows = [trial(seed * 100000 + t) for seed in SEEDS for t in range(N_TRIALS)]
    result = {
        'protocol': {'seeds': SEEDS, 'trials_per_seed': N_TRIALS,
                     'updates_per_trial': N_UPDATES, 'keys': len(KEYS),
                     'values': len(VALUES), 'state_bytes_delta': 64 * 16 * 4,
                     'dictionary_minimum_value_bytes': len(KEYS),
                     'query_timing': 'all late; shuffled; never given to writer',
                     'note': 'Dictionary is an explicit, stronger, cheaper baseline, not a matched-parameter model.'},
        'mean_accuracy': {name: round(float(np.mean([r[name] for r in rows])), 6)
                          for name in rows[0]},
        'std_across_trials': {name: round(float(np.std([r[name] for r in rows])), 6)
                              for name in rows[0]},
    }
    print(json.dumps(result, indent=2))
    out = Path('runs/benchmark.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
