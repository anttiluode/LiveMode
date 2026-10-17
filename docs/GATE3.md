# Gate 3 — late query after interfering overwrites

**Status before GPU evaluation:** Synthetic generator and tests have passed locally. No Gate 3 Qwen results have been measured. This protocol and its criteria were recorded after Gate 2, before the first Gate 3 Qwen run; it is not independent preregistration.

## Rationale

Gate 2 always asked about the most recently rewritten address. A last-event-only model could have solved it without associative memory. Gate 3 removes this shortcut. The frozen Qwen backbone and frozen 3,584-parameter Gate 2 lift are reused without further training. Only the resident **4 x 64 float32 = 1,024-byte matrix** changes online.

## Experiment

For every independent episode:

1. Initialize N different hashed addresses with independently sampled colors. Default N = 8, 16, or 32. The first 16 names were held out from Gate 2 training; larger tests introduce further names.
2. Rewrite four different, randomly chosen addresses to a different color. Any one of these four can be the final queried address.
3. Continue 10, 50, 100 or 500 updates to **other addresses**, with repeated overwrites. The four candidates are intentionally left untouched during the suffix to test *retention amid interference*. The most recent event is never about the queried address.
4. Only **after the entire sequence** choose one of the four candidate addresses at random and ask for its present color. The stream writer never receives the future query.

Memory itself retains only its fixed matrix. The evaluation harness keeps the list of experiences to construct the stale-memory counterfactual and an exact dictionary baseline; the memory writer does not see this history. Memory resets between episodes; this is not a single long-lived user session.

## Seven arms

- live: query the updated matrix at the correct address.
- erased: original frozen model, no injection.
- stale: all the same events, except omit the last rewrite of the queried address.
- wrong_address: read another address with a different correct color.
- last_only: inject just the most recent observation's value, **no associative query**. This is a generous last-event baseline.
- oracle_code: inject the exact true one-hot color; the exact indexed dictionary equivalent when passed through the same lift.
- dictionary_context: give the model the exact truth as ordinary text context with no intervention; this adds tokens and is not byte matched.

Report unrestricted-vocabulary strict next-token accuracy (primary metric), four-color-restricted accuracy, and color logit margin for every arm in **every** load/delay cell.

## Default settings and frozen engineering gates

    model: Qwen/Qwen2.5-0.5B-Instruct
    layer: 12
    gain: 1.0
    checkpoint: runs/hf_gate2_adapter.pt
    writer eta: 0.95; max_update_norm: 3.0
    key_dim: 64; values: red/blue/green/yellow
    slots: 8,16,32
    interfering writes: 10,50,100,500
    seeds: 281828,314159
    trials per seed per cell: 20
    total: 480 independent synthetic episodes
    trainable Gate 3 parameters: 0

Gate 3A: **16 slots, 100 interfering writes** — live strict accuracy at least 80% AND 30 percentage points above last_only.

Gate 3B: **16 slots, 500 interfering writes** — live strict accuracy at least 60% AND 30 percentage points above last_only.

The 8/32-slot cells have no pass threshold: they are capacity stress tests. Criteria are engineering thresholds, not p-values.

## Windows RTX 3060

Synthetic and mock tests, no Qwen download:

    git pull
    python3.13 -m unittest discover -s tests -v
    python3.13 -m live_mode.hf_gate3 --dry-run

Exploratory eight-episode GPU pilot, not the frozen full protocol:

    python3.13 -m live_mode.hf_gate3 --slots 16 --delays 100,500 --trials 4 --seeds 281828 --report runs/hf_gate3_pilot.json

Complete grid, no new training (expect substantial model inference):

    python3.13 -m live_mode.hf_gate3 --trials 20 --report runs/hf_gate3_result.json

Output reports per-cell accuracy, controls, selected examples, and engineering pass/fail. The existing adapter checkpoint is not modified.

## Limitations

All addresses are provided as hashed IDs; values are supplied as one-hot color codes. Qwen does not extract facts from raw language, change its pretrained weights, or develop new architecture. The 1,024-byte matrix is not demonstrated to be more efficient than a tiny exact dictionary. The challenge here is **selective readout after interference**, not natural-language learning or biological validation.
