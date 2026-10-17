# Gate 1: continuous local read/write, with late unpredictable questions

**Development protocol, recorded after the first exploratory run. Not a prospective preregistration.** The experiment is intentionally synthetic and deliberately gives the writer explicit keys and ground-truth values. It does not test semantic understanding or transformer quality.

## Task

- 5 seeds: 41, 97, 127, 223, 509; 40 independent trials per seed.
- 12 fixed possible keys. All address vectors are pseudo-random, fixed by text+trial seed and 64-dimensional.
- 16 possible labels, represented by orthogonal one-hot value codes.
- 96 independent uniformly sampled `(key, value)` updates. The same key may be reassigned. A target is revealed only at write time.
- All 12 keys are queried **after** the final update, in an independently shuffled order unknown to the writer. The correct answer is each key's *last* value.

## Arms and costs

1. **Delta:** `W <- W + .85 (v - Wk) k^T`, 16 x 64 float32 = 4096 state bytes. Each observation first reads its old prediction. Maximum error norm clip = 2.0; it does not activate for one-hot residuals in the ordinary run. No decay (rho=1).
2. **Hebb:** `W <- W + .85 v k^T`, same byte size, no negative feedback.
3. **Frozen:** all-zero matrix, no writes.
4. **Last-only:** remembers just the last key/value pair, not an operator.
5. **Dictionary:** perfect explicit keyed last-value map; state is *much smaller* here, the relevant economic attacker.

Metric: average per-trial exact value accuracy, plus standard deviation across 200 trials. All models use the same streams and no test labels are fed back after late questions.

## Expectations, falsifiers and claims

- Delta > frozen and Hebb under repeated overwrites. If it fails, the basic implementation has no case here.
- Correctness under exact dictionary is 100%; **this deliberately prevents any claim of state-efficiency advantage** for delta memory.
- A same-key correction should measurably overwrite the old response (unit test).
- Read-only probes must not mutate the operator; save/load must preserve state; no hidden history/cache can enter predictions.
- Gate 1 is only an engineering validation. Passing does not establish biological plausibility, emergent architecture, language capabilities, autonomous self-supervision or superiority over retrieval.

## Next pre-registration (not part of Gate 1)

For a real-model Gate 2: train the *shared* interface on source streams only, lock it before evaluation, then do live **resident-state-only** writes at test time. Score generated answers and a same-budget retrieval attacker, measure contamination/overwrites and intervention effects. Do not use the logit-change-only transformer bridge as a substitute for this evaluation.