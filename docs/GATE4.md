# Gate 4 — Does the reader understand the memory it receives?

**Status:** committed design, source, unit tests, and CPU diagnostic. **No Gate 4 Qwen accuracy is yet measured.** This is a research hypothesis, not independent preregistration. Gate 3, its checkpoints, code, and receipts remain unchanged.

## Two separate questions inspired by the recent interpretability discussions

1. **Reader alignment under interference (primary).** Can a reader trained to interpret noisy, multiply overwritten 1,024-byte associative state outperform one trained only on clean, short episodes? Existing Gate 2's linear `lift` has 3,584 parameters. All Qwen backbone parameters remain frozen. Only the external `W` matrix changes when a test stream is running. This is *not* a J-space/Jacobian lens experiment or an autonomous language write mechanism.
2. **Reader-native key geometry (optional diagnostic).** What happens if the 64-dimensional random SHA key is replaced by a deterministic 64-dimensional projection of **Qwen's own** layer-12 representation of the explicitly supplied name, before or after centering it using **only training names**? Compare with original hash keys at exactly 1,024 resident bytes. This is a direct matrix-retrieval diagnostic, not semantic binding or proof that Qwen's native representations are better keys. It requires Qwen inference to extract keys, so compute and fixed-projection bytes must be tracked separately.

Cammarata's "become another reader" motivates measuring how state is interpreted rather than presuming a random code is already optimal. The Anthropic J-space study suggests that some internal directions are unusually reusable by downstream computations. **Neither paper validates the proposed Gate 4 mechanism.** Claude's separate J-lens/native intervention implementation can be evaluated independently in RealTimeReadWrite.

## Primary controlled comparison

Start from the identical trained Gate 2 checkpoint `runs/hf_gate2_adapter.pt` in both cases. Use an additional equal number of gradient updates and the same optimizer hyperparameters in two independent branches:

- `clean_extra_steps`: train on familiar Gate-2-style short clean episodes, using exclusively `TRAIN_NAMES`.
- `robust_reader`: train on episodes with 8, 16 or 32 addresses, 10/100/500 interfering writes and an unpredictable late query, using exclusively `TRAIN_NAMES` and `CalibName...` synthetic extras.

Both readers have the *same 3,584 free adapter parameters* and the same 1,024-byte W writer at evaluation. The robust reader does **not** learn new pretrained Qwen weights, a new writer, different key embeddings or extra state. Training targets are supervised color tokens; no hidden Qwen reasoning or autonomous write policy is implied.

Compare *paired identical* held-out episodes, never used by either calibration arm, across:

- `gate2_reader`: original checkpoint before extra training.
- `clean_extra_steps`: same training budget as robust on easy streams.
- `robust_reader`: extra training on interference streams.
- `erased`, `stale`, `wrong_address`, `last_only`, `oracle_code`, and `dictionary_context` controls, scored with robust reader.
- `direct_memory_four_color_accuracy`: argmax of the actual 4-vector from W *before Qwen*, diagnosing whether information survived in the memory at all.

**Train/test separation:** training uses Alice...Pia plus `CalibName...` synthetic names; evaluation uses Quinn...Faye plus `Novice...`. They have no overlap. Training random seed `729011`; evaluation seeds `813887`, `997001`, distinct from Gate 3. The question is chosen *after* every full observation stream; memory resets per episode, as in Gate 3.

**Declared grid:** 16 and 32 slots × 100 and 500 intervening writes × 20 trials × 2 test seeds = 160 paired evaluation episodes. Strict unrestricted next-token color accuracy is primary. Four-color accuracy and true-color margin are reported as secondary metrics.

**Hypothesis gates (not measured yet):**

- On 32-slot/500-write tests, `robust_reader` exceeds **clean_extra_steps by at least 10 percentage points**.
- In that same cell, `robust_reader` is **at least 30 points above erased**.
- On 16-slot/100-write tests, `robust_reader` doesn't trail original `gate2_reader` by more than 5 points.

The source also reports direct-memory accuracy. A good Qwen reader cannot recover a label destroyed in W; low direct-memory accuracy is evidence against interpreting a poor Qwen result as a decoding-only failure. At 40 trials/cell uncertainty remains considerable. Exact dictionary may be much cheaper than W—no storage/compression win is presumed.

## Windows RTX 3060

```bat
cd E:\DocsHouse\1018\LiveMode
git pull
python3.13 -m unittest discover -s tests -v
python3.13 -m live_mode.hf_gate4 --dry-run

REM Small 8-episode pilot. Do NOT treat as final research result.
python3.13 -m live_mode.hf_gate4 --steps 60 --trials 2 --seeds 813887 --report runs/hf_gate4_pilot.json --reader-checkpoint runs/hf_gate4_pilot_reader.pt --clean-checkpoint runs/hf_gate4_pilot_clean.pt

REM Full primary experiment; both reader curricula get 300 extra training steps.
python3.13 -m live_mode.hf_gate4 --steps 300 --trials 20 --native-keys --report runs/hf_gate4_result.json
```

`--native-keys` adds an independent address-geometry comparison, with fixed seeded projection of Qwen intermediate hidden states, a centering mean estimated on training names, and no new learned weights. Omit it to reduce run time without changing the main reader evaluation.

`--eval-only` reloads both reader checkpoints and reruns held-out evaluation with frozen reader and writer rules; it does not retrain. The result includes every cell's paired scores, configuration, and honest pass/fail criteria.

## Critical limits

- **Neither method learns how to identify the entity in arbitrary natural-language observations.** The binder receives the entity name explicitly; the four colors remain manually coded one-hot values.
- The robust reader is recalibrated *offline* using supervised next-token loss before test. Only W updates online at test time; this is not live weight optimization within Qwen.
- "Qwen-native" keys are still supplied with a handcrafted textual template and projected with a fixed random matrix. They are not Anthropic's J-space.
- Equal **mutable** byte counts do not mean equal model size, scaffold weights, training compute or key-generation overhead; all are named separately.
- The project has not demonstrated general reasoning through the workspace, biological neuron-like spikes, spontaneous specialization, growth, or an efficiency advantage over exact retrieval.