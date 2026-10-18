# LiveMode — the same stream reads and writes

**An incoming event first asks the current operator what it predicts, then uses what actually arrived to update that operator.** The next event sees the altered computation. There is no backward pass through the pretrained model at inference time, and no need to retain an ever-growing raw history.

This is a *research laboratory*, not a new language model, proof of biological learning, or a memory-compression win. V1 separates a self-contained, reproducible streaming read/write gate from an **experimental** hook into a frozen Hugging Face transformer.

```text
observation (key, value)
         |                         subsequent late query
         v                                |
   read W_t @ key                         v
         |                          read W_current @ key
   prediction / surprise                   |
         |                           memory response
  online local delta write                 |
         |                          transformer layer + response
         v                                |
     W_(t+1)                         next-token logits
```

## Gate 3 — query after interference (frozen adapter)

Gate 2 achieved perfect accuracy on a task where it was always asked about the most recent write. Gate 3 removes that shortcut: four possible facts are rewritten, followed by 10/50/100/500 unrelated updates, and **only after the stream** is a random one of those facts queried. The 1,024-byte state keeps changing online, while both Qwen and the trained Gate 2 lift stay frozen.

The seven arms include current memory, erased memory, a counterfactual missing the queried rewrite, wrong address, last-observation-only, exact true-color code and dictionary fact in prompt. We report full-vocabulary next-token accuracy (primary), four-color accuracy and margins for 8/16/32 addresses. The original checkpoint is reused without further training. **No Qwen Gate 3 results are claimed yet.**

On Windows:

```bat
git pull
python3.13 -m unittest discover -s tests -v
python3.13 -m live_mode.hf_gate3 --dry-run
python3.13 -m live_mode.hf_gate3 --slots 16 --delays 100,500 --trials 4 --seeds 281828 --report runs/hf_gate3_pilot.json
python3.13 -m live_mode.hf_gate3 --trials 20 --report runs/hf_gate3_result.json
```

See [the frozen Gate 3 protocol](docs/GATE3.md). The larger grid is a deliberate memory-capacity stress test; this remains an artificial four-color task with explicit address binding, and indexed dictionaries are more efficient for such facts.

## Run the live flow (CPU, Python 3.10+, NumPy)

```bash
python -m pip install -e .
python -m live_mode.cli
```

At the `live>` prompt:

```text
see alice.color = blue
ask alice.color
see alice.color = red
ask alice.color
see bob.direction = north
ask bob.direction
stats
quit
```

This toy interface uses a **fixed 16-word value vocabulary**, **explicit text address**, and **fixed random-key encoding**, not a learned semantic encoder. It never claims to understand arbitrary natural language. The memory does not store the original observations or a Python dictionary. `observe` uses a *read-before-write*: it returns what the medium thought before the actual value changed it. `ask` reads without writing. The matrix is constant size as events stream in.

## The equation

Let each address become a unit query `k`, and let `v` be the observed value code. At event `t`:

\[
\hat v_t=W_t k_t,\quad e_t=v_t-\hat v_t,\quad
W_{t+1}=\rho W_t+\eta\, \operatorname{clip}(e_t,c) k_t^\top.
\]

`clip` caps the **norm** of the error. A fixed learning rate and optional retention govern stability. A read-only event returns `W_t k_t` and leaves the state byte-for-byte unchanged. These are **test-time updates of a resident matrix**, not gradient training of the shared parameters. The code is under `live_mode/memory.py`.

**Same-event read/write:** the observation reads `W_t` before its target is incorporated; future questions read `W_(t+1)`. This sequencing is intentional: an event cannot use an answer that it just secretly inserted as its own prediction.

## Reproduce the late-query gate

```bash
python -m unittest discover -s tests -v
python benchmarks/run.py
```

Development protocol recorded in `PREREG.md` (not prospectively preregistered): 5 seeds × 40 trials; 12 independently addressed slots; 16 possible categorical values; 96 random updates per trial, and **only then** unpredictable, shuffled queries. The writer receives the current address and actual value at each update, not the future question. After any slot is overwritten, its earlier value is now wrong.

Local development run (NumPy; run the command to reproduce; numbers may vary slightly by version):

| Reader after all updates | Late-query accuracy | Note |
|---|---:|---|
| Online normalized delta operator | **99.92%** | 4,096 resident-state bytes |
| Uncorrected Hebbian outer products | 28.08% | Same matrix size |
| No-write / frozen zero matrix | 6.83% | Chance about 6.25% |
| Last observation only | 8.33% | Very small state |
| Exact keyed dictionary | **100.00%** | **Stronger and vastly cheaper here** |

The experiment demonstrates *functional* online overwrite and late retrieval. It does **not** demonstrate that a matrix is better than a simple dictionary. The dictionary needs only 12 stored value IDs plus its known address convention; the 4,096-byte matrix is grossly inefficient on this task. Current key/value coding and intervention sites are engineered. This gate establishes plumbing and identifies limits, not a new AI architecture.

## Experimental transformer bridge (optional)

```bash
python -m pip install -e '.[transformer]'
python -m live_mode.hf_sidecar --model sshleifer/tiny-gpt2 --layer 0
# or try a local Qwen/HF causal model on a 3060, with a middle-layer index:
python -m live_mode.hf_sidecar --model PATH_TO_LOCAL_HF_MODEL --layer 12
```

`HFSidecar` freezes all pretrained weights. It encodes a key and a true observed value from a chosen layer into a 64-wide shared projection, performs a delta-rule write, then **injects the retrieved vector into the residual stream of that layer** on the next forward pass via a PyTorch hook. The probe runs a matched with/without intervention forward pass and reports changes in next-token logits. The keys are supplied explicitly; Qwen's model layout is supported if it exposes `model.layers`, and GPT-2 via `transformer.h`.

**Important:** This is a causal transport and compatibility test, not a trained adapter or evidence of better answer quality. The current random projection and untrained lift will often change logits without improving answers. `device_map=auto` may require `accelerate`. The external model and tokenizer must be downloaded separately. Nothing is sent to an API. The CPU tests do not download a model.

## What next would count as a real win?

Gate 2: learn the query/value interface and residual injection on independent training streams, then **freeze the adapter**. On held-out streams, update **only** resident state. Test actual generated answers on late, unknown-at-write-time questions and compare to a matched-byte key/value cache, full-context baseline, and a no-write control. Require evidence that information was not just cached in the prompt and that residual intervention causally changes the answer.

Gate 3: mix correlated and previously unseen entity relations, conflicting evidence, noise, delays, and repeated zero-error observations. Test whether write selection prevents interference; compare memory bytes **and compute** to straightforward retrieval. Only if a matrix is competitive do we test phase-graph geometry or physical-like growth.

### Related research thread

- [FridayRepo](https://github.com/anttiluode/FridayRepo): state-dependent sender waveform × receiver susceptibility; warns that predictable questions let an RNN precompute the answer.
- [AdaptiveObserverCache](https://github.com/anttiluode/AdaptiveObserverCache): changing a frozen transformer's read direction does not guarantee changed final behavior.
- [SolsBaby](https://github.com/anttiluode/SolsBaby): activity writes conductance; an identical later probe reads altered history.
- [PhaseGraphFastWeightCell](https://github.com/anttiluode/PhaseGraphFastWeightCell): selective online overwrites are harder than simply bounding phase state.
- Martin-Burgos et al. (2026), [*Action potential waveforms are state-dependent*](https://doi.org/10.64898/2026.09.15.751814): evidence for history-dependent biological waveforms, **not** validation of this artificial learning rule.

MIT licensed (see repository LICENSE).

## Gate 2: train the live-memory-to-Qwen interface (new; Qwen run pending)

The original random projection moved Qwen logits but did not change its chosen next token. **Gate 2 trains only a small residual-stream lift**, with Qwen fully frozen, using short read-then-write experience streams. Train names and held-out test names do not overlap. The test changes one color at a known address, then asks an unpredictable late question. The operator keeps only **1,024 bytes** (4 colors × 64 address dimensions, float32), with a fixed supplied symbolic address binder.

Run on your RTX 3060:

```bat
python3.13 -m pip install -e ".[transformer]" accelerate
python3.13 -m live_mode.hf_gate2 --model Qwen/Qwen2.5-0.5B-Instruct --layer 12 --steps 300 --trials 80
```

Outputs: `runs/hf_gate2_adapter.pt` and `runs/hf_gate2_result.json`. The report includes strict full-vocabulary next-token accuracy, four-color restricted accuracy, stale/erased/wrong-address and oracle controls, plus a dictionary-in-context baseline. See [Gate 2 protocol](docs/GATE2.md). **The Qwen accuracy result has not been measured yet.** Passing would demonstrate a trained interface on this simple synthetic task, not general semantic learning or a win over retrieval.


## Gate 4 — Reader alignment after interference (GPU results pending)

Gate 4 compares the original Gate 2 reader with two equally trained alternatives: extra clean training and interference-heavy training. The Qwen backbone stays frozen and the per-episode memory remains **1,024 bytes**. It separately measures direct-matrix color recall, and optionally compares hash keys with raw and mean-centered Qwen activation-derived keys. Neither approach implements the Jacobian lens or autonomous language fact extraction.

Full frozen design, controls, limits and commands: [Gate 4 protocol](docs/GATE4.md).

Windows quick check: `git pull`, `python3.13 -m unittest discover -s tests -v`, then `python3.13 -m live_mode.hf_gate4 --dry-run`.

Short GPU pilot: `python3.13 -m live_mode.hf_gate4 --steps 60 --trials 2 --seeds 813887 --report runs/hf_gate4_pilot.json --reader-checkpoint runs/hf_gate4_pilot_reader.pt --clean-checkpoint runs/hf_gate4_pilot_clean.pt`.

Full experiment: `python3.13 -m live_mode.hf_gate4 --steps 300 --trials 20 --native-keys --report runs/hf_gate4_result.json`.

**No Gate 4 Qwen results are known yet.** This test deliberately allows failure: better decoding cannot recover information destroyed in the memory matrix. The exact dictionary remains the stronger simplicity/efficiency baseline for these tiny categorical tasks.
