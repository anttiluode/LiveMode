# Gate 2 — learned interface to a frozen transformer

**Status:** Gate 2 code and local synthetic/mock-gradient tests are implemented. **Qwen performance has not been measured.** This experimental protocol is recorded before the first Gate 2 Qwen result; it is not a claim of independent preregistration.

## Question

Gate 1 gave nonzero Qwen logit changes after online memory writes, but the top token remained ` Alice`. An untrained random lift only perturbs a model. Can a *trained, then frozen* lift make a stream-updated operator drive the **correct next token** after a change of fact?

## Contract

- Freeze all original transformer parameters.
- Use a 64-dimension hashed **explicit address** and four supplied one-hot colors: red, blue, green, yellow. This experiment does **not** learn semantic addressing or arbitrary words.
- An episode observes four randomly assigned colors at four names, then may rewrite one. Only the matrix state (4 × 64 float32 = **1024 bytes**) updates during that episode.
- Train a small memory-value-to-residual linear lift on 16 training names, using only cross-entropy of the real frozen LM's next-token logits. No gradients update the transformer.
- Test with **16 disjoint held-out names**, 80 episodes, each with a rewrite and a late unknown-at-write-time question. Reinitialize memory between independent episodes; this is **not** a long-horizon continuous deployment.
- Evaluate original LM output over the entire vocabulary (strict next-token match) as well as ranking among four colors (an easier diagnostic).

Controls measured on the *same queries*: `erased` (no injected memory), `stale` (pre-rewrite operator), `wrong_address` (another key with a different color), `oracle_code` (correct one-hot directly, tests lift independent of retrieval), and `dictionary_context` (the true fact literally inserted into text; a strong conventional retrieval baseline, but a different budget).

## Engineering success gates, set ahead of Qwen run

1. `live` strict next-token accuracy at least **60%** (chance across the four labels 25%).
2. `erased` and `wrong_address` strict next-token accuracy each at most **35%**.
3. The fraction where `live` is correct while `stale` is wrong at least **45%**.
4. No evaluation names, observations, or matrices used to train the lift.

These are thresholds for this contrived gate, not statistical significance or broad language-model improvement. Always report failed thresholds and control values. Training uses 300 updates by default; changes prompted by test outcomes require a new dev/test split.

## Windows RTX 3060

```bat
git pull
python3.13 -m pip install -e ".[transformer]" accelerate
python3.13 -m unittest discover -s tests -v
python3.13 -m live_mode.hf_gate2 --model Qwen/Qwen2.5-0.5B-Instruct --layer 12 --steps 300 --trials 80
```

Writes `runs/hf_gate2_adapter.pt` (calibrated lift) and `runs/hf_gate2_result.json` (readout, controls, sample predictions). Repeat evaluation from the adapter checkpoint without retraining:

```bat
python3.13 -m live_mode.hf_gate2 --eval-only --model Qwen/Qwen2.5-0.5B-Instruct --layer 12 --trials 80
```

The checkpoint contains only the lift and metadata; no untrusted Python pickle is needed to load it (weights-only load).

## Scientific boundaries

A pass would show that a trained external readout can make a frozen LM respond to small stream-written matrix state in a four-color artificial task. It would **not** establish biology, improved memory efficiency over a dictionary, general-language facts, novel online optimization mathematics, semantic identity resolution, interference robustness over long streams, or fully simultaneous training of the pretrained transformer. The model's own weights remain unchanged throughout; within held-out experience, only the operator learns.

The original `hf_sidecar.py` remains unchanged for independent intervention checks.
