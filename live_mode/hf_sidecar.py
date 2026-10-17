"""Optional *experimental* Hugging Face causal hook, not a trained language adapter.

Pretrained model weights stay frozen. External, keyed experiences update a
64x64 matrix; at inference the retrieved value is lifted into one transformer
layer's residual stream. The current implementation proves causal injection,
not semantic question-answer improvement.

Requires: pip install -e '.[transformer]'
"""
from __future__ import annotations
import argparse
import numpy as np
from .memory import LiveMemory, Settings


class HFSidecar:
    def __init__(self, model_name: str, layer_index: int = 5, width: int = 64,
                 gain: float = 0.25, device: str = 'auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        opts = {'torch_dtype': 'auto'}
        if device == 'auto':
            opts['device_map'] = 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **opts).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self._layers()
        if not 0 <= layer_index < len(self.layers):
            raise ValueError(f'layer index must be in [0, {len(self.layers)-1}]')
        self.layer_index = layer_index
        self.gain = gain
        hidden = int(self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size')
                     else self.model.config.n_embd)
        rng = np.random.default_rng(20261017)
        proj = rng.standard_normal((width, hidden)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)
        self.proj = proj
        self.memory = LiveMemory(Settings(key_dim=width, value_dim=width, learning_rate=.8))

    def _layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise ValueError('unsupported model layout: expected model.layers or transformer.h')

    def _model_input(self, text: str):
        tokens = self.tokenizer(text, return_tensors='pt')
        device = self.model.get_input_embeddings().weight.device
        return {key: val.to(device) for key, val in tokens.items()}

    def _encode(self, text: str) -> np.ndarray:
        with self.torch.inference_mode():
            output = self.model(**self._model_input(text), output_hidden_states=True,
                                use_cache=False)
            h = output.hidden_states[self.layer_index + 1][0, -1].float().cpu().numpy()
        code = self.proj @ h
        return code / max(float(np.linalg.norm(code)), 1e-8)

    def observe(self, address: str, target_text: str):
        """Teacher-forced factual observation: both address and truth arrive externally."""
        key = self._encode(address)
        val = self._encode(target_text)
        return self.memory.step(key, val)

    def probe(self, prompt: str, address: str) -> dict:
        """Run the same prompt with intervention off/on and compare next logits."""
        query = self._encode(address)
        retrieved = self.memory.read(query)
        lift = self.proj.T @ retrieved
        tokens = self._model_input(prompt)

        def get_logits(scale: float):
            def hook(_module, _inputs, output):
                old = output[0] if isinstance(output, tuple) else output
                new = old.clone()
                delta = self.torch.as_tensor(lift, dtype=old.dtype, device=old.device)
                new[:, -1, :] += delta * scale
                return (new, *output[1:]) if isinstance(output, tuple) else new
            handle = self.layers[self.layer_index].register_forward_hook(hook)
            try:
                with self.torch.inference_mode():
                    return self.model(**tokens, use_cache=False).logits[0, -1].float().cpu()
            finally:
                handle.remove()

        baseline = get_logits(0)
        injected = get_logits(self.gain)
        return {
            'logit_l2_change': float(self.torch.norm(injected - baseline)),
            'baseline_token': self.tokenizer.decode(int(baseline.argmax())),
            'live_token': self.tokenizer.decode(int(injected.argmax())),
            'state_bytes': self.memory.state_bytes,
            'note': 'A logit change is not evidence of improved answers.',
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='sshleifer/tiny-gpt2')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--gain', type=float, default=0.25)
    args = p.parse_args()
    sidecar = HFSidecar(args.model, layer_index=args.layer, device=args.device, gain=args.gain)
    print('Teaching: alice.color -> blue')
    sidecar.observe('alice.color', 'blue')
    print(sidecar.probe('What color is Alice? Answer:', 'alice.color'))
    print('Rewriting: alice.color -> red')
    sidecar.observe('alice.color', 'red')
    print(sidecar.probe('What color is Alice? Answer:', 'alice.color'))


if __name__ == '__main__':
    main()
