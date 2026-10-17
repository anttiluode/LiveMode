"""Optional gradient-routing test with a frozen torch toy network. No HF download."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from live_mode.hf_gate2 import FrozenLMAdapter


@unittest.skipIf(torch is None, 'PyTorch is an optional dependency')
class HookTests(unittest.TestCase):
    def test_only_adapter_receives_gradients(self):
        class Tokenizer:
            def __call__(self, text, return_tensors='pt'):
                return {'input_ids': torch.tensor([[3, 4, 5]])}
        class Block(torch.nn.Module):
            def forward(self, x):
                return x + torch.tanh(x)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(10, 8)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Block()])
                self.head = torch.nn.Linear(8, 10)
            def forward(self, input_ids, use_cache=False):
                h = self.model.layers[0](self.emb(input_ids))
                return type('Output', (), {'logits': self.head(h)})
        a = FrozenLMAdapter.__new__(FrozenLMAdapter)
        a.torch = torch
        a.tokenizer = Tokenizer()
        a.model = Model()
        a.model.requires_grad_(False)
        a.layers = a.model.model.layers
        a.layer_index = 0
        a.device = torch.device('cpu')
        a.gain = 1.
        a.lift = torch.nn.Linear(4, 8, bias=False)
        torch.nn.init.zeros_(a.lift.weight)
        before = a._logits('demo', None).detach()
        output = a._logits('demo', np.array([0, 1, 0, 0], dtype='float32'))
        torch.testing.assert_close(output.detach(), before)
        (-output[1]).backward()
        self.assertGreater(float(a.lift.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in a.model.parameters()))
        with torch.no_grad():
            a.lift.weight -= .4 * a.lift.weight.grad
        after = a._logits('demo', np.array([0, 1, 0, 0], dtype='float32')).detach()
        self.assertGreater(float(torch.linalg.norm(after - before)), 0.0001)
