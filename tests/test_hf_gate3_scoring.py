"""Mock-scoring integration test for Gate 3; no actual HF model download."""
import unittest
try:
    import torch
except ImportError:
    torch = None
from live_mode.hf_gate2 import COLORS
from live_mode.hf_gate3 import evaluate, gate_report


@unittest.skipIf(torch is None, 'PyTorch optional')
class Gate3ScoringTests(unittest.TestCase):
    def test_reports_all_arms_and_no_gate_cherry_pick(self):
        class FakeModel:
            def eval(self):
                return self

        class FakeAdapter:
            model = FakeModel()
            label_ids = {c: i + 1 for i, c in enumerate(COLORS)}
            lift = torch.nn.Linear(4, 3, bias=False)

            def _logits(self, prompt, readout):
                logits = torch.zeros(12)
                logits[0] = 1.0
                if readout is not None:
                    logits[1:5] = torch.as_tensor(readout) * 12
                elif prompt.startswith('Fact: '):
                    for color in COLORS:
                        if f'color {color}.\n' in prompt:
                            logits[self.label_ids[color]] = 12
                return logits

        report = evaluate(FakeAdapter(), slots=(8,), delays=(10,),
                          trials=2, seeds=(21,))
        self.assertEqual(len(report['cells']), 1)
        scores = report['cells'][0]['summary']
        self.assertEqual(scores['trials'], 2)
        self.assertEqual(set(scores['scores']), {
            'live', 'erased', 'stale', 'wrong_address', 'last_only',
            'oracle_code', 'dictionary_context'})
        self.assertEqual(scores['scores']['oracle_code']['strict_next_token_accuracy'], 1)
        self.assertEqual(scores['scores']['dictionary_context']['four_color_accuracy'], 1)
        self.assertEqual(gate_report(report['cells'])['slots_16_delay_100'], 'not_run')


if __name__ == '__main__':
    unittest.main()
