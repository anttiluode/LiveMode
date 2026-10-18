"""Deterministic Gate 4 structural controls; no downloaded model required."""
import random
import unittest
import numpy as np
from live_mode.hf_gate2 import TRAIN_NAMES, TEST_NAMES
from live_mode.hf_gate3 import available_names
from live_mode.hf_gate4 import (CODEC, generate, names_for_training,
                                 hashed_keys, make_reads, replay, direct_score,
                                 pairwise_cosine)


class Gate4Tests(unittest.TestCase):
    def test_train_test_are_disjoint_even_at_32(self):
        for n in (8, 16, 32, 64):
            self.assertFalse(set(names_for_training(n)) & set(available_names(n)))
        self.assertFalse(set(TRAIN_NAMES) & set(TEST_NAMES))

    def test_query_selected_only_after_observations(self):
        spec = generate(random.Random(32), available_names(32), 500)
        self.assertEqual(len(spec.events), 32 + 4 + 500)
        self.assertNotEqual(spec.events[-1][0], spec.selected)
        self.assertEqual(sum(n == spec.selected for n, _ in spec.events), 2)
        self.assertNotEqual(spec.previous, spec.truth)
        self.assertEqual(spec.events[spec.selected_rewrite], (spec.selected, spec.truth))
        self.assertGreaterEqual(len(spec.events) - 1 - spec.selected_rewrite, 500)
        self.assertEqual(sum(n == spec.wrong for n, _ in spec.events) >= 1, True)

    def test_stale_drops_only_target_rewrite_not_entire_suffix(self):
        spec = generate(random.Random(120), available_names(16), 100)
        keys = hashed_keys(available_names(16))
        orig = replay(spec, keys)
        counterfactual = replay(spec, keys, exclude_write=spec.selected_rewrite)
        self.assertEqual(orig.writes, len(spec.events))
        self.assertEqual(counterfactual.writes, len(spec.events) - 1)
        self.assertGreater(float(np.linalg.norm(orig.W - counterfactual.W)), .01)
        self.assertEqual(orig.state_bytes, 1024)
        self.assertEqual(counterfactual.state_bytes, 1024)

    def test_no_future_answer_in_prompt_and_seed_replay(self):
        names = available_names(16)
        a = generate(random.Random(813887), names, 500)
        b = generate(random.Random(813887), names, 500)
        self.assertEqual(a, b)
        self.assertNotIn(a.truth, a.prompt.lower())
        self.assertNotEqual(a.selected, a.wrong)

    def test_reader_codes_not_reinterpreted_by_direct_baseline(self):
        names = available_names(16)
        spec = generate(random.Random(281828), names, 100)
        reads = make_reads(spec, hashed_keys(names))
        self.assertEqual(set(reads), {'live', 'stale', 'wrong_address',
                                       'last_only', 'oracle_code', 'erased'})
        self.assertEqual(reads['live'].shape, (4,))
        self.assertIsNone(reads['erased'])
        self.assertEqual(direct_score(reads, spec.truth), int(CODEC.decode(reads['live']) == spec.truth))

    def test_cosines_are_measured_and_not_assumed(self):
        c = pairwise_cosine(hashed_keys(available_names(16)))
        self.assertGreater(c['mean_abs'], 0)
        self.assertLess(c['mean_abs'], 1)
        self.assertGreaterEqual(c['max_abs'], c['mean_abs'])

    def test_invalid_samples_fail(self):
        with self.assertRaises(ValueError):
            generate(random.Random(1), ('a', 'b', 'c', 'd'), 10)
        with self.assertRaises(ValueError):
            generate(random.Random(1), ('a', 'b', 'c', 'd', 'e'), 0)


if __name__ == '__main__':
    unittest.main()