"""Gate 3 integrity tests. Only NumPy is required; no Hugging Face download."""
import random
import unittest
import numpy as np
from live_mode.hf_gate2 import COLORS, TRAIN_NAMES, TEST_NAMES
from live_mode.hf_gate3 import (CODEC_SEED, available_names, new_memory,
                                stream_episode, parse_positive_ints, gate_report)
from live_mode.stream import VectorCodec


class Gate3Tests(unittest.TestCase):
    def setUp(self):
        self.codec = VectorCodec(COLORS, key_dim=64, seed=CODEC_SEED)

    def test_new_names_disjoint_from_training_names(self):
        for count in (8, 16, 32, 64, 128):
            names = available_names(count)
            self.assertEqual(len(names), count)
            self.assertEqual(len(names), len(set(names)))
            self.assertFalse(set(names) & set(TRAIN_NAMES))
            self.assertTrue(set(names[:min(count, 16)]).issubset(set(TEST_NAMES)))

    def test_late_question_and_no_latest_event_shortcut(self):
        for slots in (8, 16, 32, 64):
            for delay in (10, 50, 100, 500):
                ep = stream_episode(random.Random(100 + slots + delay),
                                    self.codec, slots, delay)
                self.assertEqual(ep.memory_bytes, 1024)
                self.assertEqual(len(ep.events), slots + 4 + delay)
                self.assertNotEqual(ep.selected_name, ep.latest_name)
                self.assertNotEqual(ep.previous, ep.truth)
                self.assertEqual(ep.events[ep.selected_rewrite_index],
                                 (ep.selected_name, ep.truth))
                self.assertEqual(ep.final_truth[ep.selected_name], ep.truth)
                self.assertGreaterEqual(len(ep.events) - 1 -
                                        ep.last_target_write_index, delay)
                self.assertEqual(len(ep.protected_names), 4)
                self.assertFalse(any(name in ep.protected_names
                                     for name, _ in ep.events[-delay:]))
                self.assertIn(ep.selected_name, ep.prompt)
                self.assertNotIn(ep.truth, ep.prompt.lower())

    def test_stale_omits_only_queried_rewrite(self):
        ep = stream_episode(random.Random(21), self.codec, 16, 100)
        actual = new_memory(self.codec)
        stale = new_memory(self.codec)
        for index, (name, color) in enumerate(ep.events):
            key = self.codec.key(name + '.color')
            value = self.codec.value(color)
            actual.step(key, value)
            if index != ep.selected_rewrite_index:
                stale.step(key, value)
        key = self.codec.key(ep.address)
        np.testing.assert_array_equal(actual.read(key), ep.response)
        np.testing.assert_array_equal(stale.read(key), ep.stale_response)
        self.assertGreater(float(np.linalg.norm(ep.response -
                                                 ep.stale_response)), .1)

    def test_last_only_is_not_a_key_lookup(self):
        ep = stream_episode(random.Random(115), self.codec, 16, 50)
        self.assertNotEqual(ep.selected_name, ep.latest_name)
        np.testing.assert_array_equal(
            ep.last_value_response, self.codec.value(ep.events[-1][1]))
        self.assertTrue(any(n == ep.selected_name for n, _ in ep.events[:-50]))

    def test_determinism_and_independent_seed(self):
        a = stream_episode(random.Random(100), self.codec, 16, 100)
        b = stream_episode(random.Random(100), self.codec, 16, 100)
        c = stream_episode(random.Random(101), self.codec, 16, 100)
        self.assertEqual(a.events, b.events)
        np.testing.assert_array_equal(a.response, b.response)
        self.assertNotEqual(a.events, c.events)

    def test_missing_gates_are_not_false_passes(self):
        self.assertEqual(gate_report([])['slots_16_delay_100'], 'not_run')
        self.assertEqual(parse_positive_ints('10,50,100','delays'), (10,50,100))
        for malformed in ('', '2,2', '0', '3,bad'):
            with self.assertRaises(ValueError):
                parse_positive_ints(malformed, 'delays')


if __name__ == '__main__':
    unittest.main()
