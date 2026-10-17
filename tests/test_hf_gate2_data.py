import random
import unittest
import numpy as np
from live_mode.hf_gate2 import COLORS, TRAIN_NAMES, TEST_NAMES, episode
from live_mode.stream import VectorCodec


class Gate2EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.codec = VectorCodec(COLORS, key_dim=64, seed=20261017)

    def test_disjoint_names(self):
        self.assertFalse(set(TRAIN_NAMES) & set(TEST_NAMES))

    def test_no_truth_in_prompt_and_reproducible(self):
        a = episode(random.Random(1), TEST_NAMES, self.codec)
        b = episode(random.Random(1), TEST_NAMES, self.codec)
        self.assertEqual((a.prompt, a.truth, a.previous), (b.prompt, b.truth, b.previous))
        np.testing.assert_array_equal(a.response, b.response)
        self.assertNotIn(a.truth, a.prompt.lower())
        self.assertNotEqual(a.truth, a.previous)

    def test_rewrite_changes_operator_and_wrong_key(self):
        for seed in range(40):
            e = episode(random.Random(seed), TEST_NAMES, self.codec, rewrite=True)
            self.assertGreater(float(np.linalg.norm(e.response - e.stale_response)), .2)
            self.assertGreater(float(np.linalg.norm(e.response - e.distractor_response)), .2)
            self.assertEqual(e.response.shape, (4,))

    def test_no_rewrite_no_operator_change(self):
        e = episode(random.Random(17), TEST_NAMES, self.codec, rewrite=False)
        np.testing.assert_array_equal(e.response, e.stale_response)
        self.assertEqual(e.truth, e.previous)
