import tempfile
import unittest
from pathlib import Path
import numpy as np
from live_mode import LiveMemory, Settings, VectorCodec, LiveStream, DEFAULT_VALUES


class LiveMemoryTests(unittest.TestCase):
    def test_read_before_write_and_future_state(self):
        m = LiveMemory(Settings(key_dim=4, value_dim=3, learning_rate=1))
        key, val = np.eye(4, dtype=np.float32)[0], np.eye(3, dtype=np.float32)[1]
        first = m.step(key, val)
        np.testing.assert_array_equal(first.prediction, np.zeros(3))
        np.testing.assert_allclose(m.read(key), val, atol=1e-7)
        second = m.step(key, val)
        self.assertAlmostEqual(second.error_before_write, 0)
        self.assertFalse(second.changed)
        self.assertEqual(m.writes, 1)

    def test_read_only_is_not_write(self):
        m = LiveMemory(Settings(key_dim=3, value_dim=2))
        key = np.ones(3, dtype=np.float32)
        m.step(key, np.array([1, 0], dtype=np.float32))
        original = m.W.copy()
        for _ in range(50):
            m.step(key)
        np.testing.assert_array_equal(m.W, original)
        self.assertEqual(m.events, 51)

    def test_rewrite_changes_answer(self):
        c = VectorCodec(DEFAULT_VALUES, 64)
        s = LiveStream(c, learning_rate=1)
        s.observe('alice/color', 'blue')
        self.assertEqual(s.ask('alice/color')[0], 'blue')
        s.observe('alice/color', 'red')
        self.assertEqual(s.ask('alice/color')[0], 'red')

    def test_state_persists_without_history(self):
        m = LiveMemory(Settings(key_dim=7, value_dim=3))
        k = np.arange(1, 8)
        m.step(k, np.array([0, 1, 0]))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'memory.npz'
            m.save(file)
            copy = LiveMemory.load(file)
            np.testing.assert_array_equal(m.W, copy.W)
            self.assertEqual(copy.state_bytes, 7 * 3 * 4)
            self.assertEqual(copy.events, 1)

    def test_rejects_nonfinite_vectors(self):
        m = LiveMemory(Settings(key_dim=3, value_dim=2))
        with self.assertRaises(ValueError):
            m.step([np.nan, 1, 2])
        with self.assertRaises(ValueError):
            m.step([1, 0, 0], [np.inf, 1])


if __name__ == '__main__':
    unittest.main()
