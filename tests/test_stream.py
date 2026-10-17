import unittest
from benchmarks.run import trial
from live_mode import VectorCodec, LiveStream, DEFAULT_VALUES


class StreamTests(unittest.TestCase):
    def test_deterministic_key_and_independent_addresses(self):
        code = VectorCodec(DEFAULT_VALUES, key_dim=64)
        self.assertTrue((code.key('alice.color') == code.key('alice.color')).all())
        self.assertGreater(float(((code.key('alice.color') - code.key('bob.color'))**2).sum()), .1)

    def test_live_ask_has_no_write(self):
        stream = LiveStream(VectorCodec(DEFAULT_VALUES))
        stream.observe('alice.color', 'blue')
        n = stream.memory.writes
        stream.ask('alice.color')
        self.assertEqual(n, stream.memory.writes)

    def test_late_queries_not_freeze(self):
        result = trial(41)
        self.assertEqual(result['dictionary'], 1)
        self.assertGreater(result['delta'], result['frozen'])


if __name__ == '__main__':
    unittest.main()
