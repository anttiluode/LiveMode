"""An inspectable continuous stream over late, unpredictable questions.

Only shared address/value codebooks are predeclared. No event history is kept;
we explicitly compare against a dictionary to avoid claiming compression.
"""
from __future__ import annotations

import hashlib
import numpy as np
from .memory import LiveMemory, Settings


class VectorCodec:
    """Fixed pseudo-random semantic address and value codebooks.

    This is *not* a language encoder. Equal text gives equal vectors. Values
    must be in the preregistered vocabulary and are never added at run time.
    """
    def __init__(self, values: tuple[str, ...], key_dim: int = 64, seed: int = 0):
        if len(values) > 32 or len(values) < 2 or len(set(values)) != len(values):
            raise ValueError('expect 2..32 distinct predefined values')
        self.values = values
        self.key_dim = key_dim
        self.seed = seed
        # Orthogonal values eliminate output-label classification as a confound.
        self.value_vectors = np.eye(len(values), dtype=np.float32)

    @property
    def value_dim(self) -> int:
        return len(self.values)

    def key(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(f'{self.seed}:{text.strip().lower()}'.encode('utf-8')).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        x = generator.standard_normal(self.key_dim).astype(np.float32)
        return x / np.linalg.norm(x)

    def value(self, label: str) -> np.ndarray:
        return self.value_vectors[self.values.index(label)].copy()

    def decode(self, vector: np.ndarray) -> str:
        if np.max(np.abs(vector)) < 1e-7 and 'unknown' in self.values:
            return 'unknown'
        return self.values[int(np.argmax(vector))]


class LiveStream:
    def __init__(self, codec: VectorCodec, learning_rate: float = 0.85, retention: float = 1.0):
        self.codec = codec
        self.memory = LiveMemory(Settings(key_dim=codec.key_dim, value_dim=codec.value_dim,
                                         learning_rate=learning_rate, retention=retention))

    def observe(self, address: str, value: str) -> tuple[str, float]:
        result = self.memory.step(self.codec.key(address), self.codec.value(value))
        return self.codec.decode(result.prediction), float(result.error_before_write)

    def ask(self, address: str) -> tuple[str, float]:
        result = self.memory.step(self.codec.key(address))
        idx = int(np.argmax(result.prediction))
        return self.codec.decode(result.prediction), float(result.prediction[idx])


DEFAULT_VALUES = ('red', 'blue', 'green', 'yellow', 'black', 'white',
                  'circle', 'square', 'triangle', 'north', 'south', 'east', 'west',
                  'yes', 'no', 'unknown')
