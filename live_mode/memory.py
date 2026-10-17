"""Continuous, fixed-budget associative memory: every observed event reads then writes.

There is no historical event store here. An event with a target updates one
matrix using a local normalized delta rule. A read-only probe never mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Settings:
    key_dim: int = 64
    value_dim: int = 16
    learning_rate: float = 0.8
    retention: float = 1.0
    error_gate: float = 0.0
    max_update_norm: float = 2.0

    def __post_init__(self) -> None:
        if self.key_dim < 1 or self.value_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0 < self.learning_rate <= 1 or not 0 < self.retention <= 1:
            raise ValueError("learning_rate and retention must lie in (0, 1]")
        if self.error_gate < 0 or self.max_update_norm <= 0:
            raise ValueError("gate must be nonnegative and max_update_norm positive")


@dataclass
class EventResult:
    prediction: np.ndarray
    error_before_write: float | None
    write_norm: float
    changed: bool


class LiveMemory:
    """A recurrent operator with explicitly synchronized read-then-write semantics.

    W has shape (value_dim, key_dim). Call step(key, target) on observations;
    call step(key) for probes. Writes use only the current target, current
    query and the operator's own prediction. They do not see future queries.
    """

    def __init__(self, settings: Settings = Settings()):
        self.settings = settings
        self.W = np.zeros((settings.value_dim, settings.key_dim), dtype=np.float32)
        self.events = 0
        self.writes = 0

    def _key(self, key: np.ndarray) -> np.ndarray:
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.shape != (self.settings.key_dim,) or not np.all(np.isfinite(k)):
            raise ValueError("invalid key")
        norm = float(np.linalg.norm(k))
        if norm <= 1e-8:
            raise ValueError("zero key")
        return k / norm

    def read(self, key: np.ndarray) -> np.ndarray:
        return (self.W @ self._key(key)).copy()

    def step(self, key: np.ndarray, target: np.ndarray | None = None) -> EventResult:
        k = self._key(key)
        prediction = (self.W @ k).copy()
        self.events += 1
        if target is None:
            return EventResult(prediction, None, 0.0, False)
        v = np.asarray(target, dtype=np.float32).reshape(-1)
        if v.shape != (self.settings.value_dim,) or not np.all(np.isfinite(v)):
            raise ValueError("invalid target")
        error = v - prediction
        size = float(np.linalg.norm(error))
        self.W *= self.settings.retention
        if size <= self.settings.error_gate:
            return EventResult(prediction, size, 0.0, False)
        # Normalize a unit key, clip only the residual (not the whole matrix).
        scaled_error = error * min(1.0, self.settings.max_update_norm / max(size, 1e-12))
        update = self.settings.learning_rate * np.outer(scaled_error, k)
        self.W += update
        self.writes += 1
        return EventResult(prediction, size, float(np.linalg.norm(update)), True)

    @property
    def state_bytes(self) -> int:
        return int(self.W.nbytes)

    def save(self, path: str | Path) -> None:
        # State only: no raw past examples or Python pickles.
        path = Path(path)
        np.savez_compressed(path, W=self.W, settings=json.dumps(self.settings.__dict__),
                            events=self.events, writes=self.writes)

    @classmethod
    def load(cls, path: str | Path) -> "LiveMemory":
        with np.load(path, allow_pickle=False) as data:
            config = Settings(**json.loads(str(data['settings'].item())))
            obj = cls(config)
            matrix = data['W']
            if matrix.shape != obj.W.shape or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                raise ValueError("invalid saved matrix")
            obj.W[:] = matrix
            obj.events = int(data['events'])
            obj.writes = int(data['writes'])
            return obj
