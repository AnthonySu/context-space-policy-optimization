"""Context library for storing and managing optimized context prefixes.

A context library maps environment IDs to curated sets of trajectory
prefixes that produce high-advantage rollouts with a frozen DT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ContextEntry:
    """A single context prefix with its evaluation score."""

    context: np.ndarray  # shape: (context_len, state_dim + action_dim + 1)
    score: float
    metadata: dict = field(default_factory=dict)


class ContextLibrary:
    """Stores and manages optimized context prefixes.

    A context library maps task/environment IDs to curated sets of
    trajectory prefixes that produce high-advantage rollouts.

    Examples
    --------
    >>> lib = ContextLibrary()
    >>> lib.add("halfcheetah-medium-v2", np.zeros((20, 18)), score=5000.0)
    >>> best = lib.get_best("halfcheetah-medium-v2", k=1)
    >>> len(best)
    1
    """

    def __init__(self) -> None:
        self.entries: dict[str, list[ContextEntry]] = {}

    def add(
        self,
        env_id: str,
        context: np.ndarray,
        score: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a context prefix with its score to the library.

        Parameters
        ----------
        env_id : str
            Environment identifier (e.g. ``"halfcheetah-medium-v2"``).
        context : np.ndarray
            The trajectory prefix array.
        score : float
            Rollout return achieved with this context.
        metadata : dict, optional
            Extra information (epoch found, trajectory index, etc.).
        """
        if env_id not in self.entries:
            self.entries[env_id] = []
        entry = ContextEntry(
            context=np.array(context, copy=True),
            score=float(score),
            metadata=metadata or {},
        )
        self.entries[env_id].append(entry)

    def get_best(self, env_id: str, k: int = 1) -> list[ContextEntry]:
        """Return the top-k context entries for the given environment.

        Parameters
        ----------
        env_id : str
            Environment identifier.
        k : int
            Number of top entries to return.

        Returns
        -------
        list[ContextEntry]
            Sorted best-first by score.
        """
        if env_id not in self.entries:
            return []
        sorted_entries = sorted(
            self.entries[env_id], key=lambda e: e.score, reverse=True
        )
        return sorted_entries[:k]

    def get_all(self, env_id: str) -> list[ContextEntry]:
        """Return all entries for the given environment."""
        return list(self.entries.get(env_id, []))

    @property
    def env_ids(self) -> list[str]:
        """Return all environment IDs in the library."""
        return list(self.entries.keys())

    def size(self, env_id: Optional[str] = None) -> int:
        """Return number of entries, optionally filtered by env_id."""
        if env_id is not None:
            return len(self.entries.get(env_id, []))
        return sum(len(v) for v in self.entries.values())

    def merge(self, other: "ContextLibrary") -> None:
        """Merge another library into this one (for cross-domain transfer).

        Parameters
        ----------
        other : ContextLibrary
            Library to merge from.  Entries are copied.
        """
        for env_id, entries in other.entries.items():
            for entry in entries:
                self.add(env_id, entry.context, entry.score, entry.metadata)

    def save(self, path: str) -> None:
        """Save the library to disk as a .npz + .json pair.

        Parameters
        ----------
        path : str
            Base path (without extension).  Creates ``path.npz`` and
            ``path.json``.
        """
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, list[dict]] = {}

        for env_id, entries in self.entries.items():
            for i, entry in enumerate(entries):
                key = f"{env_id}___{i}"
                arrays[key] = entry.context
                if env_id not in meta:
                    meta[env_id] = []
                meta[env_id].append(
                    {"score": entry.score, "metadata": entry.metadata}
                )

        np.savez_compressed(f"{path}.npz", **arrays)
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContextLibrary":
        """Load a library from disk.

        Parameters
        ----------
        path : str
            Base path (without extension).  Expects ``path.npz`` and
            ``path.json``.

        Returns
        -------
        ContextLibrary
            The loaded library.
        """
        lib = cls()
        npz_path = f"{path}.npz"
        json_path = f"{path}.json"

        if not os.path.exists(npz_path) or not os.path.exists(json_path):
            raise FileNotFoundError(f"Library files not found at {path}")

        data = np.load(npz_path)
        with open(json_path) as f:
            meta = json.load(f)

        for env_id, entries_meta in meta.items():
            for i, entry_meta in enumerate(entries_meta):
                key = f"{env_id}___{i}"
                context = data[key]
                lib.add(
                    env_id,
                    context,
                    entry_meta["score"],
                    entry_meta.get("metadata", {}),
                )

        return lib

    def __repr__(self) -> str:
        parts = [f"ContextLibrary(envs={len(self.entries)}"]
        for env_id, entries in self.entries.items():
            scores = [e.score for e in entries]
            best = max(scores) if scores else float("-inf")
            parts.append(f"  {env_id}: {len(entries)} entries, best={best:.1f}")
        return "\n".join(parts) + ")"
