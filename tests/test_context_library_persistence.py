"""Tests for context library persistence (save/load, merge, edge cases)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from src.cspo.context_library import ContextLibrary


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self):
        """Save and load a library; verify contents match exactly."""
        lib = ContextLibrary()
        ctx1 = np.random.randn(20, 17).astype(np.float32)
        ctx2 = np.random.randn(15, 17).astype(np.float32)
        lib.add("env-a", ctx1, score=100.0, metadata={"epoch": 1})
        lib.add("env-a", ctx2, score=200.0, metadata={"epoch": 2})
        lib.add("env-b", ctx1, score=50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lib")
            lib.save(path)
            loaded = ContextLibrary.load(path)

            assert loaded.size() == 3
            assert loaded.size("env-a") == 2
            assert loaded.size("env-b") == 1
            assert set(loaded.env_ids) == {"env-a", "env-b"}

            # Verify data integrity
            best_a = loaded.get_best("env-a", k=1)
            assert best_a[0].score == 200.0
            assert best_a[0].metadata == {"epoch": 2}
            np.testing.assert_array_almost_equal(
                best_a[0].context, ctx2
            )

    def test_save_load_empty_metadata(self):
        """Entries without metadata survive round-trip."""
        lib = ContextLibrary()
        lib.add("env1", np.zeros((5, 3)), 42.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lib")
            lib.save(path)
            loaded = ContextLibrary.load(path)
            entry = loaded.get_best("env1", k=1)[0]
            assert entry.score == 42.0


class TestMergeLibraries:
    def test_merge_libraries(self):
        """Merge two libraries and verify combined entries."""
        lib1 = ContextLibrary()
        lib2 = ContextLibrary()
        lib1.add("env1", np.zeros((5, 3)), 10.0)
        lib1.add("env1", np.ones((5, 3)), 20.0)
        lib2.add("env1", np.full((5, 3), 2.0), 30.0)
        lib2.add("env2", np.full((5, 3), 3.0), 40.0)

        lib1.merge(lib2)
        assert lib1.size("env1") == 3
        assert lib1.size("env2") == 1
        assert lib1.size() == 4

        # Best in env1 should be from lib2
        best = lib1.get_best("env1", k=1)
        assert best[0].score == 30.0

    def test_merge_disjoint(self):
        """Merge libraries with no overlapping envs."""
        lib1 = ContextLibrary()
        lib2 = ContextLibrary()
        lib1.add("env-a", np.zeros((3, 2)), 1.0)
        lib2.add("env-b", np.ones((3, 2)), 2.0)

        lib1.merge(lib2)
        assert set(lib1.env_ids) == {"env-a", "env-b"}
        assert lib1.size() == 2


class TestEmptyLibrary:
    def test_empty_library(self):
        """An empty library has size 0 and no env ids."""
        lib = ContextLibrary()
        assert lib.size() == 0
        assert lib.env_ids == []
        assert lib.get_best("any-env") == []
        assert lib.get_all("any-env") == []

    def test_empty_library_save_load(self):
        """Empty library survives save/load."""
        lib = ContextLibrary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lib")
            lib.save(path)
            loaded = ContextLibrary.load(path)
            assert loaded.size() == 0


class TestLargeLibrary:
    def test_large_library(self):
        """Library with many entries maintains correct ordering."""
        lib = ContextLibrary()
        n = 100
        for i in range(n):
            ctx = np.random.randn(5, 3).astype(np.float32)
            lib.add("env1", ctx, score=float(i))

        assert lib.size("env1") == n

        best_10 = lib.get_best("env1", k=10)
        assert len(best_10) == 10
        # Scores should be descending
        scores = [e.score for e in best_10]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 99.0

    def test_large_library_multiple_envs(self):
        """Library with many envs."""
        lib = ContextLibrary()
        for env_idx in range(20):
            env_id = f"env-{env_idx}"
            for i in range(5):
                lib.add(env_id, np.zeros((3, 2)), float(i))
        assert lib.size() == 100
        assert len(lib.env_ids) == 20


class TestGetBestK:
    def test_get_best_k(self):
        """get_best returns top-K entries sorted by score descending."""
        lib = ContextLibrary()
        scores = [10.0, 50.0, 30.0, 20.0, 40.0]
        for s in scores:
            lib.add("env1", np.zeros((3, 2)), s)

        best_3 = lib.get_best("env1", k=3)
        assert len(best_3) == 3
        assert [e.score for e in best_3] == [50.0, 40.0, 30.0]

    def test_get_best_k_exceeds_size(self):
        """Requesting more than available returns all entries."""
        lib = ContextLibrary()
        lib.add("env1", np.zeros((3, 2)), 1.0)
        lib.add("env1", np.zeros((3, 2)), 2.0)

        best = lib.get_best("env1", k=10)
        assert len(best) == 2

    def test_get_best_default_k(self):
        """Default k=1 returns single best entry."""
        lib = ContextLibrary()
        for i in range(5):
            lib.add("env1", np.zeros((3, 2)), float(i))

        best = lib.get_best("env1")
        assert len(best) == 1
        assert best[0].score == 4.0
