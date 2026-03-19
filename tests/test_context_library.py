"""Tests for context library."""

import os
import tempfile

import numpy as np
import pytest

from src.cspo.context_library import ContextLibrary


class TestContextLibrary:
    def test_add_and_retrieve(self):
        lib = ContextLibrary()
        ctx = np.random.randn(20, 17).astype(np.float32)
        lib.add("env1", ctx, score=100.0)
        assert lib.size("env1") == 1
        assert lib.size() == 1

    def test_get_best(self):
        lib = ContextLibrary()
        for i in range(5):
            ctx = np.random.randn(20, 17).astype(np.float32)
            lib.add("env1", ctx, score=float(i * 10))

        best = lib.get_best("env1", k=2)
        assert len(best) == 2
        assert best[0].score == 40.0
        assert best[1].score == 30.0

    def test_get_best_nonexistent(self):
        lib = ContextLibrary()
        assert lib.get_best("nonexistent") == []

    def test_env_ids(self):
        lib = ContextLibrary()
        lib.add("env1", np.zeros((5, 3)), 1.0)
        lib.add("env2", np.zeros((5, 3)), 2.0)
        assert set(lib.env_ids) == {"env1", "env2"}

    def test_merge(self):
        lib1 = ContextLibrary()
        lib2 = ContextLibrary()
        lib1.add("env1", np.zeros((5, 3)), 1.0)
        lib2.add("env1", np.ones((5, 3)), 2.0)
        lib2.add("env2", np.ones((5, 3)), 3.0)

        lib1.merge(lib2)
        assert lib1.size("env1") == 2
        assert lib1.size("env2") == 1
        assert lib1.size() == 3

    def test_save_and_load(self):
        lib = ContextLibrary()
        ctx1 = np.random.randn(20, 17).astype(np.float32)
        ctx2 = np.random.randn(20, 17).astype(np.float32)
        lib.add("env1", ctx1, 100.0, metadata={"epoch": 3})
        lib.add("env1", ctx2, 200.0)
        lib.add("env2", ctx1, 50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_lib")
            lib.save(path)

            loaded = ContextLibrary.load(path)
            assert loaded.size("env1") == 2
            assert loaded.size("env2") == 1

            best = loaded.get_best("env1", k=1)
            assert best[0].score == 200.0

            np.testing.assert_array_almost_equal(
                loaded.get_all("env1")[0].context, ctx1
            )

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ContextLibrary.load("/nonexistent/path/lib")

    def test_metadata_preserved(self):
        lib = ContextLibrary()
        lib.add("env1", np.zeros((5, 3)), 1.0, metadata={"key": "value"})
        entry = lib.get_best("env1")[0]
        assert entry.metadata == {"key": "value"}

    def test_get_all(self):
        lib = ContextLibrary()
        for i in range(3):
            lib.add("env1", np.zeros((5, 3)), float(i))
        assert len(lib.get_all("env1")) == 3
        assert len(lib.get_all("nonexistent")) == 0

    def test_repr(self):
        lib = ContextLibrary()
        lib.add("env1", np.zeros((5, 3)), 42.0)
        r = repr(lib)
        assert "env1" in r
        assert "42.0" in r
