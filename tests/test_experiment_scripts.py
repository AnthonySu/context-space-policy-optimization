"""Tests for experiment runner scripts (--quick mode only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(script_name: str, extra_args: list[str] | None = None) -> dict:
    """Run a script with --quick and return parsed JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", script_name),
            "--quick",
            "--output-dir",
            tmpdir,
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"{script_name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Find the JSON output file
        json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert len(json_files) >= 1, f"No JSON output found in {tmpdir}"

        with open(os.path.join(tmpdir, json_files[0])) as f:
            return json.load(f)


class TestD4RLRunner:
    def test_d4rl_runner_quick(self):
        """run_d4rl_experiments.py --quick produces valid JSON output."""
        data = _run_script("run_d4rl_experiments.py")
        assert data["experiment"] == "d4rl_cspo"
        assert data["quick_mode"] is True
        assert "results" in data
        assert len(data["results"]) > 0

    def test_d4rl_runner_has_dt_and_cspo(self):
        """Output contains both DT and CSPO results per environment."""
        data = _run_script("run_d4rl_experiments.py")
        for env_name, result in data["results"].items():
            assert "dt" in result, f"Missing DT results for {env_name}"
            assert "cspo" in result, f"Missing CSPO results for {env_name}"
            assert "mean" in result["dt"]["normalized"]


class TestAblationRunner:
    def test_ablation_runner_quick(self):
        """run_ablation.py --quick produces valid JSON output."""
        data = _run_script(
            "run_ablation.py", ["--sweep", "group_size"]
        )
        assert data["experiment"] == "cspo_ablation"
        assert data["quick_mode"] is True
        assert "sweeps" in data
        assert "group_size" in data["sweeps"]

    def test_ablation_sweep_points(self):
        """Each sweep point has best_score and time_seconds."""
        data = _run_script(
            "run_ablation.py", ["--sweep", "group_size"]
        )
        for point in data["sweeps"]["group_size"]:
            assert "best_score" in point
            assert "time_seconds" in point
            assert isinstance(point["best_score"], (int, float))


class TestComputeComparison:
    def test_compute_comparison_quick(self):
        """run_compute_comparison.py --quick produces valid JSON output."""
        data = _run_script("run_compute_comparison.py")
        assert data["experiment"] == "compute_comparison"
        assert "results" in data
        assert "DT" in data["results"]
        assert "CSPO" in data["results"]
        assert "speedups" in data["results"]

    def test_compute_cspo_zero_gpu(self):
        """CSPO should report 0 GPU-hours."""
        data = _run_script("run_compute_comparison.py")
        assert data["results"]["speedups"]["cspo_gpu_hours"] == 0.0
