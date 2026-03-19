"""Tests for baseline scores and environment configs."""

from __future__ import annotations

from src.baselines.baseline_scores import (
    BASELINE_SCORES,
    ENV_CONFIGS,
    get_baseline_table,
    get_env_config,
)
from src.utils.metrics import D4RL_REFERENCE_SCORES, normalized_score

ALL_METHODS = ["BC", "CQL", "IQL", "DT", "DT+FT", "Diffuser"]


class TestBaselineScores:
    def test_baseline_scores_structure(self):
        """Every environment should have scores for all 6 methods."""
        for env_name, scores in BASELINE_SCORES.items():
            for method in ALL_METHODS:
                assert method in scores, (
                    f"{env_name} missing method {method}"
                )
                assert isinstance(scores[method], (int, float))

    def test_env_configs_match_baselines(self):
        """Every env in BASELINE_SCORES should also appear in ENV_CONFIGS."""
        for env_name in BASELINE_SCORES:
            assert env_name in ENV_CONFIGS, (
                f"{env_name} in BASELINE_SCORES but not ENV_CONFIGS"
            )

    def test_env_configs_state_action_dims(self):
        """Verify known state/action dims for each domain."""
        expected = {
            "halfcheetah": {"state_dim": 17, "act_dim": 6},
            "hopper": {"state_dim": 11, "act_dim": 3},
            "walker2d": {"state_dim": 17, "act_dim": 6},
        }
        for env_name, cfg in ENV_CONFIGS.items():
            domain = env_name.split("-")[0]
            assert cfg["state_dim"] == expected[domain]["state_dim"], env_name
            assert cfg["act_dim"] == expected[domain]["act_dim"], env_name

    def test_normalized_score_formula(self):
        """Verify the normalization formula: 100 * (raw - random) / (expert - random)."""
        random_score, expert_score = D4RL_REFERENCE_SCORES["halfcheetah"]
        # Expert-level raw score should give ~100
        ns = normalized_score("halfcheetah-medium-v2", expert_score)
        assert abs(ns - 100.0) < 1e-6

        # Random-level raw score should give ~0
        ns = normalized_score("halfcheetah-medium-v2", random_score)
        assert abs(ns - 0.0) < 1e-6

    def test_normalized_score_unknown_env(self):
        """Unknown environment should return raw score unchanged."""
        raw = 42.0
        ns = normalized_score("unknown-env-v99", raw)
        assert ns == raw

    def test_get_baseline_table_all(self):
        """get_baseline_table() with no filter returns all envs."""
        table = get_baseline_table()
        assert len(table) == len(BASELINE_SCORES)

    def test_get_baseline_table_filtered(self):
        """get_baseline_table() with filter returns only requested envs."""
        envs = ["halfcheetah-medium-v2", "hopper-medium-v2"]
        table = get_baseline_table(envs)
        assert set(table.keys()) == set(envs)

    def test_get_env_config_known(self):
        """get_env_config returns correct config for known env."""
        cfg = get_env_config("hopper-medium-v2")
        assert cfg["state_dim"] == 11
        assert cfg["act_dim"] == 3

    def test_get_env_config_unknown_fallback(self):
        """get_env_config falls back to halfcheetah defaults for unknown env."""
        cfg = get_env_config("unknown-env-v99")
        assert cfg["state_dim"] == 17
        assert cfg["act_dim"] == 6
