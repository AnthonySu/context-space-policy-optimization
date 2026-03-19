"""Tests for the traffic signal control environment wrapper."""

from src.envs.traffic_env import TrafficSignalEnv, create_traffic_env


class TestTrafficSignalEnv:
    """Tests for TrafficSignalEnv."""

    def test_create_factory(self):
        """Factory function returns a valid TrafficSignalEnv."""
        env = create_traffic_env(rows=4, cols=4, max_ep_len=50, seed=42)
        assert isinstance(env, TrafficSignalEnv)
        assert env.rows == 4
        assert env.cols == 4
        assert env.max_ep_len == 50

    def test_reset_returns_valid_obs(self):
        """Reset returns an observation matching the observation space."""
        env = create_traffic_env(rows=4, cols=4, max_ep_len=50, seed=42)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_returns_correct_format(self):
        """Step returns the standard gymnasium 5-tuple."""
        env = create_traffic_env(rows=4, cols=4, max_ep_len=50, seed=42)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self):
        """Episode terminates within max_ep_len steps."""
        env = create_traffic_env(rows=4, cols=4, max_ep_len=30, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within max_ep_len"
        assert steps <= 30

    def test_info_contains_traffic_metrics(self):
        """Info dict contains expected traffic-specific keys."""
        env = create_traffic_env(rows=4, cols=4, max_ep_len=50, seed=42)
        _, info = env.reset()
        expected_keys = {"ev_link_idx", "ev_arrived", "step", "ev_travel_time"}
        assert expected_keys.issubset(info.keys())
