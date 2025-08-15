import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MerakiMXDualWAN(gym.Env):
    """
    Jednoduché MX Dual-WAN prostredie.
    Stav: [w1_latency_ms, w1_loss, w2_latency_ms, w2_loss, voice_share]
    Akcie: 0 -> WAN1, 1 -> WAN2 (všetok dopyt v kroku)
    Reward = bulk_throughput_gain + voice_delivery_gain - voice_quality_penalty - wan2_cost
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 50, seed: int | None = 42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.episode_len = episode_len

        low = np.array([10.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([200.0, 0.2, 200.0, 0.2, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.total_demand_mbps = 50.0
        self.w1_bw = 100.0  # DIA Mbps
        self.w2_bw = 30.0   # LTE Mbps
        self.w2_cost_per_mbps = 0.005

        self.t = 0
        self.state = None

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def _sample_initial_state(self):
        w1_lat = self.rng.uniform(20, 60)
        w2_lat = self.rng.uniform(40, 120)
        w1_loss = self.rng.uniform(0.0, 0.03)
        w2_loss = self.rng.uniform(0.01, 0.08)
        voice_share = self.rng.uniform(0.2, 0.6)
        return np.array([w1_lat, w1_loss, w2_lat, w2_loss, voice_share], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.state = self._sample_initial_state()
        return self.state.copy(), {}

    def _evolve_links(self, s: np.ndarray):
        w1_lat, w1_loss, w2_lat, w2_loss, voice_share = s

        def evolve_lat(lat):
            lat = 0.9 * lat + 0.1 * self.rng.uniform(20, 150) + self.rng.normal(0, 5)
            return float(np.clip(lat, 10.0, 200.0))

        def evolve_loss(loss):
            loss = 0.9 * loss + 0.1 * self.rng.uniform(0.0, 0.1) + self.rng.normal(0, 0.005)
            return float(np.clip(loss, 0.0, 0.2))

        w1_lat = evolve_lat(w1_lat)
        w2_lat = evolve_lat(w2_lat)
        w1_loss = evolve_loss(w1_loss)
        w2_loss = evolve_loss(w2_loss)

        voice_share = float(np.clip(0.95 * voice_share + 0.05 * self.rng.uniform(0.0, 1.0) + self.rng.normal(0, 0.02), 0.0, 1.0))

        if self.rng.random() < 0.03:
            w1_loss = float(np.clip(w1_loss + 0.5, 0.0, 1.0))
        if self.rng.random() < 0.04:
            w2_loss = float(np.clip(w2_loss + 0.5, 0.0, 1.0))

        return np.array([w1_lat, w1_loss, w2_lat, w2_loss, voice_share], dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action)
        s = self.state
        w1_lat, w1_loss, w2_lat, w2_loss, voice_share = s

        total = self.total_demand_mbps
        v_demand = total * voice_share
        b_demand = total - v_demand

        if action == 0:
            lat, loss, bw, cost = w1_lat, w1_loss, self.w1_bw, 0.0
        else:
            lat, loss, bw, cost = w2_lat, w2_loss, self.w2_bw, self.w2_cost_per_mbps

        v_delivered = min(v_demand, bw)
        remaining = max(0.0, bw - v_delivered)
        b_delivered = min(b_demand, remaining)
        delivered = v_delivered + b_delivered

        voice_quality_penalty = (0.02 * lat + 12.0 * loss) * (1.0 + voice_share)
        voice_delivery_gain = 0.08 * v_delivered
        bulk_throughput_gain = 0.015 * b_delivered
        wan_cost = cost * delivered

        reward = (voice_delivery_gain + bulk_throughput_gain) - voice_quality_penalty - wan_cost
        if loss > 0.4 and total > 0:
            reward -= 2.0

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_len

        self.state = self._evolve_links(s)
        info = {"v_delivered": v_delivered, "b_delivered": b_delivered, "lat_ms": lat, "loss": loss, "selected": int(action)}
        return self.state.copy(), float(reward), terminated, truncated, info