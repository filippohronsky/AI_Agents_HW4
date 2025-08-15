import numpy as np
from pathlib import Path
from collections import deque
from env_mx import MerakiMXDualWAN

lat_edges = np.array([70, 120,])
loss_edges = np.array([0.02, 0.08])
voice_edges = np.array([0.3])

edges_list = [lat_edges, loss_edges, lat_edges, loss_edges, voice_edges]
bins_sizes = [len(e) + 1 for e in edges_list]

def obs_to_idx(obs: np.ndarray) -> int:
    ks = [int(np.digitize(obs[i], edges_list[i])) for i in range(len(edges_list))]
    return int(np.ravel_multi_index(ks, bins_sizes))


def train_q(episodes=2500, alpha=0.1, gamma=0.97, eps_start=1.0, eps_min=0.10, eps_decay=0.999,
            ep_len=50, seed=42):
    env = MerakiMXDualWAN(episode_len=ep_len, seed=seed)
    n_states = int(np.prod(bins_sizes))
    n_actions = 2
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rng = np.random.default_rng(seed)
    epsilon = eps_start

    avg_buf = deque(maxlen=50)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        s = obs_to_idx(obs)
        total_r = 0.0

        for _ in range(ep_len):
            if rng.random() < epsilon:
                a = rng.integers(0, n_actions)
            else:
                a = int(np.argmax(Q[s]))

            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = obs_to_idx(obs2)

            td_target = r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (td_target - Q[s, a])

            total_r += r
            s = s2
            if terminated or truncated:
                break

        avg_buf.append(total_r)
        epsilon = max(eps_min, epsilon * eps_decay)

        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes}  avg_return={np.mean(avg_buf):.2f}  epsilon={epsilon:.2f}")

    def greedy_eval(runs=20):
        vals = []
        for _ in range(runs):
            obs, _ = env.reset()
            s = obs_to_idx(obs)
            total = 0.0
            for _ in range(ep_len):
                a = int(np.argmax(Q[s]))
                obs, r, term, trunc, _ = env.step(a)
                s = obs_to_idx(obs)
                total += r
                if term or trunc:
                    break
            vals.append(total)
        return float(np.mean(vals))

    mean_eval = greedy_eval(runs=20)
    print(f"Greedy eval: mean_return={mean_eval:.2f} over 20 ep")

    out = Path(__file__).parent / "q_table.npy"
    np.save(out, Q)
    print(f"Saved Q-table to {out}")

if __name__ == "__main__":
    train_q()