import numpy as np
from train_q import obs_to_idx, bins_sizes
from env_mx import MerakiMXDualWAN

Q = np.load("q_table.npy")

expected_states = int(np.prod(bins_sizes))
if Q.shape[0] != expected_states:
    raise RuntimeError(
        f"Q-table shape {Q.shape[0]} does not match discretization ({expected_states}). "
        "This means your choise_test discretization must match train_q. Retrain to regenerate q_table.npy."
    )

def one_step_reward(obs, action: int) -> float:
    env = MerakiMXDualWAN()
    env.state = np.array(obs, dtype=np.float32)
    # step mutates state; we only care about immediate reward
    _, r, *_ = env.step(action)
    return float(r)

def decide(obs):
    s = obs_to_idx(np.array(obs, dtype=np.float32))
    q = Q[s]
    # tie-break or untrained state: prefer the action with better immediate reward
    if (abs(q[0] - q[1]) < 1e-6) or (abs(q[0]) < 1e-7 and abs(q[1]) < 1e-7):
        r0 = one_step_reward(obs, 0)
        r1 = one_step_reward(obs, 1)
        return 0 if r0 >= r1 else 1
    return int(np.argmax(q))

scenarios = [
    # 1) intf WAN1 je OK, WAN2 je horšia → očakávaj WAN1 (0)
    ("Podiel hlasu 80 %, intf. WAN1 OK (latency 40 ms, packet loss 1 %), WAN2 ZHORŠENÁ (latency 95 ms, packet loss 5 %)",
     [40, 0.01, 95, 0.05, 0.8]),

    # 2) WAN1 je VÝRAZNE ZLÁ, WAN2 je VÝBORNÁ a unesie hlas (voice 30 Mb/s <= WAN2 30 Mb/s) → očakávaj WAN2 (1)
    ("Podiel hlasu 60 %, intf. WAN1 VÝRAZNE ZLÁ (latency 160 ms, packet loss 12 %), WAN2 VÝBORNÁ (latency 35 ms, packet loss 0 %)",
     [160, 0.12, 35, 0.00, 0.6]),

    # 3) Nízky hlas, WAN2 VÝBORNÁ, WAN1 len OK → často WAN2 (1)
    ("Podiel hlasu 10 %, intf. WAN1 OK (latency 60 ms, packet loss 2 %), WAN2 VÝBORNÁ (latency 40 ms, packet loss 0 %)",
     [60, 0.02, 40, 0.00, 0.1]),

    # 4) WAN2 výpadok → očakávaj návrat na WAN1 (0)
    ("Podiel hlasu 60 %, intf. WAN2 VÝPADOK (latency 40 ms, packet loss 60 %), WAN1 OK (latency 50 ms, packet loss 2 %)",
     [50, 0.02, 40, 0.60, 0.6]),
]

for name, obs in scenarios:
    a = decide(obs)
    print(f"{name}: action={a} (0=WAN1, 1=WAN2), obs={obs}")