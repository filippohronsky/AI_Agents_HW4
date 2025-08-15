# Nový projekt – **AI Agents Home Work 4** (jednoduchý RL agent pre výber WAN cesty)

Toto je malý, samostatný projekt, ktorý **spĺňa zadanie**:
- *Implementuj ľubovolné prostredie* → simulácia **Meraki MX Dual‑WAN** (DIA + LTE) v `env_mx.py`.
- *Natrénuj ľubovolného agenta* → **Q‑learning s Q‑tabuľkou** v `train_q.py`.
- Dôraz na jednoduchosť – bez neurónových sietí, len `numpy` + `gymnasium`.

## Štruktúra
```
AI_Agents_HW4/
├─ pyproject.toml
├─ README.md
├─ .gitignore
├─ env_mx.py          # Gymnasium Env – MX Dual‑WAN
└─ train_q.py         # Q‑learning tréning + greedy hodnotenie
```

---

- **Stav**: `[w1_latency_ms, w1_loss, w2_latency_ms, w2_loss, voice_share]`
- **Akcia**: `0 = WAN1`, `1 = WAN2`
- **Odmena**: + za doručený hlas/bulk, − penalizácia za latenciu/stratu, − malá cena za WAN2
- **Cieľ**: chrániť hlas (QoS) a zároveň držať throughput + nízke náklady
- **Na trénovanie sa nepoužil statický dataset**: Využil sa vektor 5 hodnôt [w1_lat_ms, w1_loss, w2_lat_ms, w2_loss, voice_share]

### Spustenie
```bash
cd AI_Agents_HW4
uv sync   # nainštaluje numpy, gymnasium
uv run python train_q.py
```
**Krátky popis prostredia:** Vymysleli sme jednoduché prostredie, ktoré napodobňuje **Meraki MX dual‑WAN**. Stav obsahuje päť čísel: latenciu a stratu na **WAN1**, latenciu a stratu na **WAN2** a **voice_share** (podiel hlasovej prevádzky v celkovej záťaži). Akcia je len jedna voľba: **0 = pošli všetko na WAN1**, **1 = pošli všetko na WAN2** (aby to bolo prehľadné). Odmena zvýhodňuje doručený **hlas** a **bulk throughput**, ale **penalizuje** vysokú latenciu/stratu (pre kvalitu hlasu) a malé **náklady** účtujeme za WAN2 (simulácia metered/LTE). Linky sa v čase menia (náhodná chôdza) a občas majú výpadky (nárazovo vysoká strata). Cieľom agenta je balansovať kvalitu hlasu, priepustnosť a náklady – teda „vyberať lepší uplink pre daný okamih“.

Očakávaný výstup (príklad):
```
❯ uv run python train_q.py
Episode 100/2500  avg_return=-52.08  epsilon=0.74
Episode 200/2500  avg_return=-39.09  epsilon=0.55
Episode 300/2500  avg_return=-36.78  epsilon=0.41
.. výstup je krátený ..
Episode 2400/2500  avg_return=-20.78  epsilon=0.05
Episode 2500/2500  avg_return=-19.15  epsilon=0.05
Greedy eval: mean_return=-8.91 over 20 ep
Saved Q-table to /Users/filippohronsky/PycharmProjects/AI_Agents_HW4/q_table.npy
```
**Čo sa agent naučil (kedy volí WAN1 vs. WAN2):**
- Vo väčšine času **preferuje WAN1**, lebo má vyššiu kapacitu a **neplatí sa** zaň (0 € náklad), takže ak má WAN1 **normálne** hodnoty latencie/strát, je výhodnejší.
- **Prepne na WAN2**, keď potrebuje **ochrániť hlas** a WAN2 má **výrazne nižšiu latenciu/stratu** než WAN1 (napr. pri dočasnom zhoršení WAN1), hoci je za to malá cena.
- Ak má **WAN2 vysokú stratu** (napr. pri „výpadku“), agent sa **vracia na WAN1**.
- Pri **nízkom voice_share** (málo hlasu) a **dobrej kvalite WAN2** sa môže oplatiť použiť WAN2 aj napriek nákladu, pokiaľ prinesie lepšiu okamžitú kvalitu/throughput.


### Ako rýchlo overiť, čo agent aktuálne volí (WAN1 vs. WAN2)
Nižšie je mini‑skript, ktorý načíta `q_table.npy`, použije rovnakú diskretizáciu ako tréning a na pár **scenároch** ukáže, či politika volí 0 (WAN1) alebo 1 (WAN2):
```bash
uv run python choise_test.py
```
Testovacie scenáre:
1) intf WAN1 je OK, WAN2 je horšia → očakávaj WAN1 (0) *Podiel hlasu 80 %, intf. WAN1 OK (latency 40 ms, packet loss 1 %), WAN2 ZHORŠENÁ (latency 95 ms, packet loss 5 %)*
2) WAN1 je VÝRAZNE ZLÁ, WAN2 je VÝBORNÁ a unesie hlas (voice 30 Mb/s <= WAN2 30 Mb/s) → očakávaj WAN2 (1) *Podiel hlasu 60 %, intf. WAN1 VÝRAZNE ZLÁ (latency 160 ms, packet loss 12 %), WAN2 VÝBORNÁ (latency 35 ms, packet loss 0 %)*
3) Nízky hlas, WAN2 VÝBORNÁ, WAN1 len OK → často WAN2 (1) *Podiel hlasu 10 %, intf. WAN1 OK (latency 60 ms, packet loss 2 %), WAN2 VÝBORNÁ (latency 40 ms, packet loss 0 %)*
4) WAN2 výpadok → očakávaj návrat na WAN1 (0) *Podiel hlasu 60 %, intf. WAN2 VÝPADOK (latency 40 ms, packet loss 60 %), WAN1 OK (latency 50 ms, packet loss 2 %)*
