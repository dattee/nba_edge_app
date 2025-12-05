# NBA Edge – Process & Rules

_Last updated: YYYY-MM-DD_

## 1. Daily Workflow

1. Run `update_stats_pipeline.py slate` (once per day, early afternoon).
2. Open the app (Single Game + Full Slate).
3. Use **Full Slate** to find candidate games (MEDIUM/STRONG, edges ~3–7).
4. For each candidate, use **Single Game** to:
   - Check injuries (`injury_heavy`, impact flag).
   - Check B2B / schedule.
   - Check cheatsheet alignment.
5. Decide A-tier / B-tier / watch.
6. Log all picks you care about in **Single Game**.
7. After games finish, use **Prediction Log → Refresh Final Scores**.
8. Periodically review **Performance** tab + analysis notebook.

---

## 2. Play Classification

### A-tier (primary plays)

- Edge: **3–7 pts** (hybrid vs line).
- Confidence: **MEDIUM** or **STRONG**.
- **Models aligned** = 1 (Hybrid & CTG same side).
- `injury_heavy = 0` (injury impact low/medium).
- No brutal schedule red flags (B2B, 4-in-6, etc., unless accounted for).

### B-tier (high-volatility / smaller stake)

- Edge ≥ ~7–8 pts **or**:
- `injury_heavy = 1` (injury-driven).
- Still aligned and numerically strong, but driven by fragile components
  (recent injuries, extreme rating gaps, etc.).

### Watch / log-only

- PASS edges (<2–3 pts).
- Models not aligned.
- Injury/schedule chaos you don’t trust.
- Logged for data, **no stake** (or symbolic tiny stake).

---

## 3. Injury-Heavy Rules

- Use `DaysSinceLastGame` to down-weight long-term absences:
  - 0–3 days → full weight.
  - 4–7 days → partial weight.
  - 8–14 days → small weight.
  - >14 days → effectively 0 (baked into market & ratings).
- Effective injury adjustment (`effective_player_adj`) is **capped** (e.g., ±6 pts).
- `injury_heavy = 1` when:
  - `|effective_player_adj| ≥ 4 pts` (after capping).
- **Stance:**  
  - A-tier should **prefer** `injury_heavy = 0`.  
  - `injury_heavy = 1` → B-tier or log-only until proven reliable.

---

## 4. Stake Sizing (example – adjust to your comfort)

- Define a unit: `1u = X` dollars.
- Daily max risk: **2–3% of bankroll**.
- Per-play:
  - A-tier: **1u** standard bet.
  - B-tier: **0.5u** (or 0.25u) smaller bet.
  - Watch/log-only: **0u** (logged only).

If bankroll or risk tolerance changes, update this section.

---

## 5. Model Change Log

Track major changes so performance can be tied to model versions.

- `YYYY-MM-DD` – Injury modeling v2:
  - Recency weighting + cap on `effective_player_adj`.
  - Added `injury_heavy` flag and performance slice.
- `YYYY-MM-DD` – Performance tab UI cleanup, added injury-heavy section.
- `YYYY-MM-DD` – Added delete-specific-pick UI in Prediction Log.

(Keep appending as you change weights, add features, or adjust rules.)
