## Berghain Challenge (Minimal Implementation)

Full post (methods, math, reasoning) lives **[here](https://elmoddy.github.io/berghain-challenge/)**.

This repository only ships the minimal code referenced in the write-up.

### Files
| File | Purpose |
|------|---------|
| `main.py` | Runs one game for a chosen scenario (1–3). Orchestrates API calls and dynamic policy recomputation. |
| `policy.py` | Contains the acceptance policy optimizer (SLSQP-based). |
| `mu_builder.py` | Offline helper to build a joint distribution consistent with provided marginals + correlations plus a prior empirical distribution. |

### Setup (uv)
```bash
# 1. Install uv (see https://docs.astral.sh/uv/)
# 2. Install dependencies
uv sync

# 3. Export your player/client id
export CLIENT_ID=your_id_here

# 4. Run a scenario (1, 2, or 3)
uv run python main.py 1
```

### Environment
`CLIENT_ID` must be set in the environment; the script reads it at runtime.

### Rebuild Distribution (optional)
```bash
uv run python mu_builder.py
```
Edit `mu_builder.py` inline to adjust prior or correlation config.

