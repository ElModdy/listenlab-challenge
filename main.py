import json
import httpx
import numpy as np
import time
import sys
import random
import backoff
from scipy.optimize import minimize
import os

CLIENT_ID = os.getenv("CLIENT_ID")       # GitHub secret for client ID

# ----------------- Global State -----------------
BEST_REJECTED = [790, 3314, 4317]

# Scenario data
mu_array = [
    {(0, 0): np.float64(0.499), (0, 1): np.float64(0.17850000000000002), (1, 0): np.float64(0.1785), (1, 1): np.float64(0.14400000000000002)},
    {(0, 0, 0, 0): np.float64(0.03425090875645645), (0, 0, 0, 1): np.float64(0.04712353132226759), (0, 0, 1, 0): np.float64(0.0014340346430559681), (0, 0, 1, 1): np.float64(0.0017665252782197165), (0, 1, 0, 0): np.float64(0.030651434603756727), (0, 1, 0, 1): np.float64(0.24927812531751903), (0, 1, 1, 0): np.float64(0.003433621996730672), (0, 1, 1, 1): np.float64(0.005561818081993641), (1, 0, 0, 0): np.float64(0.4193666991917123), (1, 0, 0, 1): np.float64(0.013378360729563535), (1, 0, 1, 0): np.float64(0.003848357408775115), (1, 0, 1, 1): np.float64(0.008831582669949203), (1, 1, 0, 0): np.float64(0.09733195744807445), (1, 1, 0, 1): np.float64(0.04634898263064985), (1, 1, 1, 0): np.float64(0.011682985951438242), (1, 1, 1, 1): np.float64(0.02571107396983744)},
    {(1, 1, 1, 0, 0, 0): 0.2681, (1, 0, 0, 0, 0, 1): 0.1542, (0, 1, 0, 0, 0, 1): 0.0069, (1, 1, 0, 0, 0, 0): 0.0446, (1, 0, 1, 0, 0, 1): 0.1059, (0, 1, 1, 0, 0, 0): 0.148, (0, 1, 0, 0, 0, 0): 0.0145, (0, 1, 1, 0, 0, 1): 0.0176, (0, 0, 0, 0, 0, 0): 0.0144, (0, 1, 0, 1, 0, 0): 0.0002, (0, 0, 1, 0, 0, 1): 0.0656, (1, 0, 1, 0, 1, 1): 0.0029, (0, 0, 1, 0, 0, 0): 0.0113, (0, 0, 0, 0, 0, 1): 0.024, (1, 1, 1, 1, 0, 1): 0.0034, (1, 0, 1, 0, 0, 0): 0.0132, (1, 1, 0, 0, 0, 1): 0.012, (1, 1, 0, 0, 1, 1): 0.0043, (1, 1, 0, 1, 1, 1): 0.0021, (1, 1, 1, 1, 1, 1): 0.005, (1, 0, 1, 1, 0, 0): 0.0016, (1, 0, 0, 0, 0, 0): 0.0067, (1, 1, 1, 0, 0, 1): 0.0245, (1, 0, 0, 0, 1, 1): 0.0059, (1, 0, 0, 1, 0, 1): 0.0028, (1, 0, 0, 1, 1, 0): 0.0007, (1, 0, 0, 0, 1, 0): 0.0013, (1, 0, 1, 1, 1, 1): 0.0023, (0, 0, 1, 1, 0, 1): 0.001, (0, 0, 1, 1, 0, 0): 0.0009, (1, 1, 1, 1, 1, 0): 0.0029, (0, 1, 1, 1, 0, 0): 0.0021, (1, 0, 0, 1, 1, 1): 0.0037, (1, 0, 1, 1, 0, 1): 0.0035, (0, 1, 1, 1, 1, 1): 0.0009, (1, 1, 1, 0, 1, 0): 0.0013, (1, 1, 1, 0, 1, 1): 0.0019, (1, 1, 1, 1, 0, 0): 0.0045, (1, 1, 0, 0, 1, 0): 0.0021, (1, 1, 0, 1, 0, 0): 0.0012, (0, 0, 1, 1, 1, 0): 0.0002, (0, 1, 1, 1, 1, 0): 0.0008, (1, 0, 0, 1, 0, 0): 0.0006, (0, 0, 1, 1, 1, 1): 0.0003, (1, 0, 1, 0, 1, 0): 0.0006, (0, 0, 1, 0, 1, 1): 0.0003, (0, 0, 0, 0, 1, 0): 0.0002, (1, 1, 0, 1, 1, 0): 0.0006, (0, 1, 0, 0, 1, 0): 0.0001, (1, 1, 0, 1, 0, 1): 0.0003, (0, 1, 1, 1, 0, 1): 0.0012, (0, 1, 0, 0, 1, 1): 0.0006, (1, 0, 1, 1, 1, 0): 0.0014, (0, 1, 0, 1, 1, 1): 0.0001, (0, 1, 1, 0, 1, 0): 0.0006, (0, 1, 0, 1, 1, 0): 0.0001, (0, 0, 0, 1, 0, 0): 0.0001, (0, 0, 0, 1, 1, 1): 0.0002, (0, 0, 0, 1, 0, 1): 0.0003, (0, 1, 0, 1, 0, 1): 0.0004, (0, 0, 0, 0, 1, 1): 0.0004, (0, 0, 0, 1, 1, 0): 0.0001, (0, 1, 1, 0, 1, 1): 0.0004, (0, 0, 1, 0, 1, 0): 0.0001}
]

attr_to_index_array = [
    {'well_dressed': 0, 'young': 1},
    {'techno_lover': 0, 'well_connected': 1, 'creative': 2, 'berlin_local': 3},
    {'underground_veteran': 0, 'international': 1, 'fashion_forward': 2, 'queer_friendly': 3, 'vinyl_collector': 4, 'german_speaker': 5}
]

# ----------------- Optimizer -----------------
def optimize_policy_slsqp_grid(N, constraints, mu, quiet=True, n_init=1, ftol=1e-6):
    states = list(mu.keys())
    n_states = len(states)
    n_attrs = len(states[0])
    mu_arr = np.array([mu[s] for s in states], dtype=np.float64)
    all_ones = tuple([1] * n_attrs)
    if all_ones not in mu:
        raise ValueError("mu does not contain the all-ones state")
    opt_states = [s for s in states if s != all_ones]
    n_opt = len(opt_states)
    opt_indices = [states.index(s) for s in opt_states]
    all_ones_idx = states.index(all_ones)
    mask = np.array([[s[i] for i in range(n_attrs)] for s in states], dtype=np.float64)

    def objective(x):
        policy_vec = np.zeros(n_states, dtype=np.float64)
        policy_vec[opt_indices] = x
        policy_vec[all_ones_idx] = 1.0
        Z = np.sum(mu_arr * policy_vec)
        return -Z

    cons_list = []
    for i, c in enumerate(constraints):
        if c <= 0:
            continue
        def make_cfun(i=i, c=c):
            def cfun(x):
                policy_vec = np.zeros(n_states, dtype=np.float64)
                policy_vec[opt_indices] = x
                policy_vec[all_ones_idx] = 1.0
                Z = np.sum(mu_arr * policy_vec)
                if Z < 1e-10:
                    return -c
                expected = np.sum(mask[:, i] * mu_arr * policy_vec / Z * N)
                return expected - c
            return cfun
        cons_list.append({"type": "ineq", "fun": make_cfun()})

    bounds = [(0.0, 1.0)] * n_opt
    best = None
    best_val = float("inf")
    for i in range(n_init):
        x0 = np.full(n_opt, 0.5) if i == 0 else np.random.rand(n_opt)
        res = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=cons_list,
            options={"ftol": ftol, "disp": False, "maxiter": 500}
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x
    if best is None:
        return {"policy": {s: 0.0 for s in mu}, "Z": 0.0, "success": False}
    policy = {s: best[i] for i, s in enumerate(opt_states)}
    policy[all_ones] = 1.0
    Z = float(np.sum(mu_arr * np.array([policy[s] for s in states], dtype=np.float64)))
    return {"policy": policy, "Z": Z, "success": True}

# ----------------- Helpers -----------------
def parse_constraints(constraints, attr_to_index):
    result = np.zeros(len(attr_to_index), dtype=int)
    for c in constraints:
        attr = c["attribute"]
        if attr in attr_to_index:
            result[attr_to_index[attr]] = c["minCount"]
    return result

def parse_next_person(person, attr_to_index):
    idx = person["personIndex"]
    attrs = person["attributes"]
    n = len(attr_to_index)
    key = [0]*n
    for attr,i in attr_to_index.items():
        key[i] = int(attrs.get(attr,0))
    return idx, tuple(key), attrs

# ----------------- Client -----------------
class GameClient:
    def __init__(self, api_base, player_id, scenario):
        self.api_base = api_base
        self.player_id = player_id
        self.scenario = scenario
        self.client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            headers={
                "User-Agent": "Mozilla/5.0 ...",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            },
            http2=False
        )

    def close(self):
        self.client.close()

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.TimeoutException),
        max_tries=5,
        jitter=None
    )
    @backoff.on_predicate(
        backoff.expo,
        max_tries=5,
        predicate=lambda r: r is not None and r.status_code >= 500
    )
    def _get(self, url):
        resp = self.client.get(url)
        return resp

    def new_game(self):
        url = f"{self.api_base}/new-game?scenario={self.scenario}&playerId={self.player_id}"
        resp = self._get(url)
        resp.raise_for_status()
        return resp.json()

    def decide(self, game_id, person_index, accept=None):
        url = f"{self.api_base}/decide-and-next?gameId={game_id}&personIndex={person_index}"
        if accept is not None:
            url += f"&accept={'true' if accept else 'false'}"
        resp = self._get(url)
        resp.raise_for_status()
        return resp.json()

    def play(self):
        mu = mu_array[self.scenario-1]
        attr_to_index = attr_to_index_array[self.scenario-1]

        game = self.new_game()
        game_id = game["gameId"]
        constraints = parse_constraints(game["constraints"], attr_to_index)
        N_remain = 1000
        rejected = 0
        state = self.decide(game_id, 0)

        policy = None
        accepted_since_recompute = False
        aborted = False

        while state["status"] == "running" and N_remain > 0:
            idx, key, attrs = parse_next_person(state["nextPerson"], attr_to_index)
            if policy is None or accepted_since_recompute:
                result = optimize_policy_slsqp_grid(N_remain, constraints, mu, quiet=True)
                policy = result['policy']
                accepted_since_recompute = False
            accept = np.random.rand() < policy.get(key,0)
            if accept:
                accepted_since_recompute = True
                N_remain -= 1
                for i, val in enumerate(key):
                    if val==1:
                        constraints[i]-=1
            else:
                rejected += 1
                if rejected > BEST_REJECTED[self.scenario-1]:
                    aborted = True
                    break
            state = self.decide(game_id, idx, accept)

        if not aborted and rejected < BEST_REJECTED[self.scenario-1]:
            BEST_REJECTED[self.scenario-1] = rejected

        return {"success": state.get("status")=="won" if not aborted else False, "rejected": rejected, "aborted": aborted}

# ----------------- Runner -----------------
def main():
    scenario = int(sys.argv[1]) if len(sys.argv)>1 else 1
    time.sleep(random.random()*1.0)
    client = GameClient(
        "https://berghain.challenges.listenlabs.ai",
        CLIENT_ID,
        scenario
    )

    try:
        result = client.play()
        print(f"🎮 Game finished: {result}")
    finally:
        client.close()

if __name__=="__main__":
    main()

