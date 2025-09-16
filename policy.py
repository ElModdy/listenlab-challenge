import numpy as np
from scipy.optimize import minimize

def optimize_acceptance_policy(remaining_acceptances, required_counts, state_distribution, quiet=True, n_init=3, ftol=1e-6):
    """Compute stochastic acceptance policy via SLSQP.

    Parameters
    ----------
    remaining_acceptances : int
        Number of people still allowed to be accepted (N in objective).
    required_counts : array-like[int]
        Remaining minimum counts required for each attribute (may be <=0 meaning already satisfied).
    state_distribution : dict[tuple[int,...], float]
        Probability mass function over attribute state tuples (each tuple length = number of attributes).
    quiet : bool
        Placeholder flag for future logging (unused currently).
    n_init : int
        Number of initialization attempts (first is all ones, rest random in [0,1]).
    ftol : float
        Convergence tolerance for SLSQP.

    Returns
    -------
    dict
        Keys: policy (mapping state->prob), Z (normalization value), success (bool).
    """
    mu = state_distribution
    constraints = required_counts
    N = remaining_acceptances
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
        x0 = np.full(n_opt, 1) if i == 0 else np.random.rand(n_opt)
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

