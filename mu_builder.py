import json
import numpy as np
from itertools import product
from scipy.optimize import minimize

def parse_config(config_json):
    """Return (p, R) from configuration JSON.

    config_json : dict or JSON string with keys relativeFrequencies, correlations.
    p : numpy array of marginal probabilities.
    R : numpy array correlation matrix.
    """
    if isinstance(config_json, str):
        config_json = json.loads(config_json)
    relfreq = config_json["relativeFrequencies"]
    attrs = list(relfreq.keys())
    p = np.array([relfreq[a] for a in attrs], dtype=float)
    corr_json = config_json["correlations"]
    R = np.zeros((len(attrs), len(attrs)))
    for i, ai in enumerate(attrs):
        for j, aj in enumerate(attrs):
            R[i, j] = corr_json[ai][aj]
    return p, R



def build_mu(config_json, prior_nu, attr_indices, tol=1e-8, n_starts=100):
    """Fit maximum-entropy style joint distribution close to prior.

    config_json : mapping defining target marginals and correlations.
    prior_nu : dict[state_tuple] -> prior mass (not necessarily normalized).
    attr_indices : dict attribute name -> index (returned unchanged).
    tol : float threshold for pruning negligible probabilities.
    n_starts : int number of random restarts.
    Returns (mu_opt, attr_indices).
    """
    n = len(attr_indices)
    p, R = parse_config(config_json)
    states = list(product([0, 1], repeat=n))
    m = len(states)
    nu_array = np.array([prior_nu.get(s, 1e-8) for s in states])
    nu_array /= nu_array.sum()
    def objective(mu):
        return -np.sum(nu_array * np.log(np.maximum(mu, 1e-12)))
    cons = []
    for i in range(n):
        row = np.array([s[i] for s in states])
        cons.append({'type': 'eq', 'fun': lambda mu, r=row, val=p[i]: np.dot(mu, r) - val})
    for i in range(n):
        for j in range(i+1, n):
            target = R[i, j]
            cov = target * np.sqrt(p[i]*(1-p[i])*p[j]*(1-p[j]))
            val = p[i]*p[j] + cov
            row = np.array([s[i]*s[j] for s in states])
            cons.append({'type': 'eq', 'fun': lambda mu, r=row, val=val: np.dot(mu, r) - val})
    cons.append({'type': 'eq', 'fun': lambda mu: np.sum(mu) - 1.0})
    bounds = [(0, 1)] * m
    best_res = None
    best_fun = np.inf
    for _ in range(n_starts):
        x0 = np.random.dirichlet(np.ones(m))
        x0 = 0.9 * x0 + 0.1 * nu_array
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': tol, 'maxiter': 1000})
        if res.success and res.fun < best_fun:
            best_res = res
            best_fun = res.fun
    if best_res is None:
        raise ValueError("Optimization failed")
    mu_opt = {states[i]: best_res.x[i] for i in range(m) if best_res.x[i] > tol}
    return mu_opt, attr_indices


def main():
    """Manual invocation helper to print built distribution."""
    prior_nu = {(1, 1, 1, 0, 0, 0): 0.2681, (1, 0, 0, 0, 0, 1): 0.1542, (0, 1, 0, 0, 0, 1): 0.0069, (1, 1, 0, 0, 0, 0): 0.0446, (1, 0, 1, 0, 0, 1): 0.1059, (0, 1, 1, 0, 0, 0): 0.148, (0, 1, 0, 0, 0, 0): 0.0145, (0, 1, 1, 0, 0, 1): 0.0176, (0, 0, 0, 0, 0, 0): 0.0144, (0, 1, 0, 1, 0, 0): 0.0002, (0, 0, 1, 0, 0, 1): 0.0656, (1, 0, 1, 0, 1, 1): 0.0029, (0, 0, 1, 0, 0, 0): 0.0113, (0, 0, 0, 0, 0, 1): 0.024, (1, 1, 1, 1, 0, 1): 0.0034, (1, 0, 1, 0, 0, 0): 0.0132, (1, 1, 0, 0, 0, 1): 0.012, (1, 1, 0, 0, 1, 1): 0.0043, (1, 1, 0, 1, 1, 1): 0.0021, (1, 1, 1, 1, 1, 1): 0.005, (1, 0, 1, 1, 0, 0): 0.0016, (1, 0, 0, 0, 0, 0): 0.0067, (1, 1, 1, 0, 0, 1): 0.0245, (1, 0, 0, 0, 1, 1): 0.0059, (1, 0, 0, 1, 0, 1): 0.0028, (1, 0, 0, 1, 1, 0): 0.0007, (1, 0, 0, 0, 1, 0): 0.0013, (1, 0, 1, 1, 1, 1): 0.0023, (0, 0, 1, 1, 0, 1): 0.001, (0, 0, 1, 1, 0, 0): 0.0009, (1, 1, 1, 1, 1, 0): 0.0029, (0, 1, 1, 1, 0, 0): 0.0021, (1, 0, 0, 1, 1, 1): 0.0037, (1, 0, 1, 1, 0, 1): 0.0035, (0, 1, 1, 1, 1, 1): 0.0009, (1, 1, 1, 0, 1, 0): 0.0013, (1, 1, 1, 0, 1, 1): 0.0019, (1, 1, 1, 1, 0, 0): 0.0045, (1, 1, 0, 0, 1, 0): 0.0021, (1, 1, 0, 1, 0, 0): 0.0012, (0, 0, 1, 1, 1, 0): 0.0002, (0, 1, 1, 1, 1, 0): 0.0008, (1, 0, 0, 1, 0, 0): 0.0006, (0, 0, 1, 1, 1, 1): 0.0003, (1, 0, 1, 0, 1, 0): 0.0006, (0, 0, 1, 0, 1, 1): 0.0003, (0, 0, 0, 0, 1, 0): 0.0002, (1, 1, 0, 1, 1, 0): 0.0006, (0, 1, 0, 0, 1, 0): 0.0001, (1, 1, 0, 1, 0, 1): 0.0003, (0, 1, 1, 1, 0, 1): 0.0012, (0, 1, 0, 0, 1, 1): 0.0006, (1, 0, 1, 1, 1, 0): 0.0014, (0, 1, 0, 1, 1, 1): 0.0001, (0, 1, 1, 0, 1, 0): 0.0006, (0, 1, 0, 1, 1, 0): 0.0001, (0, 0, 0, 1, 0, 0): 0.0001, (0, 0, 0, 1, 1, 1): 0.0002, (0, 0, 0, 1, 0, 1): 0.0003, (0, 1, 0, 1, 0, 1): 0.0004, (0, 0, 0, 0, 1, 1): 0.0004, (0, 0, 0, 1, 1, 0): 0.0001, (0, 1, 1, 0, 1, 1): 0.0004, (0, 0, 1, 0, 1, 0): 0.0001}
    attr_indices = {'underground_veteran': 0, 'international': 1, 'fashion_forward': 2, 'queer_friendly': 3, 'vinyl_collector': 4, 'german_speaker': 5}

    config_json = {
        "relativeFrequencies": {
            "underground_veteran": 0.6794999999999999,
            "international": 0.5735,
            "fashion_forward": 0.6910000000000002,
            "queer_friendly": 0.04614,
            "vinyl_collector": 0.044539999999999996,
            "german_speaker": 0.4565000000000001
        },
        "correlations": {
            "underground_veteran": {
                "underground_veteran": 1,
                "international": -0.08110175777152992,
                "fashion_forward": -0.1696563475505309,
                "queer_friendly": 0.03719928376753885,
                "vinyl_collector": 0.07223521156389842,
                "german_speaker": 0.11188766703422799
            },
            "international": {
                "underground_veteran": -0.08110175777152992,
                "international": 1,
                "fashion_forward": 0.375711059360155,
                "queer_friendly": 0.0036693314388711686,
                "vinyl_collector": -0.03083247098181075,
                "german_speaker": -0.7172529382519395
            },
            "fashion_forward": {
                "underground_veteran": -0.1696563475505309,
                "international": 0.375711059360155,
                "fashion_forward": 1,
                "queer_friendly": -0.0034530926793377476,
                "vinyl_collector": -0.11024719606358546,
                "german_speaker": -0.3521024461597403
            },
            "queer_friendly": {
                "underground_veteran": 0.03719928376753885,
                "international": 0.0036693314388711686,
                "fashion_forward": -0.0034530926793377476,
                "queer_friendly": 1,
                "vinyl_collector": 0.47990640803167306,
                "german_speaker": 0.04797381132680503
            },
            "vinyl_collector": {
                "underground_veteran": 0.07223521156389842,
                "international": -0.03083247098181075,
                "fashion_forward": -0.11024719606358546,
                "queer_friendly": 0.47990640803167306,
                "vinyl_collector": 1,
                "german_speaker": 0.09984452286269897
            },
            "german_speaker": {
                "underground_veteran": 0.11188766703422799,
                "international": -0.7172529382519395,
                "fashion_forward": -0.3521024461597403,
                "queer_friendly": 0.04797381132680503,
                "vinyl_collector": 0.09984452286269897,
                "german_speaker": 1
            }
        }
    }
    mu, attrs = build_mu(config_json, prior_nu, attr_indices)
    print(mu)

if __name__ == "__main__":
    main()