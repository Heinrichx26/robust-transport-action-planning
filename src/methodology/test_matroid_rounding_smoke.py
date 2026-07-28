"""Small-instance check for the grouped robust-rounding theorem."""

from itertools import product

import numpy as np
from scipy.optimize import linprog


def feasible_plans(groups, group_caps, total_cap):
    plans = []
    for bits in product((0.0, 1.0), repeat=len(groups)):
        x = np.asarray(bits)
        if x.sum() <= total_cap and all(
            x[groups == g].sum() <= group_caps[g] for g in range(len(group_caps))
        ):
            plans.append(x)
    return np.asarray(plans)


def main():
    groups = np.array([0, 0, 1, 1, 2, 2])
    group_caps = np.array([1, 1, 1])
    total_cap = 3
    response = np.array(
        [[8.0, 2.0], [3.0, 7.0], [7.0, 3.0], [2.0, 8.0], [6.0, 4.0], [4.0, 6.0]]
    )
    n_actions, n_settings = response.shape
    c = np.r_[np.zeros(n_actions), -1.0]
    a_ub, b_ub = [], []
    for g, cap in enumerate(group_caps):
        row = np.zeros(n_actions + 1)
        row[np.where(groups == g)[0]] = 1.0
        a_ub.append(row)
        b_ub.append(cap)
    a_ub.append(np.r_[np.ones(n_actions), 0.0])
    b_ub.append(total_cap)
    for theta in range(n_settings):
        a_ub.append(np.r_[-response[:, theta], 1.0])
        b_ub.append(0.0)
    lp = linprog(
        c, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
        bounds=[(0, 1)] * n_actions + [(0, None)], method="highs"
    )
    assert lp.success
    y, z_lp = lp.x[:n_actions], lp.x[-1]
    plans = feasible_plans(groups, group_caps, total_cap)
    decomposition = linprog(
        np.zeros(len(plans)),
        A_eq=np.vstack([plans.T, np.ones(len(plans))]),
        b_eq=np.r_[y, 1.0], bounds=(0, None), method="highs"
    )
    assert decomposition.success
    probabilities = np.maximum(decomposition.x, 0.0)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(20260728)
    draws = plans[rng.choice(len(plans), size=50_000, p=probabilities)]
    assert np.all(draws.sum(axis=1) <= total_cap)
    for g, cap in enumerate(group_caps):
        assert np.all(draws[:, groups == g].sum(axis=1) <= cap)
    max_marginal_error = float(np.max(np.abs(draws.mean(axis=0) - y)))
    assert max_marginal_error < 0.015
    values = (draws @ response).min(axis=1)
    optimum_integral = float(np.max((plans @ response).min(axis=1)))
    assert z_lp + 1e-9 >= optimum_integral
    assert float(values.max()) >= optimum_integral - 1e-9
    print({
        "fractional_value": round(float(z_lp), 6),
        "integral_optimum": round(optimum_integral, 6),
        "best_sampled_value": round(float(values.max()), 6),
        "mean_sampled_value": round(float(values.mean()), 6),
        "max_marginal_error": round(max_marginal_error, 6),
        "feasibility_violations": 0,
    })


if __name__ == "__main__":
    main()
