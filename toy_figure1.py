"""Reproduce Figure 1 of Schaipp, Gower, Taylor:
Step-Size Stability in Stochastic Optimization (arXiv:2602.09842).

One-step updates of SGD / SPS / NGN / SPP on
    f(x) = ln(1 + exp(-x)) + max{x - 2, 0}
from x_t = -3, for step sizes alpha in (0, 15].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


X_T = -3.0
C_T = 0.0  # SPS lower bound used in the paper's experiments
ALPHA_MAX = 15.0
HINGE_KINK = 2.0


def f(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    val = np.logaddexp(0.0, -x) + np.maximum(x - HINGE_KINK, 0.0)
    return float(val) if val.ndim == 0 else val


def grad(x: float) -> float:
    """Subgradient: logistic part plus hinge (0 left of 2, 1 right of 2)."""
    g_log = -1.0 / (1.0 + np.exp(x))
    if x < HINGE_KINK:
        return float(g_log)
    if x > HINGE_KINK:
        return float(g_log + 1.0)
    lo, hi = g_log, g_log + 1.0
    if lo <= 0.0 <= hi:
        return 0.0
    return float(lo if abs(lo) < abs(hi) else hi)


def sgd_step(x: float, g: float, alpha: float) -> float:
    return x - alpha * g


def sps_step(x: float, fx: float, g: float, alpha: float) -> tuple[float, float]:
    g2 = g * g
    tau = min(alpha, (fx - C_T) / g2)
    return x - tau * g, tau


def ngn_step(x: float, fx: float, g: float, alpha: float) -> tuple[float, float]:
    g2 = g * g
    gamma = alpha / (1.0 + (alpha / (2.0 * fx)) * g2)
    return x - gamma * g, gamma


def spp_step(x: float, alpha: float) -> float:
    def obj(y: float) -> float:
        return f(y) + 0.5 / alpha * (y - x) ** 2

    res = minimize_scalar(
        obj,
        bounds=(x - 80.0, x + 80.0),
        method="bounded",
        options={"xatol": 1e-14},
    )
    return float(res.x)


def delta_sgd(g: float, alpha: float) -> float:
    return 0.5 * alpha * g * g


def delta_sps(g: float, alpha: float, tau: float) -> float:
    return tau * (1.0 - tau / (2.0 * alpha)) * g * g


def delta_ngn(g: float, gamma: float) -> float:
    return 0.5 * gamma * g * g


def delta_spp(fx: float, x: float, x_next: float, alpha: float) -> float:
    return fx - f(x_next) - (0.5 / alpha) * (x_next - x) ** 2


def main() -> None:
    alphas = np.linspace(1e-3, ALPHA_MAX, 800)
    x, g = X_T, grad(X_T)
    fx = f(x)
    g2 = g * g

    f_next = {k: np.empty_like(alphas) for k in ("SGD", "SPS", "NGN", "SPP")}
    deltas = {k: np.empty_like(alphas) for k in ("SGD", "SPS", "NGN", "SPP")}

    for i, alpha in enumerate(alphas):
        x_sgd = sgd_step(x, g, alpha)
        x_sps, tau = sps_step(x, fx, g, alpha)
        x_ngn, gamma = ngn_step(x, fx, g, alpha)
        x_spp = spp_step(x, alpha)

        f_next["SGD"][i] = f(x_sgd)
        f_next["SPS"][i] = f(x_sps)
        f_next["NGN"][i] = f(x_ngn)
        f_next["SPP"][i] = f(x_spp)

        deltas["SGD"][i] = delta_sgd(g, alpha)
        deltas["SPS"][i] = delta_sps(g, alpha, tau)
        deltas["NGN"][i] = delta_ngn(g, gamma)
        deltas["SPP"][i] = delta_spp(fx, x, x_spp, alpha)

    # SGD lands at the minimizer x=2 when -3 - alpha * g = 2.
    alpha_sgd_best = (x - HINGE_KINK) / g

    colors = {
        "SGD": "#1f77b4",
        "SPS": "#ff7f0e",
        "NGN": "#2ca02c",
        "SPP": "#d62728",
    }
    order = ("SGD", "SPS", "NGN", "SPP")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")

    ax0, ax1 = axes
    for name in order:
        ax0.plot(alphas, f_next[name], color=colors[name], lw=2.0, label=name)
        ax1.plot(alphas, deltas[name], color=colors[name], lw=2.0, label=name)

    for ax in axes:
        ax.axvline(alpha_sgd_best, color="0.45", ls="--", lw=1.2, zorder=0)
        ax.set_xlim(0.0, ALPHA_MAX)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(True, alpha=0.25)

    ax0.set_ylabel(r"$f(x_{t+1})$")
    ax1.set_ylabel(r"$\delta_t$")
    ax0.legend(frameon=False, loc="upper left")
    ax1.legend(frameon=False, loc="upper left")
    ax0.set_title(r"Next-iterate loss ($x_t=-3$)")
    ax1.set_title(r"Stability index")

    fig.savefig("figure1.png", dpi=200)
    print(f"g(x_t) = {g:.6f}")
    print(f"f(x_t) = {fx:.6f}")
    print(f"SPS Polyak cap (f-C)/g^2 = {(fx - C_T) / g2:.6f}")
    print(f"SGD best alpha (lands at x=2) = {alpha_sgd_best:.6f}")
    print("saved figure1.png")


if __name__ == "__main__":
    main()
