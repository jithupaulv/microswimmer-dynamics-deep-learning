from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    xdot: np.ndarray
    summary: Dict[str, float]


def reduced_rhs(t: float, z: np.ndarray, beta: float, gamma: float, omega: float) -> np.ndarray:
    """Reduced nondimensional ODE from the paper's Eq. (4).

    z = [theta, phi]
    """
    theta, phi = float(z[0]), float(z[1])
    forcing = gamma * math.sin(theta) + phi - beta * math.cos(theta) * math.sin(t * omega)
    denom = math.cos(2.0 * phi) - 17.0

    theta_dot_num = (
        3.0 * (math.cos(phi) ** 2) * forcing
        - 3.0 * ((math.sin(phi) ** 2) - 19.0) * forcing
        + 36.0 * phi * math.cos(phi)
    )
    phi_dot_num = 6.0 * ((math.cos(phi) + 3.0) ** 2) * (gamma * math.sin(theta) + 2.0 * phi - beta * math.cos(theta) * math.sin(t * omega))

    theta_dot = theta_dot_num / denom
    phi_dot = phi_dot_num / denom
    return np.array([theta_dot, phi_dot], dtype=float)


def compact_theta_residual(theta: np.ndarray, theta_t: np.ndarray, theta_tt: np.ndarray,
                           t: np.ndarray, beta: np.ndarray, gamma: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compact second-order residual from Eq. (8). Useful for PINN regularization."""
    return (
        theta_tt
        + (15.0 / 4.0) * theta_t * (beta * np.sin(theta) * np.sin(t * omega) + gamma * np.cos(theta))
        + 9.0 * gamma * np.sin(theta)
        + 12.0 * theta_t
        - (15.0 / 4.0) * beta * omega * np.cos(theta) * np.cos(t * omega)
        - 9.0 * beta * np.cos(theta) * np.sin(t * omega)
    )


def xdot(theta: np.ndarray, phi: np.ndarray, t: np.ndarray, beta: float, gamma: float, omega: float) -> np.ndarray:
    """Instantaneous x velocity from the paper's Eq. (25)."""
    forcing = -beta * np.cos(theta) * np.sin(t * omega) + gamma * np.sin(theta) + phi

    term1_num = -3.0 * np.sin(theta) * (
        (np.cos(phi) ** 2) * forcing
        - ((np.sin(phi) ** 2) + 5.0) * forcing
        + 4.0 * phi * np.cos(phi)
    )
    term1_den = 34.0 - 2.0 * np.cos(2.0 * phi)

    term2_num = -3.0 * np.sin(phi) * np.cos(theta) * (np.cos(phi) * forcing + 3.0 * phi)
    term2_den = (np.cos(phi) ** 2) - 9.0

    return term1_num / term1_den + term2_num / term2_den


def rk4_step(fun, t: float, z: np.ndarray, dt: float, *args) -> np.ndarray:
    k1 = fun(t, z, *args)
    k2 = fun(t + 0.5 * dt, z + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, z + 0.5 * dt * k2, *args)
    k4 = fun(t + dt, z + dt * k3, *args)
    return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(beta: float, gamma: float, omega: float,
             theta0: float, phi0: float,
             t_final: float = 15.0, dt: float = 0.02,
             discard_fraction: float = 0.4) -> SimResult:
    steps = int(round(t_final / dt)) + 1
    t = np.linspace(0.0, t_final, steps)
    z = np.zeros((steps, 2), dtype=float)
    z[0] = np.array([theta0, phi0], dtype=float)

    for i in range(steps - 1):
        z[i + 1] = rk4_step(reduced_rhs, t[i], z[i], dt, beta, gamma, omega)

    theta = z[:, 0]
    phi = z[:, 1]
    vel = xdot(theta, phi, t, beta, gamma, omega)

    k0 = int(discard_fraction * steps)
    theta_ss = theta[k0:]
    phi_ss = phi[k0:]
    vel_ss = vel[k0:]

    mean_theta = float(np.mean(theta_ss))
    mean_phi = float(np.mean(phi_ss))
    mean_speed = float(np.mean(vel_ss))
    regime = int(np.cos(mean_theta) < 0.0)

    summary = {
        "mean_theta": mean_theta,
        "mean_phi": mean_phi,
        "mean_speed": mean_speed,
        "regime": regime,
        "beta": beta,
        "gamma": gamma,
        "omega": omega,
        "theta0": theta0,
        "phi0": phi0,
    }
    return SimResult(t=t, theta=theta, phi=phi, xdot=vel, summary=summary)
