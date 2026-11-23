"""Formule analitice pentru R, L, C (SI: metri, ohmi, henry, farazi).

- R: R = L / (σ A) pentru curent uniform.
- L parțial (Ruehli 1972): L_p = μ0 / (4π A_i A_j) ∫∫ dV dV' / |r - r'| pentru curent uniform pe secțiune.
  Pentru bare paralele integrala pe lungime e închisă (Neumann); integrala pe secțiune se face numeric,
  ca o convoluție de funcții dreptunghi, deci rezultatul e „exact” până la toleranța cuadraturii.
- Grover: aproximația închisă pentru bara dreptunghiulară (l ≫ w + t).
- C: sferă, cub (constantă de referință), plăci paralele, microstrip peste plan (Sakurai–Tamaru).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

MU0 = 4e-7 * math.pi
EPS0 = 8.854187817e-12
CUBE_CAPACITANCE_CONSTANT = 0.66067813  # C_cube = k · 4π ε a (Hwang & Mascagni 2004)


# --- R -----------------------------------------------------------------------------------------------------------
def resistance_bar(length: float, width: float, thickness: float, sigma: float) -> float:
    return length / (sigma * width * thickness)


# --- L -----------------------------------------------------------------------------------------------------------
def _H(u, d):
    return u * np.arcsinh(u / d) - np.sqrt(u * u + d * d)


def neumann_parallel_filaments(a1: float, b1: float, a2: float, b2: float, d: float) -> float:
    """Partial mutual inductance [H] of two parallel filaments [a1,b1], [a2,b2] on axes at distance d > 0."""
    integral = -(_H(b1 - b2, d) - _H(a1 - b2, d) - _H(b1 - a2, d) + _H(a1 - a2, d))
    return MU0 / (4 * math.pi) * integral


def partial_inductance_parallel_bars(length: float, w1: float, t1: float, w2: float, t2: float,
                                     dy: float = 0.0, dz: float = 0.0, epsrel: float = 1e-9) -> float:
    """Partial mutual inductance [H] of two aligned parallel rectangular bars of equal length.

    Bar 1 cross-section [0,w1]×[0,t1]; bar 2 is [dy,dy+w2]×[dz,dz+t2]. With dy = dz = 0 and equal sections this is
    the partial self-inductance.
    """

    def kernel(u, v):
        rho = math.hypot(u, v)
        if rho == 0.0:
            return 0.0  # integrable log singularity on a set of measure zero
        return 2.0 * (length * math.asinh(length / rho) - math.sqrt(length**2 + rho**2) + rho)

    # u = y - y', v = z - z' with y in bar 1, y' in bar 2 → u in [-(dy+w2), w1-dy]
    u_lo, u_hi = -(dy + w2), w1 - dy
    v_lo, v_hi = -(dz + t2), t1 - dz
    def wy(u):
        return max(0.0, min(w1, u + dy + w2) - max(0.0, u + dy))

    def wz(v):
        return max(0.0, min(t1, v + dz + t2) - max(0.0, v + dz))

    def breaks(lo, hi, *inner):
        return sorted({lo, hi, *(x for x in inner if lo < x < hi)})

    # split at the kinks of the weights and at the kernel singularity u = v = 0
    u_pts = breaks(u_lo, u_hi, 0.0, -dy, w1 - dy - w2)
    v_pts = breaks(v_lo, v_hi, 0.0, -dz, t1 - dz - t2)
    total = 0.0
    for ua, ub in zip(u_pts, u_pts[1:]):
        for va, vb in zip(v_pts, v_pts[1:]):
            val, _ = integrate.dblquad(lambda v, u: wy(u) * wz(v) * kernel(u, v), ua, ub, va, vb,
                                       epsabs=0.0, epsrel=epsrel)
            total += val
    return MU0 / (4 * math.pi) * total / (w1 * t1 * w2 * t2)


def partial_self_inductance_bar(length: float, width: float, thickness: float) -> float:
    """Exact (quadrature) partial self-inductance [H] of a straight rectangular bar with uniform current."""
    return partial_inductance_parallel_bars(length, width, thickness, width, thickness)


def grover_self_inductance_bar(length: float, width: float, thickness: float) -> float:
    """Grover's closed form for a straight rectangular bar [H]; accurate for length ≫ width + thickness."""
    s = width + thickness
    return MU0 * length / (2 * math.pi) * (math.log(2 * length / s) + 0.5 + 0.2235 * s / length)


# --- C -----------------------------------------------------------------------------------------------------------
def capacitance_sphere(radius: float, eps_r: float = 1.0) -> float:
    return 4 * math.pi * EPS0 * eps_r * radius


def capacitance_cube(side: float, eps_r: float = 1.0) -> float:
    return CUBE_CAPACITANCE_CONSTANT * 4 * math.pi * EPS0 * eps_r * side


def capacitance_parallel_plate(area: float, gap: float, eps_r: float = 1.0) -> float:
    return EPS0 * eps_r * area / gap


def capacitance_microstrip_sakurai(length: float, width: float, thickness: float, height: float,
                                   eps_r: float = 1.0) -> float:
    """Line over ground plane, plate + fringing (Sakurai & Tamaru 1983), height = gap to the plane [F]."""
    return EPS0 * eps_r * length * (1.15 * width / height + 2.80 * (thickness / height) ** 0.222)
