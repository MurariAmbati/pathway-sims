from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


STATE_NAMES: Tuple[str, ...] = (
    "i",   # active integrin fraction
    "t",   # talin/kindlin engagement (inside-out / clustering proxy)
    "f",   # active fak fraction
    "s",   # active src fraction
    "p",   # phosphorylated paxillin (adhesion signaling proxy)
    "e",   # active erk fraction (downstream motility/proliferation proxy)
    "r",   # active rhoa fraction (contractility / tension proxy)
)


@dataclass(frozen=True)
class Params:
    # input / context
    ecm: float = 1.0        # ecm ligand density / availability
    force: float = 0.5      # tension / mechanical input (0..1-ish)

    # integrin activation / inactivation
    k_on_i: float = 1.5
    k_off_i: float = 0.6
    k_force_i: float = 1.0  # force boosts integrin activation

    # talin/kindlin engagement (adaptor / clustering)
    k_on_t: float = 1.4
    k_off_t: float = 0.9
    k_force_t: float = 0.6

    # feedback: engaged talin promotes integrin activation
    k_talin_i: float = 1.2

    # fak activation depends on active integrin
    k_on_f: float = 2.0
    k_off_f: float = 0.9

    # additional fak activation contributions
    k_talin_f: float = 0.9
    k_src_f: float = 0.7

    # src activation depends on fak
    k_on_s: float = 1.8
    k_off_s: float = 1.2

    # paxillin phosphorylation depends on active fak
    k_on_p: float = 2.0
    k_off_p: float = 0.8

    # src can also drive paxillin phosphorylation
    k_src_p: float = 0.9

    # erk downstream of src
    k_on_e: float = 1.4
    k_off_e: float = 1.1

    # rhoa activation depends on paxillin signaling + force
    k_on_r: float = 1.5
    k_off_r: float = 1.0
    k_force_r: float = 0.8


def rhs(t: float, y: np.ndarray, p: Params) -> np.ndarray:
    """ode rhs for an integrin adhesome mini-network.

    all states are fractions in [0,1] (the solver may slightly overshoot).
    """
    i, tal, f, s, pax, erk, r = y

    # saturating (bounded) activation terms
    i_on_ecm = (p.k_on_i * p.ecm) * (1.0 - i)
    i_on_force = (p.k_force_i * p.force) * (1.0 - i)
    i_on_talin = (p.k_talin_i * tal) * (1.0 - i)
    act_i = i_on_ecm + i_on_force + i_on_talin
    deact_i = p.k_off_i * i

    # talin engagement rises with integrin and force
    act_t = (p.k_on_t * i + p.k_force_t * p.force) * (1.0 - tal)
    deact_t = p.k_off_t * tal

    f_on_i = (p.k_on_f * i) * (1.0 - f)
    f_on_talin = (p.k_talin_f * tal) * (1.0 - f)
    f_on_src = (p.k_src_f * s) * (1.0 - f)
    act_f = f_on_i + f_on_talin + f_on_src
    deact_f = p.k_off_f * f

    act_s = (p.k_on_s * f) * (1.0 - s)
    deact_s = p.k_off_s * s

    act_p = (p.k_on_p * f + p.k_src_p * s) * (1.0 - pax)
    deact_p = p.k_off_p * pax

    act_e = (p.k_on_e * s) * (1.0 - erk)
    deact_e = p.k_off_e * erk

    act_r = (p.k_on_r * pax + p.k_force_r * p.force) * (1.0 - r)
    deact_r = p.k_off_r * r

    di = act_i - deact_i
    dtal = act_t - deact_t
    df = act_f - deact_f
    ds = act_s - deact_s
    dpax = act_p - deact_p
    derk = act_e - deact_e
    dr = act_r - deact_r

    return np.array([di, dtal, df, ds, dpax, derk, dr], dtype=float)


def fluxes(y: np.ndarray, p: Params) -> Dict[str, float]:
    """instantaneous forward/backward fluxes for the process view."""
    i, tal, f, s, pax, erk, r = y

    i_on_ecm = (p.k_on_i * p.ecm) * (1.0 - i)
    i_on_force = (p.k_force_i * p.force) * (1.0 - i)
    i_on_talin = (p.k_talin_i * tal) * (1.0 - i)
    i_on = i_on_ecm + i_on_force + i_on_talin
    i_off = p.k_off_i * i

    t_on_i = (p.k_on_t * i) * (1.0 - tal)
    t_on_force = (p.k_force_t * p.force) * (1.0 - tal)
    t_on = t_on_i + t_on_force
    t_off = p.k_off_t * tal

    f_on_i = (p.k_on_f * i) * (1.0 - f)
    f_on_talin = (p.k_talin_f * tal) * (1.0 - f)
    f_on_src = (p.k_src_f * s) * (1.0 - f)
    f_on = f_on_i + f_on_talin + f_on_src
    f_off = p.k_off_f * f

    s_on = (p.k_on_s * f) * (1.0 - s)
    s_off = p.k_off_s * s

    p_on_f = (p.k_on_p * f) * (1.0 - pax)
    p_on_src = (p.k_src_p * s) * (1.0 - pax)
    p_on = p_on_f + p_on_src
    p_off = p.k_off_p * pax

    e_on = (p.k_on_e * s) * (1.0 - erk)
    e_off = p.k_off_e * erk

    r_on_p = (p.k_on_r * pax) * (1.0 - r)
    r_on_force = (p.k_force_r * p.force) * (1.0 - r)
    r_on = r_on_p + r_on_force
    r_off = p.k_off_r * r

    return {
        "i_on": float(i_on),
        "i_on_ecm": float(i_on_ecm),
        "i_on_force": float(i_on_force),
        "i_on_talin": float(i_on_talin),
        "i_off": float(i_off),

        "t_on": float(t_on),
        "t_on_i": float(t_on_i),
        "t_on_force": float(t_on_force),
        "t_off": float(t_off),

        "f_on": float(f_on),
        "f_on_i": float(f_on_i),
        "f_on_talin": float(f_on_talin),
        "f_on_src": float(f_on_src),
        "f_off": float(f_off),

        "s_on": float(s_on),
        "s_off": float(s_off),

        "p_on": float(p_on),
        "p_on_f": float(p_on_f),
        "p_on_src": float(p_on_src),
        "p_off": float(p_off),

        "e_on": float(e_on),
        "e_off": float(e_off),

        "r_on": float(r_on),
        "r_on_p": float(r_on_p),
        "r_on_force": float(r_on_force),
        "r_off": float(r_off),
    }


def network_spec() -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """nodes + directed edges (edge_id) for a simple pathway diagram."""
    nodes = ["ecm", "force", "integrin", "talin", "fak", "src", "paxillin", "erk", "rhoa"]
    edges = [
        ("ecm", "integrin", "i_on_ecm"),
        ("force", "integrin", "i_on_force"),
        ("talin", "integrin", "i_on_talin"),

        ("integrin", "talin", "t_on_i"),
        ("force", "talin", "t_on_force"),

        ("integrin", "fak", "f_on_i"),
        ("talin", "fak", "f_on_talin"),
        ("src", "fak", "f_on_src"),

        ("fak", "src", "s_on"),

        ("fak", "paxillin", "p_on_f"),
        ("src", "paxillin", "p_on_src"),

        ("src", "erk", "e_on"),

        ("paxillin", "rhoa", "r_on_p"),
        ("force", "rhoa", "r_on_force"),
    ]
    return nodes, edges
