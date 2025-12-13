from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NFkBParams:
    """minimal nf-kb negative feedback oscillator.

    state: x = [nn, im, i]
      nn: nuclear nf-kb (a.u.)
      im: iκb mrna (a.u.)
      i:  iκb protein (a.u.)

    input:
      ikk(t): kinase activity driving iκb degradation and nf-kb import
    """

    ntot: float = 1.0

    k_imp: float = 2.5
    k_exp: float = 1.2

    k_tx: float = 2.0
    k_mdeg: float = 0.6

    k_tl: float = 1.6
    k_pdeg: float = 0.2
    k_ikk: float = 1.0

    hill_h: float = 3.0
    hill_k: float = 0.25


def nfkb_rhs(t: float, x: np.ndarray, p: NFkBParams, ikk: float) -> np.ndarray:
    nn, im, i = x

    nn = float(max(nn, 0.0))
    im = float(max(im, 0.0))
    i = float(max(i, 0.0))

    ncyto = max(p.ntot - nn, 0.0)

    # nuclear import increases with ikk; export increases with inhibitor-bound complex (proxy via i).
    dnn = (p.k_imp * ikk * ncyto) - (p.k_exp * i * nn)

    # iκb transcription activated by nuclear nf-kb with cooperativity.
    act = (nn ** p.hill_h) / (p.hill_k ** p.hill_h + nn ** p.hill_h)
    dim = (p.k_tx * act) - (p.k_mdeg * im)

    # iκb translation; degradation increases with ikk.
    di = (p.k_tl * im) - ((p.k_pdeg + p.k_ikk * ikk) * i)

    return np.array([dnn, dim, di], dtype=float)
