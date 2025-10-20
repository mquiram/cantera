#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lean_nrp_dual_mech.py
--------------------------------
Lean-flame stabilization with Gaussian NRP discharges (dual-mechanism scaffold)

Neutral (transported):  e.g., 'A2NOx-skeletal.yaml' (no e-/ions/excited)
Plasma (local 0-D):     e.g., 'A2NOx_hitest.yaml' (with e-/ions/excited)

Step 1:
  * Steady free flame with neutral mech
  * Electrode auto-placement near reaction zone
  * q_plas(x,t) Gaussian pulses (energy-only for now)
  * Map neutral state -> plasma mech (stub plasma sources)

Next steps (not implemented yet):
  * Transient operator splitting (energy-only first)
  * Add plasma species sources from Two-Term/EEDF
"""
from __future__ import annotations
import argparse, sys
from typing import Dict, List, Tuple
import numpy as np

try:
    import cantera as ct
except Exception:
    ct = None

# ----------------------------- utilities -----------------------------
def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / (sigma + 1e-300)
    return np.exp(-0.5 * z * z)

def pulse_centers(f_rep: float, n_pulses: int, t0: float = 0.0) -> np.ndarray:
    period = 1.0 / f_rep
    return t0 + np.arange(n_pulses) * period

def q_plasma_field(x: np.ndarray, t: float, x0: float, sigma_x: float,
                   q0_peak: float, f_rep: float, n_pulses: int, sigma_t: float,
                   t0: float = 0.0) -> np.ndarray:
    """Volumetric plasma power density q_plas(x, t) [W/m^3] = q0 * Gx(x) * Σ Gt(t; t_k)."""
    gx = gaussian(x, x0, sigma_x)
    tk = pulse_centers(f_rep, n_pulses, t0)
    gt = np.sum(np.exp(-0.5 * ((t - tk) / (sigma_t + 1e-300))**2))
    return q0_peak * gx * gt

# ----------------------------- neutral (transported) mechanism -----------------------------
def make_neutral_gas(neutral_mech: str, fuel: str, oxid: str, phi: float,
                     Tin: float, Pin: float, transport_model: str = 'mixture-averaged'):
    if ct is None:
        raise RuntimeError("Cantera not available; install cantera to run this script.")
    gasN = ct.Solution(neutral_mech)         # must exclude e-/ions/excited states
    gasN.transport_model = transport_model   # initialize a proper Transport for Flow1D
    gasN.TP = Tin, Pin
    gasN.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=oxid)
    return gasN

def solve_free_flame_neutral(gasN,
                             width: float = 0.04,
                             flame_transport: str = 'mixture-averaged',
                             loglevel: int = 0,
                             energy_enabled: bool = True):
    flame = ct.FreeFlame(gasN, width=width)
    print("setting flame transport model to:", flame_transport)
    flame.transport_model = flame_transport  # 'mixture-averaged' | 'UnityLewis' | 'Multi'
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.10)
    print("solving neutral free flame...")
    flame.solve(loglevel=loglevel, auto=True, refine_grid=True)
    print("neutral free flame solved.")
    if energy_enabled:
        flame.energy_enabled = True
        print("re-solving with energy equation enabled...")
        flame.solve(loglevel=loglevel, refine_grid=True)
    return flame

def find_reaction_zone_center(flame) -> float:
    x = flame.grid
    dTdx = np.gradient(flame.T, x, edge_order=2)
    return x[np.argmax(np.abs(dTdx))]

def export_profiles(flame) -> Dict[str, np.ndarray]:
    prof = {
        'x': flame.grid.copy(),
        'T': flame.T.copy(),
        'u': flame.velocity.copy(),
        'rho': flame.density.copy(),
        'P': np.full_like(flame.grid, flame.P)
    }
    x   = np.array(flame.grid, copy=True)
    T   = np.array(flame.T,    copy=True)
    rho = np.array(flame.density, copy=True)
    u   = np.array(flame.velocity, copy=True)
    P   = np.full_like(x, flame.P)
    # --- species mass fractions as (N_points, N_species) ---
    nsp  = flame.gas.n_species
    npts = x.size
    Yobj = flame.Y               # may be a 2D array or an indexable container

    Ymat = np.asarray(Yobj)
    if Ymat.ndim == 2:
        # Accept either (points, species) or (species, points)
        if Ymat.shape == (npts, nsp):
            Y = Ymat.copy()
        elif Ymat.shape == (nsp, npts):
            Y = Ymat.T.copy()
        else:
            # last resort: try per-species indexing
            Y = np.vstack([np.array(Yobj[k], copy=True) for k in range(nsp)]).T
    else:
        # Y is not a 2D matrix; build it species-by-species
        Y = np.vstack([np.array(Yobj[k], copy=True) for k in range(nsp)]).T

    return {"x": x, "T": T, "u": u, "rho": rho, "P": P,
            "Y": Y, "species": flame.gas.species_names}

# ----------------------------- plasma (local micro-solver) mechanism -----------------------------
def neutral_to_plasma_composition(gasN) -> Dict[str, float]:
    """Neutral mole-fraction dict to seed the plasma mechanism (for overlapping species)."""
    namesN = gasN.species_names
    XN = gasN.X
    return {namesN[i]: float(XN[i]) for i in range(len(namesN))}

def make_plasma_gas(plasma_mech: str, T: float, P: float, comp_from_neutral: Dict[str, float]):
    """Create plasma Solution at (T,P) with composition mapped from the neutral mix.
    We do NOT set a transport model on the plasma gas.
    """
    gasP = ct.Solution(plasma_mech)
    Xp = {name: 0.0 for name in gasP.species_names}
    for sp, x in comp_from_neutral.items():
        if sp in Xp:
            Xp[sp] = float(x)
    s = sum(Xp.values())
    if s <= 0.0:
        # fallback if neutral->plasma mapping had no overlap
        if 'N2' in Xp: Xp['N2'] = 0.79
        if 'O2' in Xp: Xp['O2'] = 0.21
        s = sum(Xp.values())
    if s > 0.0 and abs(s - 1.0) > 1e-12:
        inv = 1.0 / s
        for k in Xp:
            Xp[k] *= inv
    gasP.TPX = float(T), float(P), Xp
    return gasP

# ----------------------------- plasma sources (stub for now) -----------------------------
def plasma_species_sources_stub(gasP) -> np.ndarray:
    """Return molar production rates (kmol/m^3/s) from plasma micro-solver.
    Replace with your Two-Term/EEDF + reactions. Length = gasP.n_species.
    """
    return np.zeros(gasP.n_species)

# ----------------------------- plotting (optional) -----------------------------
def demo_plot_q(x: np.ndarray, q_snaps: List[Tuple[float, np.ndarray]]):
    import matplotlib.pyplot as plt
    for (t, qx) in q_snaps:
        plt.figure()
        plt.plot(x, qx, lw=2)
        plt.xlabel('x [m]'); plt.ylabel('q_plas(x, t) [W/m^3]')
        plt.title(f't = {t*1e6:.3f} µs'); plt.tight_layout()
    plt.show()

# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Lean flame + Gaussian NRP (dual-mechanism scaffold)")
    ap.add_argument('--neutral', type=str, default='A2NOx_skeletal.yaml', help='Neutral (transported) mechanism')
    ap.add_argument('--plasma',  type=str, default='A2NOx_hitest.yaml', help='Plasma (local) mechanism')
    ap.add_argument('--fuel',    type=str, default='CH4:1')
    ap.add_argument('--oxid',    type=str, default='O2:1, N2:3.76')
    ap.add_argument('--phi',     type=float, default=0.65)
    ap.add_argument('--Tin',     type=float, default=300.0)
    ap.add_argument('--P',       type=float, default=101325.0)
    ap.add_argument('--width',   type=float, default=0.08) #0.04
    ap.add_argument('--transport', type=str, default='mixture-averaged', choices=['mixture-averaged','UnityLewis','Multi'])
    # NRP params
    ap.add_argument('--freq',    type=float, default=10e3) #1e4
    ap.add_argument('--sigma_t', type=float, default=5e-9)
    ap.add_argument('--sigma_x', type=float, default=3e-4)
    ap.add_argument('--q0',      type=float, default=3.0e9)
    ap.add_argument('--npulses', type=int,   default=100)
    ap.add_argument('--plot',    action='store_true')
    return ap.parse_args()

def main():
    args = parse_args()
    if ct is None:
        raise RuntimeError("Cantera not available; install cantera to run this script.")

    # ---- neutral baseline
    gasN = make_neutral_gas(args.neutral, args.fuel, args.oxid, args.phi, args.Tin, args.P,
                            transport_model='mixture-averaged' if args.transport!='UnityLewis' else 'mixture-averaged')
    print("neutral gas created")
    flame = solve_free_flame_neutral(gasN, width=args.width,
                                     flame_transport=('mixture-averaged' if args.transport=='UnityLewis' else args.transport))
    print("neutral flame solved")
    if args.transport == 'UnityLewis':
        flame.transport_model = 'UnityLewis'
        flame.solve(loglevel=0, refine_grid=False)

    prof = export_profiles(flame)
    print("neutral profiles exported")
    x   = prof['x']
    x0  = find_reaction_zone_center(flame)
    print(f"electrode center x0 set to reaction zone at x = {x0:.6e} m")

    # ---- build plasma gas at same (T,P) with neutral composition mapped over
    neutral_comp = neutral_to_plasma_composition(gasN)
    print("neutral to plasma composition mapping done")
    gasP = make_plasma_gas(args.plasma, T=float(gasN.T), P=float(gasN.P), comp_from_neutral=neutral_comp)
    print("plasma gas created")

    # ---- q_plas snapshots (demo)
    times = [0.0, 5e-6, 10e-6]
    snaps = [(t, q_plasma_field(x, t, x0, args.sigma_x, args.q0, args.freq, args.npulses, args.sigma_t)) for t in times]
    if args.plot:
        try:
            demo_plot_q(x, snaps)
        except Exception as e:
            print(f"[WARN] Plot failed: {e}", file=sys.stderr)

    # ---- console summary
    namesN = set(gasN.species_names)
    namesP = set(gasP.species_names)
    print("=== Dual-Mechanism Baseline ===")
    print(f"Neutral mech: {args.neutral}  | Plasma mech: {args.plasma}")
    print(f"phi={args.phi:.3f}, Tin={args.Tin:.1f} K, P={args.P/101325.0:.3f} atm")
    print(f"Grid points: {len(x)}, domain length = {x[-1]-x[0]:.6f} m")
    print(f"Electrode center x0 ≈ {x0:.6e} m")
    print(f"Species: neutral={len(namesN)}, plasma={len(namesP)}, overlap={len(namesN & namesP)}")
    examples_plasma_only = sorted(list(namesP - namesN))[:8]
    if examples_plasma_only:
        print(f"Plasma-only examples (not transported): {examples_plasma_only}")
    print("\n=== q_plas snapshots (min/mean/max) ===")
    for t, qx in snaps:
        print(f"t={t:.2e} s -> ({qx.min():.3e}, {qx.mean():.3e}, {qx.max():.3e}) W/m^3")

    print("\nNext steps:")
    print("  1) Add transient operator splitting over the neutral grid:")
    print("     (a) plasma sub-step (local, using gasP) -> q_plas, omega_plas")
    print("     (b) chemistry sub-step (local, using gasN kinetics)")
    print("     (c) transport step (1-D, mixture-averaged; start const-P/low-Mach)")
    print("  2) Begin with energy-only q_plas to validate numerics; then add species sources.")
    print("  3) In plasma sub-step, call your Two-Term/EEDF and restrict sources to electrode cells.")

if __name__ == "__main__":
    main()