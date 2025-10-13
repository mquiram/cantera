import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
ct.CanteraError.set_stack_trace_depth(10)

# --- EEDF / E/N probe helpers ---
_KB  = 1.380649e-23     # J/K
_eV  = 1.602176634e-19  # J
_Td  = 1.0e-21          # 1 Td = 1e-21 V·m^2

def _neutral_number_density(gas):
    # Ideal-gas estimate of total neutrals (m^-3). For your pressure range this is fine.
    return gas.P / (_KB * gas.T)

def _trycall(obj, name):
    if hasattr(obj, name):
        attr = getattr(obj, name)
        return attr() if callable(attr) else attr
    return None

def _print_eedf_snapshot(self, gas, tag="pre-avalanche"):
    # Grab the reduced field the way PlasmaPhase expects it
    EN_now = getattr(gas, "EN", None)

    # EEDF object is optional; probe a few common attributes if present
    eedf = getattr(gas, "eedf", None)
    Te = None
    mu = None
    try:
        # try several common names; all optional
        Te = getattr(eedf, "Te", None) or getattr(eedf, "meanEnergy", None) or getattr(eedf, "mean_energy", None)
        mu = getattr(eedf, "electronMobility", None) or getattr(eedf, "mobility", None)
    except Exception:
        pass

    print(f"[eedf:{tag}] EN={EN_now} | Te={Te} | mu={mu}")

""" def _save_state(gas: ct.Solution, fname: str):
    # Store a minimal reproducible state
    np.savez(fname.replace(".yaml",".npz"),
             T=gas.T, P=gas.P, X=gas.X)
    # Also human-readable composition
    with open(fname, "w") as f:
        f.write(f"T: {gas.T}\nP: {gas.P}\n")
        for k, Xk in zip(gas.species_names, gas.X):
            if Xk > 0:
                f.write(f"{k}: {Xk:.16e}\n") """

def _stoich_mats(gas: ct.Solution):
    # Return ν' (reactants), ν'' (products) shaped as (ns, nr)
    nu_r = gas.reactant_stoich_coeffs
    nu_p = gas.product_stoich_coeffs
    # Ensure shape (ns, nr)
    if nu_r.shape[0] != gas.n_species:   # came as (nr, ns)
        nu_r = nu_r.T
        nu_p = nu_p.T
    return nu_r, nu_p                     # (ns, nr)

def _rxn_enthalpies(gas: ct.Solution):
    # Δh_j (J/kmol) for each reaction at current state
    nu_r, nu_p = _stoich_mats(gas)
    nu = nu_p - nu_r                      # (ns, nr)
    h_k = gas.partial_molar_enthalpies    # (ns,)
    dH = nu.T @ h_k                       # (nr,)
    return dH

def _electron_impact_mask(gas: ct.Solution):
    # True if e- appears on the LHS of reaction j
    eqs = gas.reaction_equations()
    lhs_has_e = []
    for s in eqs:
        lhs = s.split('<=>')[0] if '<=>' in s else s.split('=>')[0]
        lhs = ' ' + lhs.replace('+', ' ') + ' '
        lhs_has_e.append(' e ' in lhs)
    return np.array(lhs_has_e, dtype=bool)

def _electron_budget(gas: ct.Solution):
    # Per-reaction electron source s_j = ν_ej * ROP_j (kmol e-/m^3/s)
    eidx = gas.species_index('e')
    nu_r, nu_p = _stoich_mats(gas)
    nu_e = (nu_p - nu_r)[eidx, :]         # (nr,)
    rop = gas.net_rates_of_progress       # (nr,)
    s = nu_e * rop
    return s, nu_e, rop                   # s is net e- source by reaction

def _qchem_qe(gas: ct.Solution):
    dH = _rxn_enthalpies(gas)             # (nr,) J/kmol
    rop = gas.net_rates_of_progress       # (nr,) kmol/m^3/s
    qchem = -np.sum(dH * rop)             # W/m^3
    emask = _electron_impact_mask(gas)
    qe = -np.sum(dH[emask] * rop[emask])  # W/m^3 from e-impact subset
    return qchem, qe, dH, rop, emask

def _dump_electron_breakdown(gas: ct.Solution, top=15, tag="probe"):
    qchem, qe, dH, rop, emask = _qchem_qe(gas)
    s, nu_e, _ = _electron_budget(gas)
    eqs = gas.reaction_equations()

    # Top electron sources and sinks
    src_idx = np.argsort(-s)[:top]
    snk_idx = np.argsort(s)[:top]

    print(f"[e-probe:{tag}] T={gas.T:.2f} K, P={gas.P/ct.one_atm:.3f} atm")
    print(f"[e-probe:{tag}] ne={gas.concentrations[gas.species_index('e')]:.3e} kmol/m^3")
    print(f"[e-probe:{tag}] q_chem={qchem:.3e} W/m^3, q_e={qe:.3e} W/m^3")
    print(f"[e-probe:{tag}] ROP_e_abs={np.sum(np.abs(rop[emask])):.3e} (kmol/m^3/s)")

    print(f"[e-probe:{tag}] top e- SOURCES (ν_e*ROP > 0):")
    for j in src_idx:
        if s[j] <= 0: break
        print(f"   #{j:4d}  +{s[j]:.3e}  :: {eqs[j]}")

    print(f"[e-probe:{tag}] top e- SINKS (ν_e*ROP < 0):")
    for j in snk_idx:
        if s[j] >= 0: break
        print(f"   #{j:4d}  {s[j]:.3e}  :: {eqs[j]}")

    # Which e-impact reactions are cooling the gas the most?
    e_power = -(dH * rop)                 # per-reaction heat (W/m^3), sign=heating
    e_idx = np.where(emask)[0]
    if e_idx.size:
        top_cool = e_idx[np.argsort(e_power[e_idx])[:top]]  # most negative heating
        print(f"[e-probe:{tag}] strongest e-impact COOLING (q_j most negative):")
        for j in top_cool:
            print(f"   #{j:4d}  q_j={e_power[j]:.3e} W/m^3 :: {eqs[j]}")

def _species_index_safe(gas, name_candidates=("e", "E", "electron")):
    # robustly find the electron species index (or None)
    try:
        return gas.species_index("e")
    except Exception:
        names = [s.name for s in gas.species()]
        for nm in name_candidates:
            if nm in names:
                return names.index(nm)
    return None

def _electron_impact_mask(gas):
    """Boolean mask of length n_reactions: True if e⁻ is a reactant."""
    e_idx = _species_index_safe(gas)
    if e_idx is None:
        return np.zeros(gas.n_reactions, dtype=bool)

    # reactant stoich matrix; orientation varies by build (and may be sparse)
    nu_r = gas.reactant_stoich_coeffs
    nsp, nrx = gas.n_species, gas.n_reactions

    if nu_r.shape == (nsp, nrx):        # species × reactions
        row = nu_r[e_idx, :]
        mask = (row > 0)
    elif nu_r.shape == (nrx, nsp):      # reactions × species
        col = nu_r[:, e_idx]
        mask = (col > 0)
    else:
        raise RuntimeError(f"Unexpected reactant stoich shape {nu_r.shape}")

    # Convert sparse/boolish to plain 1D boolean array
    try:
        mask = mask.A.ravel()  # if it’s a sparse matrix
    except Exception:
        mask = np.asarray(mask).ravel()
    return mask

def _rxn_qdot_vector(gas):
    # J/kmol for each species (length n_species)
    h_k = gas.partial_molar_enthalpies

    # stoich matrix (can be dense or sparse; orientation varies by build)
    nu = gas.product_stoich_coeffs - gas.reactant_stoich_coeffs

    nsp, nrx = gas.n_species, gas.n_reactions

    # Compute ΔH_rxn_j = Σ_k nu[k,j] * h_k[k], robustly
    if nu.shape == (nsp, nrx):
        # species × reactions
        # if sparse, left-multiply with nu.T; if dense, right-multiply with h_k @ nu
        if hasattr(nu, "toarray") or "sparse" in type(nu).__name__.lower():
            dH_rxn = nu.T @ h_k          # (n_reactions,)
        else:
            dH_rxn = h_k @ nu            # (n_reactions,)
    elif nu.shape == (nrx, nsp):
        # reactions × species
        dH_rxn = nu @ h_k                # (n_reactions,)
    else:
        raise RuntimeError(
            f"Unexpected stoich matrix shape {nu.shape}; expected {(nsp, nrx)} or {(nrx, nsp)}"
        )

    # kmol/m^3/s (net)
    rop_net = gas.net_rates_of_progress

    # W/m^3 per reaction; positive means heating
    qvec = -dH_rxn * rop_net

    return np.asarray(qvec), np.asarray(dH_rxn)

def _top_reaction_contribs(gas, n=10):
    qvec, dH = _rxn_qdot_vector(gas)
    idx = np.argsort(np.abs(qvec))[-n:][::-1]
    lines = []
    for i in idx:
        lines.append(f"   #{i:4d} {qvec[i]:+.3e}  :: {gas.reaction_equation(i)}")
    return "\n".join(lines)

def _scan_nonfinite_rates(gas, rxn_ids=None):
    """
    Return list of (i, kf, kr, rop, eqn) for reactions with non-finite kf/kr/rop.
    If rxn_ids is given, only scan those.
    """
    kf = gas.forward_rate_constants
    kr = gas.reverse_rate_constants
    rop = gas.net_rates_of_progress
    bad = []
    if rxn_ids is None:
        rxn_ids = range(gas.n_reactions)
    for i in rxn_ids:
        if not np.isfinite(kf[i]) or not np.isfinite(kr[i]) or not np.isfinite(rop[i]):
            bad.append((i, kf[i], kr[i], rop[i], gas.reaction_equation(i)))
    return bad

def _energy_budget(gas, reactor):
    qvec, _ = _rxn_qdot_vector(gas)
    q_chem = float(np.sum(qvec))
    rho_cp = gas.density * gas.cp_mass  # J/m^3/K
    dTdt_est = q_chem / rho_cp if rho_cp > 0 else np.nan
    q_wall = 0.0  # or reactor heat-loss term if you wire one in
    return q_chem, q_wall, dTdt_est, rho_cp

def _energy_watch(gas, reactor, tag=""):
    q_chem, q_wall, dTdt, rho_cp = _energy_budget(gas, reactor)

    # compute per-call instead of caching on `gas`
    e_mask = _electron_impact_mask(gas)

    try:
        # E from EN and ideal gas N
        kB = 1.380649e-23
        N = gas.P / (kB * gas.T)
        E = float(gas.EN) * N                # V/m

        # electron density in mol/m^3 (Cantera uses kmol/m^3)
        i_e = gas.species_index("e")
        ne_molm3 = float(gas.concentrations[i_e]) * 1e3

        # mobility: try both camel/snake names
        mu = getattr(gas, "electronMobility", None)
        if mu is None:
            mu = getattr(gas, "electron_mobility", None)
        mu = float(mu) if mu is not None else None

        q_joule = None
        if mu is not None:
            F = 96485.33212  # C/mol
            J = ne_molm3 * F * mu * E      # A/m^2
            q_joule = J * E                # W/m^3

        print(f"[eedf:pre-avalanche] EN={gas.EN:.3e} V·m^2 (~{gas.EN/1e-21:.1f} Td) | "
            f"E≈{E:.2e} V/m | mu={mu} | q_J={q_joule}")
    except Exception as ex:
        print(f"[eedf:pre-avalanche] Joule calc failed: {ex}")

    qvec, _ = _rxn_qdot_vector(gas)
    q_chem = float(np.sum(qvec))

    # add:
    e_mask = _electron_impact_mask(gas)
    q_e = float(np.sum(qvec[e_mask])) if e_mask.size else 0.0
    q_chem_heavy = q_chem - q_e      # treat e-impact as electron-mode energy, not heavy gas
    dTdt_est = q_chem_heavy / rho_cp
    rop = gas.net_rates_of_progress

    e_q = float(np.sum(qvec[e_mask])) if e_mask.size else 0.0
    e_rop_sum = float(np.sum(rop[e_mask])) if e_mask.size else 0.0

    print(f"[{tag}] T={gas.T:.2f} K, P={gas.P/1e5:.3f} bar | "
          f"q_chem={q_chem:.3e} W/m^3 | dT/dt≈{dTdt:.3e} K/s | "
          f"ROP_e={e_rop_sum:.3e} | q_e={e_q:.3e}")

    ok = np.all(np.isfinite([q_chem, dTdt])) and abs(dTdt) < 1e18
    return ok

def _hard_lowT_vt_guard(gas, vt_idx, Tcut=200.0, hyst=20.0, verbose=False):
    """
    Hard guard for VT reactions: below (Tcut - hyst) replace them with zero-rate
    Arrhenius so kf stays finite; above (Tcut + hyst) restore the originals.
    This prevents inf*0 -> NaN downstream in ROP.

    Keeps a per-gas backup of the original Reaction objects.
    """
    import cantera as ct
    kin = gas.kinetics
    T = gas.T

    # scratchpad on the thermo object so it follows the reactor's Solution
    if not hasattr(gas, "_vt_guard"):
        gas._vt_guard = {"disabled": False, "backup": {}}

    disabled = gas._vt_guard["disabled"]

    # go LOW: disable
    if (T < Tcut - hyst) and not disabled:
        if verbose:
            print(f"[vtguard] T={T:.3f} K < {Tcut-hyst:.1f} K -> DISABLE VT ({len(vt_idx)} rxns)")
        for j in vt_idx:
            if j not in gas._vt_guard["backup"]:
                r_old = kin.reaction(j)  # original Reaction object
                gas._vt_guard["backup"][j] = r_old
            r0 = ct.ElementaryReaction(
                gas._vt_guard["backup"][j].reactants,
                gas._vt_guard["backup"][j].products,
                rate=ct.Arrhenius(0.0, 0.0, 0.0)
            )
            kin.modify_reaction(int(j), r0)
        gas._vt_guard["disabled"] = True

    # go HIGH: restore
    elif (T > Tcut + hyst) and disabled:
        if verbose:
            print(f"[vtguard] T={T:.3f} K > {Tcut+hyst:.1f} K -> RESTORE VT")
        for j, r_old in gas._vt_guard["backup"].items():
            kin.modify_reaction(int(j), r_old)
        gas._vt_guard["backup"].clear()
        gas._vt_guard["disabled"] = False

def stress_T_sweep(gas, deltas=(-1.0, -2.0, -5.0, -10.0)):
    snap = _snapshot(gas)
    eqns = [rxn.equation for rxn in gas.reactions()]
    # cache VT indices once
    vt_idx = getattr(gas, "_vt_idx_for_sweep", None)
    if vt_idx is None:
        vt_idx = _find_vt_reactions(gas)
        gas._vt_idx_for_sweep = vt_idx
    for dT in deltas:
        _safe_restore(gas, snap)
        Ttest = max(1.0, gas.T + dT)
        gas.TP = Ttest, gas.P
        # **apply the same hard guard used in the solver**
        _hard_lowT_vt_guard(gas, vt_idx, Tcut=200.0, hyst=20.0, verbose=False)
        try:
            kf = gas.forward_rate_constants
            kr = gas.reverse_rate_constants
            fr = gas.forward_rates_of_progress
            rr = gas.reverse_rates_of_progress
            rop = fr - rr
            dH = gas.delta_enthalpy
            qdot = -np.dot(rop, dH)
            nonfinite = (~np.isfinite(kf)).any() or (~np.isfinite(kr)).any() or (~np.isfinite(rop)).any()
            print(f"[sweep] T-{-dT:.0f}K => T={Ttest:.3f} K: qdot={qdot:.3e} W/m^3; nonfinite={nonfinite}")
            if nonfinite:
                bad = np.where((~np.isfinite(kf)) | (~np.isfinite(kr)) | (~np.isfinite(rop)))[0]
                print("   bad reactions (first 10):")
                for i in bad[:10]:
                    print(f"     #{i:4d}  kf={kf[i]}, kr={kr[i]}, rop={rop[i]} :: {eqns[i]}")
        except Exception as ex:
            print(f"[sweep] T={Ttest:.3f} K evaluation raised: {ex}")
    _safe_restore(gas, snap)

def _safe_restore(gas, snapshot):
    T, P, X = snapshot
    gas.TPX = T, P, X

def _snapshot(gas):
    T, P = gas.TP
    X = gas.X  # copies to a NumPy array
    return (T, P, X)

def summarize_state(gas):
    rho = gas.density
    cp  = gas.cp_mass
    cv  = gas.cv_mass
    gam = cp / cv if (cp > 0 and cv > 0) else np.nan
    print(f"[state] T={gas.T:.6g} K, P={gas.P/1e5:.6g} bar, rho={rho:.6g} kg/m^3, cp={cp:.6g} J/kg/K")
    # electrons / ions (if present)
    def conc_of(sp):
        try:
            k = gas.species_index(sp)
            return gas.concentrations[k]  # kmol/m^3
        except Exception:
            return None
    for sp in ["e", "H+", "O2+", "H2+", "NO+", "N2+"]:
        c = conc_of(sp)
        if c is not None:
            print(f"[plasma] c({sp}) = {c:.3e} kmol/m^3")
    # vibrational/excited N2 if present
    for sp in ["N2(v1)","N2(v2)","N2(v3)","N2(v4)","N2(v5)","N2(v6)","N2(v7)","N2(v8)"]:
        c = conc_of(sp)
        if c is not None:
            print(f"[vib] c({sp}) = {c:.3e} kmol/m^3")

def heat_budget(gas, top_n=10):
    """
    Compute volumetric heat release/absorption rate from chemistry and list the
    top cooling/heating reactions.
    """
    rop_f = gas.forward_rates_of_progress   # kmol/m^3/s
    rop_r = gas.reverse_rates_of_progress
    rop   = rop_f - rop_r                   # net
    dH    = gas.delta_enthalpy              # J/kmol (molar enthalpy change)
    qdot  = -np.dot(rop, dH)                # J/m^3/s   (negative dH & positive ROP => heat release)
    dTdt  = qdot / (gas.density * gas.cp_mass)  # K/s, approximate
    print(f"[heat] qdot = {qdot:.6e} W/m^3  =>  dT/dt ≈ {dTdt:.6e} K/s")

    # rank reactions by their contribution to qdot magnitude
    contrib = -(rop * dH)  # J/m^3/s per reaction (positive = heating, negative = cooling)
    idx_sorted = np.argsort(np.abs(contrib))[::-1]
    eqns = [rxn.equation for rxn in gas.reactions()]
    print(f"[heat] top contributing reactions (heating/cooling):")
    for i in idx_sorted[:top_n]:
        sign = "+" if contrib[i] >= 0 else "-"
        print(f"   #{i:4d} {sign} {abs(contrib[i]):.3e}  :: {eqns[i]}")

    # strongest coolers (negative contrib)
    idx_cool = np.where(contrib < 0)[0]
    if idx_cool.size:
        cool_sorted = idx_cool[np.argsort(contrib[idx_cool])]  # most negative first
        print(f"[heat] strongest COOLING reactions:")
        for i in cool_sorted[:min(top_n, len(cool_sorted))]:
            print(f"   #{i:4d} {contrib[i]:.3e}  :: {eqns[i]}")

def species_budget(gas, species_keys, top_n=8):
    """
    For each target species index or name, list the reactions that
    produce/consume it the most at the current state.
    """
    if not species_keys:
        return
    rop_f = gas.forward_rates_of_progress
    rop_r = gas.reverse_rates_of_progress
    rop   = rop_f - rop_r
    eqns  = [rxn.equation for rxn in gas.reactions()]
    rxns  = gas.reactions()

    # build a list of net stoich dicts for speed
    net_stoich = []
    name_to_idx = {s:i for i,s in enumerate(gas.species_names)}
    for rxn in rxns:
        nu = {}
        for s, v in rxn.reactants.items():
            nu[name_to_idx[s]] = nu.get(name_to_idx[s], 0.0) - v
        for s, v in rxn.products.items():
            nu[name_to_idx[s]] = nu.get(name_to_idx[s], 0.0) + v
        net_stoich.append(nu)

    for key in species_keys:
        sk = key if isinstance(key, int) else gas.species_index(key)
        sname = gas.species_names[sk]
        # contribution to dC_k/dt from each reaction ~ nu_k * rop
        contrib = np.array([net_stoich[i].get(sk, 0.0) * rop[i] for i in range(len(rxns))])
        idx_sorted = np.argsort(np.abs(contrib))[::-1]
        print(f"[sp] {sname}: top production contributions:")
        for i in idx_sorted[:top_n]:
            sign = "+" if contrib[i] >= 0 else "-"
            print(f"   #{i:4d} {sign} {abs(contrib[i]):.3e}  :: {eqns[i]}")

def chebyshev_range_check(gas):
    """
    Warn if any Chebyshev reaction is evaluated outside its declared T/P window.
    """
    T = gas.T
    P = gas.P
    any_warn = False
    for i, rxn in enumerate(gas.reactions()):
        if rxn.__class__.__name__.lower().startswith("chebyshev"):
            Tmin = getattr(rxn, "Tmin", None)
            Tmax = getattr(rxn, "Tmax", None)
            Pmin = getattr(rxn, "Pmin", None)
            Pmax = getattr(rxn, "Pmax", None)
            badT = (Tmin is not None and T < Tmin) or (Tmax is not None and T > Tmax)
            badP = (Pmin is not None and P < Pmin) or (Pmax is not None and P > Pmax)
            if badT or badP:
                any_warn = True
                print(f"[cheb] OUT-OF-RANGE: rxn #{i}  T∈[{Tmin},{Tmax}]  P∈[{Pmin},{Pmax}]   @ T={T:.3f}, P={P:.3e}  :: {rxn.equation}")
    if not any_warn:
        print("[cheb] Chebyshev reactions within T/P ranges.")

def stress_T_sweep(gas, deltas=(-1.0, -2.0, -5.0, -10.0)):
    """
    Evaluate kf/kr/ROP and qdot a few K below current T to see if anything *would*
    blow up just under the accepted state. Non-destructive (restores state).
    """
    snap = _snapshot(gas)
    eqns = [rxn.equation for rxn in gas.reactions()]
    for dT in deltas:
        _safe_restore(gas, snap)
        Ttest = max(1.0, gas.T + dT)  # keep positive
        gas.TP = Ttest, gas.P
        try:
            kf = gas.forward_rate_constants
            kr = gas.reverse_rate_constants
            fr = gas.forward_rates_of_progress
            rr = gas.reverse_rates_of_progress
            rop = fr - rr
            dH = gas.delta_enthalpy
            qdot = -np.dot(rop, dH)
            nonfinite = (~np.isfinite(kf)).any() or (~np.isfinite(kr)).any() or (~np.isfinite(rop)).any()
            print(f"[sweep] T-{-dT:.0f}K => T={Ttest:.3f} K: qdot={qdot:.3e} W/m^3; nonfinite={nonfinite}")
            if nonfinite:
                bad = np.where((~np.isfinite(kf)) | (~np.isfinite(kr)) | (~np.isfinite(rop)))[0]
                print("   bad reactions (first 10):")
                for i in bad[:10]:
                    print(f"     #{i:4d}  kf={kf[i]}, kr={kr[i]}, rop={rop[i]} :: {eqns[i]}")
        except Exception as ex:
            print(f"[sweep] T={Ttest:.3f} K evaluation raised: {ex}")
    _safe_restore(gas, snap)

def dump_pre_failure_diagnostics(gas, focus_species=(1,18,23,207,208,220,221,222,223,224,225,226,227)):
    print("\n=== PRE-FAILURE DIAGNOSTICS ===")
    summarize_state(gas)
    heat_budget(gas, top_n=10)
    species_budget(gas, focus_species, top_n=6)
    chebyshev_range_check(gas)
    stress_T_sweep(gas, deltas=(-0.5, -1.0, -2.0, -5.0, -10.0))
    print("=== END DIAGNOSTICS ===\n")

def print_species_index_map(gas):
    print("\n=== Species index map ===")
    for k, s in enumerate(gas.species_names):
        print(f"{k:4d}: {s}")

def diagnose_nonfinite_reactions(gas, focus_species=(1, 18, 20, 220), auto_disable=False):
    """
    Inspect current state for NaN/Inf in rate constants and rates of progress.
    Optionally disables offending reactions (use with care).
    """
    # Pull reaction list and equations
    reactions = gas.reactions()                    # list[Reaction]
    eqns = [rxn.equation for rxn in reactions]

    # Arrays at *current* state
    kf = gas.forward_rate_constants
    kr = gas.reverse_rate_constants
    fr = gas.forward_rates_of_progress
    rr = gas.reverse_rates_of_progress

    bad_k = np.where(~np.isfinite(kf) | ~np.isfinite(kr))[0]
    bad_r = np.where(~np.isfinite(fr) | ~np.isfinite(rr))[0]

    print(f"\n[diag] T={gas.T:.3f} K, P={gas.P/1e5:.3f} bar")
    if bad_k.size == 0 and bad_r.size == 0:
        print("[diag] All kf/kr and ROP are finite at this state.")
        return

    if bad_k.size:
        print("\n[diag] Non-finite rate constants:")
        for i in bad_k[:20]:
            print(f"  #{i:4d}  kf={kf[i]}  kr={kr[i]}  :: {eqns[i]}")
    if bad_r.size:
        print("\n[diag] Non-finite rates of progress:")
        for i in bad_r[:20]:
            print(f"  #{i:4d}  fr={fr[i]}  rr={rr[i]}  :: {eqns[i]}")

    offenders = np.unique(np.concatenate([bad_k, bad_r])).astype(int)
    if offenders.size:
        print("\n[diag] Net contribution to focus species from first offenders:")
        for i in offenders[:10]:
            rxn = reactions[i]
            # Build net stoich vector
            nu = np.zeros(gas.n_species)
            for sp, nu_r in rxn.reactants.items():
                nu[gas.species_index(sp)] -= nu_r
            for sp, nu_p in rxn.products.items():
                nu[gas.species_index(sp)] += nu_p
            rop = fr[i] - rr[i]
            for sk in focus_species:
                if 0 <= sk < gas.n_species:
                    print(f"  rxn #{i:4d} -> dY[{sk}] ~ {nu[sk]*rop: .3e}   :: {eqns[i]}")

    if auto_disable and offenders.size:
        print("\n[diag] Disabling offending reactions (temporary)…")
        for i in offenders:
            _set_multiplier(gas, int(i), 0.0)
            print(f"  disabled #{i:4d} :: {eqns[i]}")
        print("[diag] Done.")

def disable_reactions_by_predicate(gas, predicate):
    """Utility to mass-disable reactions that satisfy a condition on the Reaction object."""
    kin = gas.kinetics
    n = 0
    for i, rxn in enumerate(kin.reactions()):
        if predicate(i, rxn):
            kin.set_multiplier(0.0, i)
            n += 1
    return n

def _find_vt_reactions(gas):
    vt = []
    for i, rxn in enumerate(gas.reactions()):
        eq = rxn.equation
        if "N2(v" in eq and "N2 +" in eq and "=>" in eq:
            vt.append(i)
        if "N2(v" in eq and "+ O2" in eq and "=>" in eq:
            vt.append(i)
    return sorted(set(vt))

def _apply_lowT_vt_clamp(gas, vt_idx, Tcut=200.0, delta=20.0):
    """
    Smooth logistic clamp: ~1 above Tcut, ~0 well below Tcut.
    Keeps kf finite without hard discontinuities.
    """
    import math
    T = gas.T
    s = 1.0 / (1.0 + math.exp((Tcut - T) / delta))
    for j in vt_idx:
        gas.set_multiplier(s, j)

class PSR_Plasma:
    def __init__(self, gas_feed, V, mdot_in, T0, P0, EN_func=None, debug=False, gas0=None, mdot_air=None, mdot_fuel=None, L_fuel_inlet=0.0, fuelStr=None, oxidizerStr=None):
        """
        Constant-P PSR. 'gas_feed' is the fresh inlet mixture at (T0,P0).
        'gas0' (optional) is the reactor's initial state (e.g., HP pre-eq).
        """
        import cantera as ct
        self._eedf_dumped = False

        self.V = V
        self.mdot_in = mdot_in
        self.T0 = T0
        self.P0 = P0
        self.EN_func = EN_func
        self.debug = debug

        # --- Inlet (fresh) ---
        gas_feed.TP = T0, P0   # keep feed at supply conditions   #1000 or T0
        self.inlet = ct.Reservoir(gas_feed)

        # --- Reactor (constant P) ---
        self.rgas = gas0 if gas0 is not None else gas_feed  # if gas0 provided, do NOT reset its TP
        self.reactor = ct.IdealGasConstPressureReactor(self.rgas, energy='on', volume=V)

        # --- Outlet reference reservoir at (T0,P0) with same mechanism ---
        mech = getattr(gas_feed, "source", None) or getattr(gas_feed, "name", None)
        env = ct.Solution(mech, transport_model='None')
        env.TPX = T0, P0, gas_feed.X
        self.outlet = ct.Reservoir(env)

        # --- Inlet(s) ---
        if (mdot_air is not None) and (mdot_fuel is not None) and (fuelStr is not None) and (oxidizerStr is not None):
            gas_air = ct.Solution(mech, transport_model='None');  gas_air.TPX = T0, P0, oxidizerStr
            gas_f   = ct.Solution(mech, transport_model='None');  gas_f.TPX   = T0, P0, {fuelStr: 1.0}
            if L_fuel_inlet > 0.0:
                # lower the fuel stream enthalpy by latent (sets a slightly cooler T)
                gas_f.HP = gas_f.enthalpy_mass - L_fuel_inlet, P0

            self.inlet_air  = ct.Reservoir(gas_air)
            self.inlet_fuel = ct.Reservoir(gas_f)
            self.mfc_air  = ct.MassFlowController(self.inlet_air,  self.reactor, mdot=mdot_air)
            self.mfc_fuel = ct.MassFlowController(self.inlet_fuel, self.reactor, mdot=mdot_fuel)
            self.mfco     = ct.MassFlowController(self.reactor,    self.outlet,  mdot=(mdot_air + mdot_fuel))
        else:
            # fall back: single premixed inlet, as you had
            gas_feed.TP = T0, P0
            self.inlet = ct.Reservoir(gas_feed)
            self.mfc   = ct.MassFlowController(self.inlet,  self.reactor, mdot=mdot_in)
            self.mfco  = ct.MassFlowController(self.reactor, self.outlet, mdot=mdot_in)

        self.sim = ct.ReactorNet([self.reactor])

        # Gentle start; relax later once past the first microseconds
        """ if hasattr(self.sim, "set_initial_time_step"):
            self.sim.set_initial_time_step(1e-10)   # older Cantera
        elif hasattr(self.sim, "initial_time_step"):
            self.sim.initial_time_step = 1e-10      # newer Cantera
        # Optional: also cap max step early
        if hasattr(self.sim, "set_max_time_step"):
            self.sim.set_max_time_step(1e-8)
        elif hasattr(self.sim, "max_time_step"):
            self.sim.max_time_step = 1e-8 """

    def run(self, t_end, dt=1e-5, dt_EN=1e-5):
        """ import cantera as ct
        t = 0.0
        states = ct.SolutionArray(self.reactor.thermo, extra=['t'])
        while t < t_end:
            # if self.EN_func: (keep your EN/EEDF updates here if needed)
            t = self.sim.step()
            states.append(self.reactor.thermo.state, t=t) """

        """ gas = self.rgas
        print("\n=== y / ydot index map (reactor) ===")
        print("0 : mass")
        print("1 : temperature")
        for k, s in enumerate(gas.species_names, start=2):
            print(f"{k} : Y[{s}]") """

        states = ct.SolutionArray(self.reactor.thermo, extra=['t'])

        eps = 1e-12  #

        # Pulse timing / EN profile
        EN_peak = 100 * 1e-21  # Td
        c0        = 24e-9
        period    = 50e-4
        sigma     = 3e-9
        pre_win   = 30e-9
        post_win  = 70e-9

        def gaussian_about_center(t, center):
            return EN_peak * np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))

        def next_pulse_center(t):
            if t <= c0:
                return c0
            n = np.ceil((t - c0) / period)
            return c0 + n * period

        # time step sizes
        dt_fine        = 1e-10
        dt_chunk_fine  = 1e-9
        dt_coarse      = 1e-5

        # Nominal multipliers @ moderate drive:
        mults = [9.0, 6.5, 5.5, 7.0, 5.0, 5.5, 7.5]

        S = (EN_peak/(100 * 1e-21))**0.5
        #print(S)
        S = max(0.5, min(2.0, S))  # clip
        #print(S)

        for i in range(7):
            M_final = 1.0 + (mults[i] - 1.0) * S  # or just mults[i] if no scaling
            self.rgas.set_multiplier(M_final, i)
            #print(f"Applied M={M_final:.3g}")

        print("electron temp", self.rgas.Te)
        EN_now = gaussian_about_center(0, c0)
        self.rgas.EN = EN_now
        print("EN ", self.rgas.EN)
        self.rgas.update_EEDF()
        print("electron temp 2", self.rgas.Te)

        # MAIN LOOP: always reference self.sim.time
        while self.sim.time < t_end - eps:
            t_now = self.sim.time
            c_next    = next_pulse_center(t_now)
            fine_start = c_next - pre_win
            fine_end   = c_next + post_win
            print(t_now)

            if t_now < fine_start - eps:
                t_chunk = min(t_end, fine_start)
                dt_use  = dt_coarse

                # EN ~ 0 here
                self.rgas.EN = 0.0
                self.rgas.update_EEDF()
                self.sim.reinitialize()

                while self.sim.time < t_chunk - eps:
                    target = min(self.sim.time + dt_use, t_chunk)
                    # guard against zero-length step
                    if target - self.sim.time <= eps:
                        # nudge or let CVODE choose a safe internal step
                        target = np.nextafter(self.sim.time, np.inf)
                        self.sim.step()
                        break
                    #self.sim.advance(target)

                    try:
                        if not hasattr(self, "_vt_idx"):
                            self._vt_idx = _find_vt_reactions(self.reactor.thermo)
                        _apply_lowT_vt_clamp(self.reactor.thermo, self._vt_idx, Tcut=200.0, delta=20.0)
                        #_hard_lowT_vt_guard(self.reactor.thermo, self._vt_idx, Tcut=200.0, hyst=20.0, verbose=False)
                        self.sim.advance(target)
                        #print("EN ", self.rgas.EN)

                        q_chem, q_wall, dTdt_est, rho_cp = _energy_budget(self.rgas, self.reactor)
                        if not np.isfinite(q_chem):
                            raise RuntimeError("[energy] Non-finite q_chem detected; aborting step.")

                        """ ok = _energy_watch(self.rgas, self.reactor, tag="step")
                        if not ok:
                            # clean stop so you can read the prints and inspect the state
                            raise RuntimeError("Aborting integration due to invalid kinetics or temp floor.") """

                        """ try:
                            qchem, qe, _, _, emask = _qchem_qe(self.rgas)
                            # "ROP_e" like yours: sum of |ROP| over e-impact reactions
                            ROP_e = float(np.sum(np.abs(self.rgas.net_rates_of_progress[emask])))

                            # Detect avalanche start: fast ROP_e growth or very large |q_e|
                            if (ROP_e > 1e5 and ROP_e > 2.0 * getattr(self, "_prev_ROP_e", 0.0)) or (abs(qe) > 1e12):
                                _dump_electron_breakdown(self.rgas, top=20, tag="avalanche")
                                # Optional: snapshot the state
                                #_save_state(self.rgas, f"avalanche_state_{self.sim.time: .3e}s.yaml")
                            self._prev_ROP_e = ROP_e
                        except Exception as ex:
                            print("[e-probe:error]", ex)

                        if (not self._eedf_dumped) and (ROP_e > 2.0e4) and (ROP_e < 7.0e4):
                            _print_eedf_snapshot(self, self.rgas, tag="pre-avalanche")
                            self._eedf_dumped = True """

                    except Exception as e:
                        """ print("\n=== CVODE failure caught; dumping diagnostics ===")
                        gas = self.reactor.thermo  # the exact Solution inside this reactor
                        print_species_index_map(gas)            # once per run is enough
                        diagnose_nonfinite_reactions(
                            gas,
                            focus_species=(1, 18, 23, 220),
                            auto_disable=False,                 # set True to auto-disable the culprits
                        ) """
                        gas = self.reactor.thermo  # the exact Solution inside THIS reactor
                        dump_pre_failure_diagnostics(gas)
                        raise  # or return early if you want to keep going after auto-disable

                    states.append(self.reactor.thermo.state, t=self.sim.time)

                # proceed to next iteration (re-evaluate fine window)
                continue

            # -------- Fine window around pulse (resolve Gaussian) --------
            t_chunk = min(t_end, fine_end)
            dt_use  = dt_fine

            while self.sim.time < t_chunk - eps:
                # size this small chunk relative to CURRENT time
                local_chunk_end = min(self.sim.time + dt_chunk_fine, t_chunk)

                # march through this small chunk with small dt
                while self.sim.time < local_chunk_end - eps:
                    target = min(self.sim.time + dt_use, local_chunk_end)
                    if target - self.sim.time <= eps:
                        # either nudge to next representable float, or take a step
                        target = np.nextafter(self.sim.time, np.inf)
                        self.sim.step()
                        break
                    #self.sim.advance(target)

                    try:
                        if not hasattr(self, "_vt_idx"):
                            self._vt_idx = _find_vt_reactions(self.reactor.thermo)
                        _apply_lowT_vt_clamp(self.reactor.thermo, self._vt_idx, Tcut=200.0, delta=20.0)
                        #_hard_lowT_vt_guard(self.reactor.thermo, self._vt_idx, Tcut=200.0, hyst=20.0, verbose=False)

                        q_chem, q_wall, dTdt_est, rho_cp = _energy_budget(self.rgas, self.reactor)
                        if not np.isfinite(q_chem):
                            raise RuntimeError("[energy] Non-finite q_chem detected; aborting step.")

                        self.sim.advance(target)
                        #print("EN ", self.rgas.EN)

                        """ ok = _energy_watch(self.rgas, self.reactor, tag="step")
                        if not ok:
                            # clean stop so you can read the prints and inspect the state
                            raise RuntimeError("Aborting integration due to invalid kinetics or temp floor.") """

                        """ try:
                            qchem, qe, _, _, emask = _qchem_qe(self.rgas)
                            # "ROP_e" like yours: sum of |ROP| over e-impact reactions
                            ROP_e = float(np.sum(np.abs(self.rgas.net_rates_of_progress[emask])))

                            # Detect avalanche start: fast ROP_e growth or very large |q_e|
                            if (ROP_e > 1e5 and ROP_e > 2.0 * getattr(self, "_prev_ROP_e", 0.0)) or (abs(qe) > 1e12):
                                _dump_electron_breakdown(self.rgas, top=20, tag="avalanche")
                                # Optional: snapshot the state
                                #_save_state(self.rgas, f"avalanche_state_{self.sim.time: .3e}s.yaml")
                            self._prev_ROP_e = ROP_e
                        except Exception as ex:
                            print("[e-probe:error]", ex)

                        if (not self._eedf_dumped) and (ROP_e > 2.0e4) and (ROP_e < 7.0e4):
                            _print_eedf_snapshot(self, self.rgas, tag="pre-avalanche")
                            self._eedf_dumped = True """


                    except Exception as e:
                        """ print("\n=== CVODE failure caught; dumping diagnostics ===")
                        gas = self.reactor.thermo  # the exact Solution inside this reactor
                        print_species_index_map(gas)            # once per run is enough
                        diagnose_nonfinite_reactions(
                            gas,
                            focus_species=(1, 18, 23, 220),
                            auto_disable=False,                 # set True to auto-disable the culprits
                        ) """
                        gas = self.reactor.thermo  # the exact Solution inside THIS reactor
                        dump_pre_failure_diagnostics(gas)
                        raise  # or return early if you want to keep going after auto-disable

                    states.append(self.reactor.thermo.state, t=self.sim.time)

                # update EN/EEDF at the end of each small chunk using current time
                EN_now = gaussian_about_center(self.sim.time, c_next)
                self.rgas.EN = EN_now
                self.rgas.update_EEDF()
                self.sim.reinitialize()

        return states

""" class PFR(object):

    def __init__(self,gas):
        self.gas = gas
        self.MW  = gas.molecular_weights

    def __call__(self, t, y, ODEParams):

        # Secondary zone parameters:
            # SZ_A   :  Secondary zone area [m2]
            # SZ_mf  :  Total SZ_mf at the point of integration [kg/s]
            # beta   :  Mass flow of air per m [kg/s/m]
            # Y_in_a :  Mass fractions air [-]
            # h_a_in :  Specific enthalpy air [J/kg]
        #print('t,y_values = ',t,y[0], y[1])
        SZ_A   = ODEParams.SZ_A

        Y_in_a = ODEParams.Y_in_a
        h_in   = ODEParams.h_in
        P0     = ODEParams.P0
        x_dil_start = ODEParams.dil_start
        x_dil_end   = ODEParams.dil_end

        x_DZ_dil_start = ODEParams.DZ_dil_start

        ### Definitions

        self.gas.set_unnormalized_mass_fractions(y[2:])  # Set the mass fractions without normalizing to force sum(Y) == 1.0
        self.gas.TP = y[1], P0

        rho = self.gas.density                           # Gas density [kg/m^3]
        wdot = self.gas.net_production_rates             # Production rates of the species [kmol/m^3/s]
        M = y[0]                                         # Mass flow rate in reactor [kg/s]

        ### Equations of the secondary zone (Ideal Gas Plug Flow Reactor)

        # Conservation of mass
        # --------------------

        beta = 0
        if (t >= x_dil_start) & (t < x_dil_end):
            beta += ODEParams.beta               # Massflow of air per m [kg/s/m]
        if (t >= x_DZ_dil_start):
            beta += ODEParams.beta_DZ            # Massflow of air per m [kg/s/m]

        dMdz = beta

        # Conservation of species
        # -----------------------
        dYdz = (wdot*self.MW*SZ_A + beta*(Y_in_a-self.gas.Y))/M

        # Conservation of energy
        # ----------------------
        dTdz = 1/(self.gas.cp_mass*M)*(beta*(h_in - np.dot(Y_in_a, self.gas.partial_molar_enthalpies/self.MW)) \
        -SZ_A*np.dot(self.gas.partial_molar_enthalpies, wdot))

        ### Outputs
        return np.hstack((dMdz,dTdz,dYdz)) """

class PFR(object):
    def __init__(self, gas):
        self.gas = gas
        self.MW  = gas.molecular_weights

    def __call__(self, t, y, ODEParams):
        import numpy as np

        SZ_A   = ODEParams.SZ_A
        Y_in_a = ODEParams.Y_in_a
        h_in   = ODEParams.h_in
        P0     = ODEParams.P0
        x_dil_start   = ODEParams.dil_start
        x_dil_end     = ODEParams.dil_end
        x_DZ_dil_start = ODEParams.DZ_dil_start

        # ---- STRICT VALIDATION ONLY (no edits to physics) ----
        ns = int(self.gas.n_species)
        if y is None or len(y) != 2 + ns:
            raise FloatingPointError(f"bad state length: {None if y is None else len(y)} (need {2+ns})")

        T = float(y[1])
        Y = np.asarray(y[2:], dtype=float)

        if not np.isfinite(T):
            raise FloatingPointError("non-finite temperature from integrator")
        if not np.all(np.isfinite(Y)):
            raise FloatingPointError("non-finite mass fractions from integrator")
        # sum(Y) doesn’t have to be exactly 1 for set_unnormalized_*; just forbid NaN/Inf:
        # (we do NOT normalize or clip here to avoid perturbing your solution!)

        # ---- Your original two lines to set state ----
        self.gas.set_unnormalized_mass_fractions(Y)
        self.gas.TP = T, P0

        # ---- Your original math from here on ----
        rho  = self.gas.density
        wdot = self.gas.net_production_rates   # [kmol/m^3/s]
        M    = y[0]                             # [kg/s]

        beta = 0.0
        if (t >= x_dil_start) and (t < x_dil_end):
            beta += ODEParams.beta
        if (t >= x_DZ_dil_start):
            beta += ODEParams.beta_DZ

        dMdz = beta
        dYdz = (wdot*self.MW*SZ_A + beta*(Y_in_a - self.gas.Y)) / M
        dTdz = (1.0 / (self.gas.cp_mass * M)) * (
            beta * (h_in - np.dot(Y_in_a, self.gas.partial_molar_enthalpies / self.MW))
            - SZ_A * np.dot(self.gas.partial_molar_enthalpies, wdot)
        )
        return np.hstack((dMdz, dTdz, dYdz))