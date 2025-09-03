import numpy as np
import cantera as ct
import CombUtils
from hy2reactor import run_psr_constP
from hybridReactors import PFR
#from Reactors import PFR
from types import SimpleNamespace
from scipy.integrate import ode
import time
from functools import reduce
from itertools import count
import sys
import matplotlib.pyplot as plt

class PlasmaParameters:
    """
    The PlasmaParameters class stores relevant input parameters to the combustor model,
    structured identically to the original architecture for compatibility.
    """

    debug = False
    # Chemical Properties
    RM = 'gri30_plasma_cpavan.yaml'             # Reaction mechanism [-]
    fuelStr = 'CH4:1'            # Fuel string [-]
    Oxidizer = 'O2:1.0, N2:3.76' # Oxidizer string [-]
    FA_st = 0.0581465            # Stoichiometric fuel-air ratio [-]
    LHV = 50748183.2665071       # Lower heating value [J/kg]

    # Combustor geometric properties
    PZ_volume = 0.005            # Volume of PZ [m^3]
    SZ_volume = 0.00025          # Volume of SZ [m^3]
    SZ_length = 0.20             # Secondary zone length [m]
    frac_dil_length = 0.1        # Ratio of dilution zone length to SZ length [-]

    # Simulation settings
    PZ_tsim = 100e-3 / 4         # Simulation time for PZ reactor [s]
    SZ_dz = 1e-3                 # Step size in SZ [m]
    SZ_dilution = 1             # Whether to model SZ dilution air [1:yes, 0:no]
    PZ_n_reactor = 1             # Number of PZ reactors
    PZ_k_pressure = 1           # Pressure controller constant in PSR

    def __init__(self):
        self.mdot_air  = 19      # Total air mass flow rate [kg/s]
        self.mdot_fuel = 0.40    # Total fuel mass flow rate [kg/s]

    def calcGeom(self):
        """Calculate SZ cross-sectional area and total combustor volume."""
        self.V_combustor = self.PZ_volume + self.SZ_volume
        self.SZ_A = self.SZ_volume / self.SZ_length

    def des_splitMassFlow(self, PZ_desPhi, SZ_desPhi):
        """Split air mass flow based on PZ and SZ design equivalence ratios."""
        self.PZ_desPhi = PZ_desPhi
        self.PZ_phi_mean = PZ_desPhi
        self.PZ_phi_sigma = 2.16 * np.exp(-2 * self.PZ_phi_mean)

        self.PZ_mdot_air = self.mdot_fuel / (PZ_desPhi * self.FA_st)
        self.PZ_airfrac = self.PZ_mdot_air / self.mdot_air
        self.PZ_mdot_in = self.PZ_mdot_air + self.mdot_fuel

        self.SZ_desPhi = SZ_desPhi
        self.SZ_airfrac = (self.mdot_fuel / self.mdot_air) / (SZ_desPhi * self.FA_st)
        self.SZ_mdot_air = self.mdot_air * self.SZ_airfrac

        self.DZ_airfrac = 1 - (self.SZ_airfrac + self.PZ_airfrac)
        self.DZ_mdot_air = self.mdot_air * self.DZ_airfrac

        if self.DZ_mdot_air < 0:
            print("WARNING NO DILUTION AIR: DZ_mdot_air =", self.DZ_mdot_air)
            self.SZ_mdot_air = self.mdot_air - self.PZ_mdot_air
            self.SZ_airfrac = self.SZ_mdot_air / self.mdot_air
            self.DZ_mdot_air = 0
            self.DZ_airfrac = 0
            print(f"\tSZ mdot air set to {self.SZ_mdot_air:.2f},\n"
                  f"\tSZ_airfrac = {self.SZ_airfrac:.5f}\n\t..and setting 0.0 DZ air")

        print(f"Design point values:\n\tDes. PZ_airfrac = {self.PZ_airfrac}\n"
              f"\tDes. SZ_airfrac = {self.SZ_airfrac}\n\tDes. DZ_airfrac = {self.DZ_airfrac}")

    def splitMassFlow(self):
        """Split total air mass flow using precomputed fractions."""
        self.PZ_mdot_air = self.mdot_air * self.PZ_airfrac
        self.PZ_phi_mean = (self.mdot_fuel / self.PZ_mdot_air) / self.FA_st
        self.PZ_phi_sigma = 2.16 * np.exp(-2 * self.PZ_phi_mean)
        self.PZ_mdot_in = self.PZ_mdot_air + self.mdot_fuel

        self.SZ_mdot_air = self.mdot_air * self.SZ_airfrac
        self.DZ_mdot_air = self.mdot_air * self.DZ_airfrac

def _ensure_comp_string(token_or_str: str) -> str:
    """Return a valid composition string 'Name:1' if a bare species name is given."""
    s = token_or_str.strip()
    return s if (':' in s or ',' in s) else f"{s}:1"

def _resolve_species_name_case_insensitive(gas: ct.Solution, token: str) -> str | None:
    """Return the mechanism's canonical species name matching token (case-insensitive), else None."""
    base = token.split(':', 1)[0].strip()
    for s in gas.species_names:
        if s.lower() == base.lower():
            return s
    return None

def _canon_comp_for_mech(comp_str: str, gas: ct.Solution) -> str:
    """
    Parse 'A:1, B:2' and rewrite with the mechanism's canonical species names (case-insensitive).
    Keep only species present in 'gas'. Renormalize mole fractions. Raise if nothing remains.
    """
    comp_str = comp_str.replace(' ', '')
    if not comp_str:
        raise ValueError("Empty composition string")
    kept = []
    for part in comp_str.split(','):
        if not part:
            continue
        if ':' in part:
            name, val = part.split(':', 1)
            try:
                v = float(val)
            except Exception:
                v = 1.0
        else:
            name, v = part, 1.0
        mech_name = _resolve_species_name_case_insensitive(gas, name)
        if mech_name is not None:
            kept.append((mech_name, float(v)))
    if not kept:
        raise ValueError(f"No species in '{comp_str}' exist in mechanism '{gas.name or gas.source}'")
    s = sum(v for _, v in kept)
    return ','.join(f"{n}:{v/s:g}" for n, v in kept)

def neutral_HP_to_plasma_IC(phi, fuel_str, oxid_str, h_after, T0, P0, mech_plasma, mech_neutral='nDodecane_ReitzNO.yaml'):
    g_feed = ct.Solution(mech_plasma, transport_model='None')
    g_feed.TP = T0, P0

    # USE the new, robust mapping (case-insensitive, guarded)
    fuel_plasma = _canon_comp_for_mech(_ensure_comp_string(fuel_str), g_feed)
    oxid_plasma = _canon_comp_for_mech(oxid_str, g_feed)          # <-- this was returning '' before

    g_feed.set_equivalence_ratio(phi, fuel_plasma, oxid_plasma)
    g_feed.HP = h_after, P0

    gN = ct.Solution(mech_neutral, transport_model='None')
    gN.TP = T0, P0
    fuel_neutral = _canon_comp_for_mech(_ensure_comp_string(fuel_str), gN)
    oxid_neutral = _canon_comp_for_mech(oxid_str, gN)
    gN.set_equivalence_ratio(phi, fuel_neutral, oxid_neutral)
    gN.HP = h_after, P0
    try:
        gN.equilibrate('HP')
    except ct.CanteraError:
        gN.equilibrate('TP'); gN.HP = h_after, P0

    g0 = ct.Solution(mech_plasma, transport_model='None')
    X_map = {sp: gN.X[gN.species_index(sp)] for sp in gN.species_names if sp in g0.species_names}
    s = sum(X_map.values()) or 1.0
    for k in list(X_map): X_map[k] /= s
    g0.TPX = gN.T, P0, X_map
    g0.HP  = h_after, P0
    return g_feed, g0

def debug_pz_tau_denominator(Params, show=15):
    print("\n[τ-debug]  i   phi       V_i [m^3]      mdot_air [kg/s]   mdot_fuel [kg/s]   V/mdot_air [s]   V/(air+fuel) [s]")
    for i, (phi, V, md_in, md_fuel) in enumerate(zip(
            Params.PZ_phi_lst,
            Params.PZ_V_lst,
            Params.PZ_mdot_in_lst,
            getattr(Params, "PZ_mdot_fuel_lst", [0.0]*len(Params.PZ_V_lst))
        ), start=1):
        # Heuristic: some codebases store air-only in mdot_in; others store total.
        # We can compute both candidate taus:
        md_air = md_in - md_fuel if md_in >= md_fuel > 0 else md_in
        tau_air    = V / md_air
        tau_total  = V / (md_air + md_fuel)
        print(f"[τ-debug] {i:02d}  {phi:7.3f}  {V:12.6e}   {md_air:14.6e}   {md_fuel:14.6e}   {tau_air:12.6e}   {tau_total:12.6e}")
        if i >= show:
            break

def print_pz_diagnostics(Params, top_n=20, min_x=0.0):
    """
    Print PZ exit mixture details + per-reactor temps/residence times + species table.
    top_n: show top N species by mole fraction (set None to print all; use min_x to filter)
    """
    import numpy as np

    def to_scalar(v):
        """Return a float; if v is an array/Sequence, use the last entry."""
        try:
            a = np.asarray(v)
            if a.ndim == 0:
                return float(a)
            return float(a.ravel()[-1])
        except Exception:
            try:
                return float(v)
            except Exception:
                return float("nan")

    print("\n================ PZ DIAGNOSTICS ================")

    # ---- Per-reactor state & residence times ----
    n = len(getattr(Params, "PZ_states", []))
    print(f"Parallel PZ reactors: {n}")
    if n:
        print("Per-reactor state (last in time):")
        for i, (state, tres, phi_i) in enumerate(
            zip(Params.PZ_states, Params.PZ_tres_lst, Params.PZ_phi_lst), start=1
        ):
            if hasattr(Params, "_pz_TP") and len(Params._pz_TP) >= i:
                T_i, P_i = Params._pz_TP[i-1]
            else:
                T_i = to_scalar(getattr(state, "T", np.nan))
                P_i = to_scalar(getattr(state, "P", np.nan)) / 1e5  # bar
            print(f"  PZ{i:02d}: T = {T_i:9.2f} K | P = {P_i:7.3f} bar | ϕ = {phi_i:6.3f} | t_res = {tres:8.5f} s")

    # Overall residence time from geometry/flow (same metric across impls)
    try:
        t_res_overall = Params.PZ_volume / Params.PZ_mdot_in
        print(f"\nOverall (geom/flow) residence time ≈ {t_res_overall:.5f} s")
    except Exception:
        pass

    # ---- Mass-averaged PZ exit mixture ----
    gas = Params.PZ_exitStream.phase  # from CombustorPZ reduce(...) of stream quantities
    try:
        phi_mix = gas.get_equivalence_ratio(oxidizers=["O2"])
    except Exception:
        phi_mix = float("nan")

    print("\nMass-averaged PZ exit mixture:")
    print(f"  T = {gas.T:.2f} K,  P = {gas.P/1e5:.3f} bar,  ϕ ≈ {phi_mix:.3f}")
    print(f"  mdot_out = {Params.PZ_exitStream.mass:.6f} kg/s")

    # ---- Species table: mole frac, mass frac, concentration ----
    species = gas.species_names
    X = gas.X
    Y = gas.Y
    C = gas.concentrations  # kmol/m^3

    order = np.argsort(-X)
    names_sorted = [species[i] for i in order]
    X_sorted = X[order]
    Y_sorted = Y[order]
    C_sorted = C[order]

    if top_n is not None:
        names_sorted = names_sorted[:top_n]
        X_sorted = X_sorted[:top_n]
        Y_sorted = Y_sorted[:top_n]
        C_sorted = C_sorted[:top_n]
    else:
        if min_x > 0:
            keep = X_sorted >= min_x
            names_sorted = [n for n, k in zip(names_sorted, keep) if k]
            X_sorted = X_sorted[keep]
            Y_sorted = Y_sorted[keep]
            C_sorted = C_sorted[keep]

    print("\nTop species at PZ exit (sorted by mole fraction):")
    print(f"{'Species':>12s}  {'X [-]':>12s}  {'Y [-]':>12s}  {'C [kmol/m^3]':>16s}")
    for n, xi, yi, ci in zip(names_sorted, X_sorted, Y_sorted, C_sorted):
        print(f"{n:>12s}  {xi:12.6e}  {yi:12.6e}  {ci:16.6e}")

    # Always show key tracers if present
    key = ["NO", "NO2", "CO", "CO2", "O2", "H2O", "OH", "O", "N2"]
    print("\nSelected species at PZ exit:")
    for sp in key:
        if sp in species:
            k = species.index(sp)
            print(f"  {sp:>4s}:  X={X[k]:.6e},  Y={Y[k]:.6e},  C={C[k]:.6e} kmol/m^3")

    print("================================================\n")

def _resolve_species_name(gas: 'ct.Solution', token: str) -> str:
    """
    Return the canonical species name from the mechanism that matches `token`
    case-insensitively. Accepts 'NC12H26', 'nc12h26', or 'NC12H26:1' etc.
    Raises if no match is found.
    """
    base = token.split(':', 1)[0].strip()
    names = gas.species_names
    for s in names:
        if s.lower() == base.lower():
            return s
    raise ValueError(f"Fuel '{token}' not found in mechanism species list.")

def _pure_species_X(spec_name: str) -> str:
    """Return a safe composition string 'Spec:1' for Cantera."""
    return f"{spec_name}:1"

def CombustorPZ(params, gas):
    """
    Run Cantera-based PSRs in parallel for the Primary Zone (PZ).
    Replaces ODE integration with native Cantera reactor tools.
    """
    PZ_start = time.time()

    # Unpack frequently used parameters
    RM            = params.RM
    fuelStr       = params.fuelStr
    oxidizerStr   = params.Oxidizer
    T0, P0        = params.T0, params.P0
    tsim          = params.PZ_tsim

    # Create the per-reactor split like the original does
    splitReactor(params)  # populates: PZ_phi_lst, PZ_V_lst, PZ_mdot_in_lst, PZ_mdot_fuel_lst
    phi_lst       = params.PZ_phi_lst
    V_lst         = params.PZ_V_lst
    mdot_in_lst   = params.PZ_mdot_in_lst
    mdot_fuel_lst = params.PZ_mdot_fuel_lst

    # Bookkeeping arrays like before
    Params.PZ_states   = []
    params.PZ_tres_lst = []
    params.PZ_streams  = []
    params.PZ_ICs_original = []

    # Helper to build a fresh Solution
    def new_gas():
        g = ct.Solution(RM, transport_model='None')
        return g

    # Loop over each parallel PSR in the PZ
    def run_one(i_phi_V_mdot):
        i, phi_i, V_i, mdot_in_i, mdot_fuel_i = i_phi_V_mdot

        # ---- STEP A: premix at (T0,P0) and compute enthalpy with latent penalty ----
        g_feed = new_gas()
        g_feed.TPX = T0, P0, oxidizerStr                  # start with oxidizer
        h_air_T0P0 = g_feed.enthalpy_mass

        g_fuel = new_gas()
        # fuelStr is already like "CH4:1" or "NC12H26:1"
        fuel_spec = _resolve_species_name(g_fuel, fuelStr)     # e.g., 'NC12H26'
        g_fuel.TPX = T0, P0, {fuel_spec: 1.0}                  # safe pure-species set
        h_fuel_T0P0 = g_fuel.enthalpy_mass

        # Use your stored fuel reference enthalpy at (Tref,Pref) built in setChem()
        # params.fuel_std_enthalpy_mass is the standard reference you saved
        # Latent (as in your original; keep same value / basis)
        L_fuel = 61.1e3/0.17033  # [J/kg] for n-dodecane example; retains original behavior

        # Build the *feed* composition (reactants) at the target φᵢ
        fuel_comp = _pure_species_X(fuel_spec)                 # 'NC12H26:1'
        g_feed.set_equivalence_ratio(float(phi_i), fuel_comp, oxidizerStr)
        h_mix_T0P0 = g_feed.enthalpy_mass

        # Your original "after vaporization" enthalpy for the *incoming* stream:
        # h_after = (m_in * h_mix - m_fuel * (L + (h_fuel_T0P0 - h_fuel_std)))/m_in
        h_after = (mdot_in_i*h_mix_T0P0
                   - mdot_fuel_i*(L_fuel + (h_fuel_T0P0 - params.fuel_std_enthalpy_mass))) / mdot_in_i

        # Impose the inlet (h,P) exactly on the feed
        mech_plasma  = Params.RM          # your plasma mechanism file
        mech_neutral = getattr(Params, 'NM', 'nDodecane_ReitzNO.yaml')  # neutral mech (configure if not GRI)
        # Respect inlet supply state (T0,P0) for the FEED gas
        g_feed, g0 = neutral_HP_to_plasma_IC(
            phi=float(phi_i),
            fuel_str=Params.fuelStr,
            oxid_str=Params.Oxidizer,
            h_after=h_after,
            T0=T0,                 # <-- add this
            P0=P0,
            mech_plasma=Params.RM,
            mech_neutral=getattr(Params, 'NM', 'nDodecane_ReitzNO.yaml')
        )
        # enforce supply state on the feed (fresh)


        # Log the "original" IC after HP pre-eq for easy side-by-side prints
        try:
            phi_i = float(Params.PZ_phi_lst[i])
        except Exception:
            phi_i = float("nan")
        Params.PZ_ICs_original.append({
            "i":        int(i)+1,
            "phi":      phi_i,
            "V":        float(V_i),
            "mdot_in":  float(mdot_in_i),
            "T":        float(g0.T),
            "P":        float(g0.P),
            "h":        float(g0.enthalpy_mass),
        })
        print("\n=== PZ ICs (original, after HP pre-eq) ===")
        print(f"{'i':>2s} {'phi':>7s} {'V[m^3]':>10s} {'mdot[kg/s]':>12s} {'T[K]':>10s} {'P[bar]':>9s} {'h[ MJ/kg ]':>12s}")
        for r in Params.PZ_ICs_original:
            print(f"{r['i']:2d} {r['phi']:7.3f} {r['V']:10.6e} {r['mdot_in']:12.6e} "
                f"{r['T']:10.2f} {r['P']/1e5:9.3f} {r['h']/1e6:12.6f}")


        # ---- STEP C: integrate a constant-P PSR fed by g_feed, IC = g_ic ----
        thermo_out, _ = run_psr_constP(
            gas_feed=g_feed,
            gas_ic=g0,
            V=Params.PZ_V_lst[i],
            mdot_in=Params.PZ_mdot_in_lst[i],
            P0=P0,
            t_end=params.PZ_tsim,
            dt=1e-5,
            debug=False
        )
        Params.PZ_states.append(thermo_out)
        if not hasattr(Params, "_pz_TP"):
            Params._pz_TP = []  # list of (T, P)
        Params._pz_TP.append((float(thermo_out.T), float(thermo_out.P)))
        """ snap = ct.Solution(thermo_out.source or Params.RM, transport_model='None')
        snap.TPX = thermo_out.T, thermo_out.P, thermo_out.X
        Params.PZ_states.append(snap) """

        # Residence time like the original (ρV/ṁ at the end)
        t_res_i = thermo_out.density * V_i / mdot_in_i
        params.PZ_tres_lst.append(float(t_res_i))

        # Package as a stream Quantity, like before (constant='HP')
        q = ct.Quantity(thermo_out, constant='HP')
        q.mass = mdot_in_i
        params.PZ_streams.append(q)

        return q

    stream_list = list(map(run_one, zip(count(), phi_lst, V_lst, mdot_in_lst, mdot_fuel_lst)))
    PZ_exit = reduce(lambda a, b: a + b, stream_list)  # sum of ct.Quantity (HP basis)
    params.PZ_exitStream = PZ_exit
    gas_out = PZ_exit.phase

    # ---- NOx/CO accounting (same as original) ----
    NOx = 1000.0 * (gas_out.Y[gas_out.species_index('NO')] * 46.0/30.0
                    + gas_out.Y[gas_out.species_index('NO2')]) * (PZ_exit.mass)
    CO  = 1000.0 * (gas_out.Y[gas_out.species_index('CO')]) * (PZ_exit.mass)

    PZ_results = [gas_out.T, gas_out.P, gas_out.density, NOx, CO, gas_out.X,
                  gas_out.cp_mass - gas_out.cv_mass]

    print(f"Primary zone simulation (Cantera PSR) complete. Elapsed: {time.time()-PZ_start:.3f} s")
    return PZ_results

### ----------------------------------------------------------------------
### Define the CombustorSZ function
### ----------------------------------------------------------------------
def CombustorSZ(Params, gas, air):
    SZ_start_time = time.time()


    def SZ_subroutine(mdot_in, A, air, mdot_air, frac_dil_length_i, mdot_DZ, SZ_length = Params.SZ_length):
        #Set gas phase to PZ exit
        gas = Params.PZ_exitStream.phase
#         print("___________________________")
#         print("SZ equivalence ratio = ", gas.get_equivalence_ratio(oxidizers = ['O2']))
#         print("___________________________")
        SZ_start_time = time.time()
        ### Unpack Params
        SZ_mdot_in = mdot_in
        SZ_mdot_air = mdot_air
        DZ_mdot_air = mdot_DZ
        SZ_A       = A
        x0_SZ      = Params.dil_start               # Initial x location secondary zone [x]
        dx_SZ      = SZ_length*Params.SZ_dz         # Space step secondary zone integrator [s]
        P0         = Params.P0

        ### Define solver
        #from Reactors import PFR
        if Params.debug: print("SZ in: T = {0}, P = {0}".format(gas.T,gas.P))
        ode_SZ = PFR(gas)

        solver_SZ = ode(ode_SZ)
        solver_SZ.set_integrator('vode',method='bdf',with_jacobian=True, nsteps = 5000, order = 5, max_step = 1e-4)

        # Initial condition
        y0 = np.hstack((SZ_mdot_in,gas.T,gas.Y))
        solver_SZ.set_initial_value(y0,0)

        states_SZ = ct.SolutionArray(gas,1, extra = {'x_SZ':[0.0],'t_SZ':[0.0],'u_SZ':[0.0], 'M_SZ':[SZ_mdot_in]})

        ### Define variables
        t_SZ_lst = [0]                            # Time list of the secondary zone [s]
        x_SZ_lst = [0]                            # Space list of the secondary zone [m]
        u_SZ_lst = [(SZ_mdot_in)/(gas.density*SZ_A)]# Velocity list of the secondary zone [m/s]

        states_SZ.append(gas.state, x_SZ = x_SZ_lst[-1], t_SZ = t_SZ_lst[-1], u_SZ = u_SZ_lst[-1], M_SZ = SZ_mdot_in)

        Y_in_a = air.Y                            # Mass fractions of incoming air [-]
        h_in_a = air.enthalpy_mass                # Specific enthalpy of incoming air [J/kg]

        """ x0   = Params.dil_start * SZ_length
        x1   = x0 + frac_dil_length_i * SZ_length
        xdz0 = SZ_length * (1.0 - Params.DZ_frac_dil_length)
        print("\n=== SZ DEBUG [{}] ===".format("outer" if DZ_mdot_air > 0 else "inner"))
        print(f"[SZ DEBUG] mdot_in (from PZ) : {SZ_mdot_in:.6f} kg/s")
        print(f"[SZ DEBUG] SZ_mdot_air (main): {SZ_mdot_air:.6f} kg/s")
        print(f"[SZ DEBUG] DZ_mdot_air       : {DZ_mdot_air:.6f} kg/s")
        print(f"[SZ DEBUG] expected mdot_out : {SZ_mdot_in + SZ_mdot_air + DZ_mdot_air:.6f} kg/s")
        print(f"[SZ DEBUG] dil_window = [{x0:.6f}, {x1:.6f}] m  (L = {x1-x0:.6f} m)")
        print(f"[SZ DEBUG] DZ_start  = {xdz0:.6f} m  (L = {SZ_length-xdz0:.6f} m)") """

        class ODEParams:
            pass

        ### Set solver parameters
        ODEParams.SZ_A   = SZ_A

        if Params.SZ_dilution == 1:
            ODEParams.beta= SZ_mdot_air/(frac_dil_length_i*SZ_length)
        else:
            ODEParams.beta= 0

        ODEParams.beta_DZ = DZ_mdot_air/(Params.DZ_frac_dil_length*SZ_length)

        ODEParams.dil_start = Params.dil_start*SZ_length
        ODEParams.dil_end = Params.dil_start*SZ_length + frac_dil_length_i*SZ_length

        ODEParams.DZ_dil_start = SZ_length*(1 - Params.DZ_frac_dil_length) #Dilution in the very end portion

        ODEParams.Y_in_a = Y_in_a
        ODEParams.h_in   = h_in_a
        ODEParams.P0     = P0

        if Params.debug:
            print("beta = {0}\nDil St: {1}\nDil end : {2}\nbeta_DZ : {3}".format(ODEParams.beta,
                                                                                 ODEParams.dil_start,
                                                                                 ODEParams.dil_end,
                                                                                 ODEParams.beta_DZ))


        solver_SZ.set_f_params(ODEParams)

        ode_SZ_core = PFR(gas)  # your current RHS

        # ---------------- DEBUG WRAPPER START ----------------
        # Integrate what the RHS is *actually* injecting:
        _x_hist, _b_main_hist, _b_dz_hist = [], [], []
        _chem_mass_src = []  # sum_k MW_k * wdot_k  [kg/m^3/s] (should be ~0)

        # scratch gas for chemistry-only diagnostics (same mechanism)
        _g_dbg = ct.Solution(gas.source or Params.RM, transport_model='None')

        def _ode_wrapped(t, y, ODEParams):
            # reproduce the same gating used inside your RHS
            b_main = ODEParams.beta if (t >= ODEParams.dil_start and t < ODEParams.dil_end) else 0.0
            b_dz   = ODEParams.beta_DZ if (t >= ODEParams.DZ_dil_start) else 0.0
            _x_hist.append(t); _b_main_hist.append(b_main); _b_dz_hist.append(b_dz)

            # optional: check chem mass conservation (should be ≈ 0)
            try:
                _g_dbg.TPY = y[1], ODEParams.P0, y[2:]
                wdot = _g_dbg.net_production_rates  # kmol/m^3/s
                chem_src = float(np.dot(_g_dbg.molecular_weights, wdot))  # kg/m^3/s
            except Exception:
                chem_src = np.nan
            _chem_mass_src.append(chem_src)

            return ode_SZ_core(t, y, ODEParams)
        # ----------------- DEBUG WRAPPER END -----------------

        solver_SZ = ode(_ode_wrapped).set_integrator('vode', method='bdf',
                                                    with_jacobian=True, nsteps=5000,
                                                    order=5, max_step=1e-4)
        y0 = np.hstack((SZ_mdot_in, gas.T, gas.Y))
        solver_SZ.set_initial_value(y0, 0.0).set_f_params(ODEParams)

        ### Run solver (Note that solver_SZ.t refers to the spatial dimension for the PFR (t = distance))
        while solver_SZ.t < SZ_length: #solver_SZ.successful() and solver_SZ.t < SZ_length:
            solver_SZ.integrate(solver_SZ.t + dx_SZ)

            gas.TPY = solver_SZ.y[1],P0,solver_SZ.y[2:]

            if not solver_SZ.successful():
                print('------------------------------------------------------------------------')
                print('--- WARNING: Secondary zone solver does not integrate successfully!!!---')
                print('------------------------------------------------------------------------')
                break
                sys.exit('Integrator Error: SZ does not integrate successfully')

            # Append the states to the solution matrix
            states_SZ.append(gas.state, x_SZ = solver_SZ.t,
                             t_SZ = solver_SZ.t/((solver_SZ.y[0])/(gas.density*SZ_A)),
                             u_SZ = (solver_SZ.y[0])/(gas.density*SZ_A),
                             M_SZ = solver_SZ.y[0])

            #print('Solver x = ', solver_SZ.t)
            #print('States x = ', states_SZ.x_SZ[-1])
            #print('phi =      ', ODEParams.phi)
            #print('mdot =     ', states_SZ.M_SZ[-1])

        ### Write outputs of SZ in specified format [T, p, rho, X, mdot]
        Results_SZ = [float(states_SZ[-1].T),
                      float(states_SZ[-1].P),
                      float(states_SZ[-1].density),
                      states_SZ[-1].X,
                      states_SZ.M_SZ]

        """ # ----- DEBUG SUMMARY: contributions and closure -----
        if len(_x_hist) > 1:
            added_main = float(np.trapz(_b_main_hist, _x_hist))
            added_dz   = float(np.trapz(_b_dz_hist,   _x_hist))
            mdot_out_solver = float(states_SZ.M_SZ[-1])  # or float(solver_SZ.y[0])
            expected_out    = float(SZ_mdot_in + SZ_mdot_air + DZ_mdot_air)

            print(f"[SZ DEBUG] ∫β_main dx (actual) = {added_main:.6f} kg/s  (target {SZ_mdot_air:.6f})")
            print(f"[SZ DEBUG] ∫β_DZ   dx (actual) = {added_dz:.6f} kg/s  (target {DZ_mdot_air:.6f})")
            print(f"[SZ DEBUG] mdot_out(solver)    = {mdot_out_solver:.6f} kg/s")
            print(f"[SZ DEBUG] expected_out        = {expected_out:.6f} kg/s")
            print(f"[SZ DEBUG] mass error (solver - expected) = {mdot_out_solver - expected_out:+.6e} kg/s")

        # chemistry mass source check (should be ~0)
        finite = np.isfinite(_chem_mass_src)
        if np.any(finite):
            print(f"[SZ DEBUG] max|ΣMW·wdot| over path = {np.nanmax(np.abs(np.array(_chem_mass_src)[finite])):.3e} kg/m^3/s") """

        if Params.debug:
            #print(np.shape(states_SZ.Y))
            plt.rcParams['grid.linestyle']='--'
            plt.rcParams['grid.linewidth']= 0.1
            plt.rcParams['grid.alpha']= 0.5

            fig,ax = plt.subplots(4,1,figsize=(10,8), dpi=100,sharex = True)
            ax[0].plot(states_SZ.x_SZ, states_SZ.M_SZ)
            ax[0].set_ylabel('Mdot [kg/s]')
            plt.grid()

            ax[1].plot(states_SZ.x_SZ, states_SZ.T)
            ax[1].set_ylabel('Temp [K]')
            plt.grid()

            ax[2].plot(states_SZ.x_SZ,1000*(states_SZ.Y[:,states_SZ.species_index('NO')]*46/30 +
                             states_SZ.Y[:,states_SZ.species_index('NO2')])*(states_SZ.M_SZ))
            ax[2].set_ylabel('NO$_x$ [g]')
            plt.grid()

            ax[3].plot(states_SZ.x_SZ,1000*states_SZ.Y[:,states_SZ.species_index('CO')]*(states_SZ.M_SZ))
            ax[3].set_ylabel('CO[g]')
            plt.grid()

            plt.xlabel('x distance [m]')
            plt.suptitle('Secondary Zone')
            plt.show()
            for axis in ax:
                axis.grid(linewidth = 0.5, linestyle = '--', alpha = 0.6)

        streamQuantity = ct.Quantity(gas, constant = 'HP')
        streamQuantity.mass = solver_SZ.y[0]

        CO_in = 1000*states_SZ.Y[0,states_SZ.species_index('CO')]*(states_SZ.M_SZ[0])
        CO_out = 1000*states_SZ.Y[-1,states_SZ.species_index('CO')]*(states_SZ.M_SZ[-1])

        NOx_in = 1000*(states_SZ.Y[0,states_SZ.species_index('NO')]*46/30 +
                             states_SZ.Y[0,states_SZ.species_index('NO2')])*(states_SZ.M_SZ[0])
        NOx_out = 1000*(states_SZ.Y[-1,states_SZ.species_index('NO')]*46/30 +
                             states_SZ.Y[-1,states_SZ.species_index('NO2')])*(states_SZ.M_SZ[-1])

        print('--------------------------- CO & NOx ------------------------------------')
        print('CO at start of SZ  = {0:.2E}\tCO  produced  = {1:.2E}\tCO end  = {2:.3f}'.format(CO_in, CO_out - CO_in, CO_out))
        print('NOx at start of SZ = {0:.2E}\tNOx produced  = {1:.2E}\tNOx end = {2:.3f}'.format(NOx_in, NOx_out - NOx_in, NOx_out))
        print('Mass Flow check: {0:.3f} = {1:.3f}'.format(solver_SZ.y[0], states_SZ.M_SZ[-1]))
        print('-------------------------------------------------------------------------')

        return streamQuantity

    SZ_outer = SZ_subroutine(Params.PZ_mdot_in*Params.SZ_mdot_frac_outer,
                         Params.SZ_A*Params.SZ_A_frac_outer,
                         air, Params.SZ_mdot_air/2, Params.frac_dil_length_outer, Params.DZ_mdot_air)

    SZ_inner = SZ_subroutine(Params.PZ_mdot_in*Params.SZ_mdot_frac_inner,
                         Params.SZ_A*Params.SZ_A_frac_inner,
                         air, Params.SZ_mdot_air/2, Params.frac_dil_length_inner, 0)

    """SZ_outer = SZ_subroutine(Params.PZ_mdot_in*Params.SZ_mdot_frac_outer,
                         Params.SZ_A*Params.SZ_A_frac_outer,
                         air, Params.SZ_mdot_air*Params.SZ_mdot_frac_outer, Params.frac_dil_length_outer, Params.DZ_mdot_air)

    SZ_inner = SZ_subroutine(Params.PZ_mdot_in*Params.SZ_mdot_frac_inner,
                         Params.SZ_A*Params.SZ_A_frac_inner,
                         air, Params.SZ_mdot_air*Params.SZ_mdot_frac_inner, Params.frac_dil_length_inner, 0)"""

    SZ_exitStream = SZ_outer + SZ_inner
    print("Total SZ mass exit = {0:.3f}  Expected {1:.3f}".format(SZ_exitStream.mass, Params.mdot_air+Params.mdot_fuel))
    print("Secondary zone simulation complete. Elapsed time: {:.3f} s".format(time.time() - SZ_start_time))

    return SZ_exitStream

Params = PlasmaParameters()

def splitReactor(Params):
    """Splits the primary zone into multiple reactors and calculates corresponding fuel and air mass flow rates."""

    def NormalDistribution(mu, sigma, x):
        return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if Params.PZ_n_reactor == 1:
        Params.PZ_phi_lst        = np.array([Params.PZ_phi_mean])
        Params.PZ_mdot_in_lst    = np.array([Params.PZ_mdot_in])
        Params.PZ_V_lst          = np.array([Params.PZ_volume])
        Params.PZ_mdot_fuel_lst  = Params.PZ_phi_lst * Params.FA_st * Params.PZ_mdot_air

    else:
        mdot_air = Params.PZ_mdot_air

        # Redefine sigma based on phi mean (if desired)
        Params.PZ_phi_sigma = 0.37 * Params.PZ_phi_mean  # or keep preset if already done

        # Set phi bounds
        phi_min = 0.025 / Params.FA_st
        phi_max = 4.0
        x1 = abs(Params.PZ_phi_mean - phi_min) / Params.PZ_phi_sigma
        x2 = abs(phi_max - Params.PZ_phi_mean) / Params.PZ_phi_sigma
        x = np.min([x1, x2, 3])

        # Linearly spaced phi values for each reactor
        Params.PZ_phi_lst = np.linspace(
            Params.PZ_phi_mean - x * Params.PZ_phi_sigma,
            Params.PZ_phi_mean + x * Params.PZ_phi_sigma,
            Params.PZ_n_reactor
        )

        # Use Gaussian distribution to determine air split
        fraction_lst = NormalDistribution(Params.PZ_phi_mean, Params.PZ_phi_sigma, Params.PZ_phi_lst)
        fraction_lst /= np.sum(fraction_lst)

        mdot_air_lst   = mdot_air * fraction_lst
        mdot_fuel_lst  = Params.PZ_phi_lst * Params.FA_st * mdot_air_lst
        Params.PZ_mdot_fuel_lst = mdot_fuel_lst

        if Params.debug:
            print("Fuel flows  =", sum(mdot_fuel_lst), "(input:", Params.mdot_fuel, ")")
            print("Air flows   =", sum(mdot_air_lst), "(input:", mdot_air, ")")
            print("Air split   =", mdot_air_lst)
            print("Fuel split  =", mdot_fuel_lst)
            print("Phi values  =", mdot_fuel_lst / mdot_air_lst / Params.FA_st)

        # Consistency check
        if abs(sum(mdot_fuel_lst) - Params.mdot_fuel) > 1e-10:
            print("Error in mass of fuel = ", sum(mdot_fuel_lst) - Params.mdot_fuel)

        # Equal volume distribution
        Params.PZ_V_lst = (Params.PZ_volume / Params.PZ_n_reactor) * np.ones(Params.PZ_n_reactor)

        # Total gas mass flow per reactor
        Params.PZ_mdot_in_lst = mdot_air_lst + mdot_fuel_lst

    if Params.debug:
        print('Primary zone phi dist    =', Params.PZ_phi_lst)
        print('Primary zone volume dist =', Params.PZ_V_lst)

def setChem(RM, Fuel, Oxidizer):
    """
    Initializes Cantera gas objects for the main gas and dilution air using the specified
    reaction mechanism and fuel/oxidizer strings. Stores results in the Params object.

    Inputs
    ------
    RM       : Reaction mechanism file (e.g. 'gri30.cti')
    Fuel     : Fuel composition string (e.g. 'CH4:1')
    Oxidizer : Oxidizer composition string (e.g. 'O2:1.0, N2:3.76')
    Params   : Instance of PlasmaParameters class
    """

    Params.Tref = 298.15    # Reference temperature [K]
    Params.Pref = 101325.0  # Reference pressure [Pa]

    Params.RM = RM
    Params.fuelStr  = Fuel
    Params.Oxidizer = Oxidizer

    # Compute stoichiometric fuel-air ratio and LHV from utilities
    Params.FA_st = CombUtils.calc_Stoichiometric_FAR(RM, Fuel, Oxidizer)
    Params.LHV   = CombUtils.calcLHV(RM, Fuel, Oxidizer)

    # Create Cantera Solution objects without transport properties
    Params.gas     = ct.Solution(RM, transport_model='None')
    Params.dil_air = ct.Solution(RM, transport_model='None')

    # Set standard enthalpy of the fuel
    Params.gas.TPX = Params.Tref, Params.Pref, Fuel + ":1"
    Params.fuel_std_enthalpy_mass = Params.gas.enthalpy_mass

### ----------------------------------------------------------------------
### Define the setInletFlow function
### ----------------------------------------------------------------------
def setInletFlow(mdot_air, mdot_fuel, Pt3, Tt3):
    """
    Sets inlet mass flow and thermodynamic state for air and fuel streams.

    Inputs
    -------
    mdot_air  : core air mass flow [kg/s]
    mdot_fuel : baseline fuel mass flow [kg/s]
    Pt3       : Total pressure at combustor inlet [Pa]
    Tt3       : Total temperature at combustor inlet [K]
    """
    Params.mdot_air  = mdot_air

    # Scale fuel flow to maintain same heating power as NPSS baseline
    LHV_NPSS = 43.5e6  # [J/kg]
    Params.fuel_scaler = (LHV_NPSS / Params.LHV)
    Params.mdot_fuel = mdot_fuel * Params.fuel_scaler
    Params.fuel_deficit = mdot_fuel * (1 - Params.fuel_scaler)

    Params.P0 = Pt3          # Static pressure for Cantera inlet state [Pa]
    Params.T0 = Tt3          # Static temperature for Cantera inlet state [K]
    Params.T_air = Tt3       # Assume same T for air inlet
    Params.P_air = Pt3       # Assume same P for air inlet

### ----------------------------------------------------------------------
### Define the setGeometry function
### ----------------------------------------------------------------------
def setGeometry(PZ_volume, SZ_volume, SZ_length):
    """
    Sets combustor volume and secondary zone geometry.

    Inputs
    -------
    PZ_volume : Primary zone volume [m^3]
    SZ_volume : Secondary zone volume [m^3]
    SZ_length : Secondary zone length [m]
    """
    Params.PZ_volume = PZ_volume
    Params.SZ_volume = SZ_volume
    Params.SZ_length = SZ_length

def des_setup(PZ_desPhi, PZ_n_reactor,
              SZ_desPhi, SZ_dilution, dil_start,
              frac_dil_length_outer, frac_dil_length_inner,
              SZ_mdot_frac_outer, SZ_A_frac_outer,
              DZ_frac_dil_length = 0.5, #0.05,
              debug = False):
    """
    Sets up the combustor at design point based on equivalence ratios and dilution configuration.

    Inputs
    -------
    PZ_desPhi              : Mean equivalence ratio in PZ
    PZ_n_reactor           : Number of reactors to discretize PZ
    SZ_desPhi              : Secondary zone equivalence ratio
    SZ_dilution            : Enable SZ dilution (1 = yes, 0 = no)
    dil_start              : Start of SZ dilution region (fraction of SZ length)
    frac_dil_length_outer  : Fractional length of outer dilution
    frac_dil_length_inner  : Fractional length of inner dilution
    SZ_mdot_frac_outer     : Fraction of SZ air for outer injection
    SZ_A_frac_outer        : Fraction of SZ area for outer injection
    DZ_frac_dil_length     : Fractional length for final DZ dilution (default 0.05)
    debug                  : Enable debug printouts
    """

    Params.debug = debug
    Params.PZ_n_reactor = PZ_n_reactor
    Params.DZ_frac_dil_length = DZ_frac_dil_length

    # Split air/fuel based on design point Φ
    Params.des_splitMassFlow(PZ_desPhi, SZ_desPhi)
    print(f"[Air Split Check] SZ_airfrac = {Params.SZ_airfrac:.5f}, "
      f"DZ_airfrac = {Params.DZ_airfrac:.5f}, "
      f"PZ_airfrac = {Params.PZ_airfrac:.5f}")

    # Update geometry and dilution config
    Params.calcGeom()
    Params.SZ_dilution = SZ_dilution

    Params.frac_dil_length_outer = frac_dil_length_outer
    Params.frac_dil_length_inner = frac_dil_length_inner

    Params.SZ_mdot_frac_outer = SZ_mdot_frac_outer
    Params.SZ_mdot_frac_inner = 1.0 - SZ_mdot_frac_outer

    Params.SZ_A_frac_outer = SZ_A_frac_outer
    Params.SZ_A_frac_inner = 1.0 - SZ_A_frac_outer

    Params.dil_start = dil_start

def setup(debug=False, PZ_airfrac=None, SZ_airfrac=None, DZ_airfrac=None):
    """
    Allows manual air mass split configuration (off-design) if not using design-point setup.

    Inputs
    -------
    debug        : Enable debug output
    PZ_airfrac   : Fraction of air to PZ (optional)
    SZ_airfrac   : Fraction of air to SZ (optional)
    DZ_airfrac   : Fraction of air to DZ (optional)
    """
    if PZ_airfrac is not None:
        print("\nSetting Airfracs based on provided inputs\n")
        Params.PZ_airfrac = PZ_airfrac
        Params.SZ_airfrac = SZ_airfrac
        Params.DZ_airfrac = DZ_airfrac

    # Check for required design-point variables
    try:
        Params.PZ_airfrac
    except AttributeError:
        raise RuntimeError("MUST SETUP DESIGN POINT BEFORE OFF-DESIGN CALCULATIONS! "
                           "\nUse des_setup(...) to configure first.")

    # Proceed with flow split and geometry update
    Params.debug = debug
    Params.splitMassFlow()
    Params.calcGeom()

def run(printResults):
    if printResults == 1:
        CombUtils.print_run_conditions()

    # Initialize diluent air (secondary zone air)
    Params.dil_air.TPX = Params.T_air, Params.P_air, Params.Oxidizer

    # --- Primary Zone ---
    PZ_results = CombustorPZ(Params, Params.gas)
    print_pz_diagnostics(Params, top_n=25)

    # Optional use of PZ output to update parameters (e.g., thermal diffusivity or flame radius scaling)
    R = PZ_results[-1]  # This could be cp_mass - cv_mass, or used to estimate diffusivity

    # --- Secondary Zone ---
    SZ_exitStream = CombustorSZ(Params, Params.gas, Params.dil_air)

    # --- Emissions Calculation ---
    NOx = 1000 * (
        SZ_exitStream.Y[SZ_exitStream.species_index('NO')] * 46 / 30 +
        SZ_exitStream.Y[SZ_exitStream.species_index('NO2')]
    ) * SZ_exitStream.mass

    CO = 1000 * SZ_exitStream.Y[SZ_exitStream.species_index('CO')] * SZ_exitStream.mass

    # Emission Indices (mass-based emissions per unit fuel mass)
    EI_NOx = NOx / (Params.mdot_fuel / Params.fuel_scaler)
    EI_CO  = CO / (Params.mdot_fuel / Params.fuel_scaler)

    if printResults == 1:
        CombUtils.print_results()

    return Params, PZ_results, SZ_exitStream, EI_NOx, EI_CO, NOx, CO