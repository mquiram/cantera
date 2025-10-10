# test_joule_heating_closed_box.py
#
# Purpose:
#   1) Measure dT/dt in a closed, adiabatic, constant-volume reactor with chemistry OFF.
#   2) Compare to predictions q/(rho*cp) and q/(rho*cv).
#   3) Provide hooks for E-scaling (Joule power ∝ E^2) and a volume sanity check.

import cantera as ct
import numpy as np

# --- USER SETTINGS (edit these) ------------------------------------------------
MECH = "A2NOx_hitest.yaml"     # your mechanism file
PHASE = "plasma"               # the phase name for your PlasmaPhase in MECH
X0 = "N2:0.79, O2:0.21, e:1e-8"   # include 'e' if your model expects it
T0 = 1200.0                    # K (hot enough to get meaningful cp,cv)
P0 = ct.one_atm                # Pa
V0 = 1.0e-5                    # m^3 (arbitrary; should not affect slope if wired correctly)
t_end = 5e-3                   # s for a short window (1–5 ms usually enough)
chemistry_off = True

# If your PlasmaPhase exposes qJ and qElastic in Python, set this to True.
# The code will try to call gas.joule_heating_power() and gas.elastic_power_loss().
USE_PLASMA_API = True

# If the API isn't exposed, set the constants below and flip USE_PLASMA_API to False
# so we can still compute predictions. (The reactor's measured slope will still tell you
# if Joule heating is actually being added by your core code.)
SIGMA_SI = 5.0      # S/m (example)
E_FIELD = 1.0e5     # V/m (example)
Q_ELASTIC = 0.0     # W/m^3 (if you model returns an elastic term, put it here)

# -----------------------------------------------------------------------------

def maybe_get_q_from_plasma(gas):
    print("entering maybe_get_q_from_plasma")
    qJ = None
    qE = None
    if not USE_PLASMA_API:
        qJ = SIGMA_SI * (E_FIELD ** 2)
        qE = Q_ELASTIC
        return qJ, qE

    print("1")

    # Try a few likely method/property names safely
    for name in ("joule_heating_power", "jouleHeatingPower", "q_joule"):
        print("2")
        f = getattr(gas, name, None)
        if callable(f):
            try:
                qJ = float(f())
                break
            except Exception:
                pass
        if isinstance(f, (int, float)):
            qJ = float(f)
        print("3")

    for name in ("elastic_power_loss", "elasticPowerLoss", "q_elastic"):
        print("4")
        f = getattr(gas, name, None)
        print("5")
        if callable(f):
            try:
                qE = float(f())
                break
            except Exception:
                pass
        if isinstance(f, (int, float)):
            qE = float(f)
        print("5")

    return qJ, qE

def main():
    gas = ct.Solution(MECH)
    print("Using mechanism:", MECH)
    gas.TPX = T0, P0, X0

    # Chemistry OFF (so heating is the only energy source)
    if chemistry_off:
        for i in range(gas.n_reactions):
            gas.set_multiplier(0.0, i)

    # Constant-volume, adiabatic reactor
    r = ct.IdealGasReactor(gas, energy="on", volume=V0)
    print(f"Reactor volume set to {V0:.3e} m^3")
    net = ct.ReactorNet([r])

    # Baseline properties at t=0
    rho0 = r.thermo.density
    cp0 = r.thermo.cp_mass
    cv0 = r.thermo.cv_mass
    T_start = r.T

    # Query plasma powers (W/m^3)
    qJ, qE = maybe_get_q_from_plasma(r.thermo)
    if qJ is None or qE is None:
        print("[warn] Could not read qJ/qElastic from PlasmaPhase API.")
    q_total = (qJ or 0.0) + (qE or 0.0)

    # Step a short interval and measure dT/dt
    t_log, T_log = [], []
    t = 0.0
    while t < t_end:
        t = net.step()
        t_log.append(t)
        T_log.append(r.T)

    # Compute measured slope using a simple finite difference over the full window
    dTdt_meas = (T_log[-1] - T_log[0]) / (t_log[-1] - t_log[0])

    # Predictions
    pred_cp = q_total / (rho0 * cp0) if q_total else None
    pred_cv = q_total / (rho0 * cv0) if q_total else None

    # Energy ledger (integral balance)
    # E_plasma = ∫ (q_total * V) dt ; ΔU_sens ≈ m * ∫ cv(T) dT
    E_plasma = (q_total * r.volume) * (t_log[-1] - t_log[0]) if q_total else None
    m0 = rho0 * r.volume
    # Approximate ∫cv(T)dT with trapezoid on a coarse grid using current cv(T0) as constant
    # (If you want, refine by sampling cv along T_log.)
    deltaU_sens = m0 * cv0 * (T_log[-1] - T_log[0])

    # Report
    print("\n=== CLOSED HOT BOX CHECK ===")
    print(f"T0 = {T_start:.2f} K, P0 = {P0/ct.one_atm:.3f} atm, V = {V0:.3e} m^3")
    print(f"rho0 = {rho0:.3f} kg/m^3, cp0 = {cp0:.1f} J/(kg·K), cv0 = {cv0:.1f} J/(kg·K)")
    print(f"Measured dT/dt  = {dTdt_meas:.3f} K/s")

    if q_total:
        print(f"qJ = {qJ:.3e} W/m^3, qElastic = {qE:.3e} W/m^3, q_total = {q_total:.3e} W/m^3")
        print(f"Pred dT/dt (cp) = {pred_cp:.3f} K/s")
        print(f"Pred dT/dt (cv) = {pred_cv:.3f} K/s")
        print(f"E_plasma ≈ {E_plasma:.3e} J,  ΔU_sens ≈ {deltaU_sens:.3e} J")
        if E_plasma != 0.0:
            print(f"Energy closure (ΔU/E_plasma) ≈ {deltaU_sens / E_plasma:.3f}")
    else:
        print("q_total unknown (API not read). Use E-scaling test below for presence/absence.")

    print("\nNext steps:")
    print("  • Re-run after doubling your model’s E (keep everything else identical).")
    print("  • Expect dT/dt_meas to ~quadruple (Joule ∝ E^2).")
    print("  • Then halve V0: dT/dt_meas should stay ≈ unchanged.")

if __name__ == "__main__":
    main()