# test_compare_plasma_heating.py
import os, sys, faulthandler
faulthandler.enable()
os.environ["PYTHONFAULTHANDLER"] = "1"

import cantera as ct

X0    = "N2:0.79, O2:0.21, e:1e-8"
T0    = 1200.0
P0    = ct.one_atm
V0    = 1.0e-5
CHEM_OFF = True
T_END = 3e-3

def run_once(mech):
    gas = ct.Solution(mech)  # assume default phase is the plasma-capable one
    gas.TPX = T0, P0, X0
    if CHEM_OFF:
        for i in range(gas.n_reactions):
            gas.set_multiplier(0.0, i)

    r = ct.IdealGasReactor(gas, energy="on", volume=V0)
    net = ct.ReactorNet([r])

    rho0 = r.thermo.density
    cp0  = r.thermo.cp_mass
    cv0  = r.thermo.cv_mass
    T_start = r.T

    t = 0.0
    t0 = None
    T_first = None
    try:
        while t < T_END:
            t = net.step()
            if t0 is None:
                t0 = t
                T_first = r.T
    except BaseException as e:
        return {"mech": mech, "error": f"{type(e).__name__}: {e}"}

    dTdt = (r.T - T_first) / (t - t0) if t0 is not None else float('nan')
    return {
        "mech": mech,
        "rho0": rho0,
        "cp0": cp0,
        "cv0": cv0,
        "T0": T_start,
        "dTdt_meas": dTdt,
        "t_window": (t - (t0 or 0.0))
    }

def main():
    mechs = sys.argv[1:] or []
    if not mechs:
        print("Usage: python test_compare_plasma_heating.py <mech_lowE.yaml> [mech_highE.yaml]")
        sys.exit(0)

    results = [run_once(m) for m in mechs]
    for r in results:
        if "error" in r:
            print(f"[{r['mech']}] ERROR: {r['error']}")
        else:
            print(f"[{r['mech']}] dT/dt = {r['dTdt_meas']:.3f} K/s over {r['t_window']:.3e} s "
                  f"(rho0={r['rho0']:.3f}, cp0={r['cp0']:.1f}, cv0={r['cv0']:.1f})")

    if len(results) == 2 and all("error" not in r for r in results):
        r1, r2 = results
        ratio = (r2["dTdt_meas"] / r1["dTdt_meas"]) if r1["dTdt_meas"] != 0 else float('inf')
        print(f"Scaling: dT/dt[{mechs[1]}] / dT/dt[{mechs[0]}] = {ratio:.2f}")

if __name__ == "__main__":
    main()
