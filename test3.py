# test3.py — stable hot-box check (CV)
# - honors energy ON/OFF per MODE
# - robust dT/dt even if the first step has Δt = 0
# - no getattr into PlasmaPhase from Python

import math, faulthandler, cantera as ct
faulthandler.enable()

# ---------- USER SETTINGS ----------
MECH  = "A2NOx_hitestlow.yaml"   # your YAML
PHASE = "plasma"                 # phase name in YAML (kept for display)
X0 = "N2:0.79, O2:0.21, e:1e-8"
T0 = 1200.0                      # K
P0 = ct.one_atm                  # Pa
V0 = 1.0e-5                      # m^3
MODE = "C"                       # "A": energy off; "B/C": energy on
CHEMISTRY_OFF = True             # leave True to test just energy injection
T_END = 5e-3                     # s
# -----------------------------------

def step_until_advanced(net, r, tries=12):
    """Take steps until time advances; return instantaneous slope or NaN."""
    t0, T0 = net.time, r.T
    for _ in range(tries):
        net.step()
        t1, T1 = net.time, r.T
        if t1 > t0:
            return (T1 - T0) / (t1 - t0)
    return float('nan')

def main():
    gas = ct.Solution(MECH)   # default phase assumed plasma-capable
    gas.TPX = T0, P0, X0

    if CHEMISTRY_OFF:
        for i in range(gas.n_reactions):
            gas.set_multiplier(0.0, i)

    energy_flag = "on" if MODE in ("B", "C") else "off"

    # Constant-volume reactor is simplest for this test
    r = ct.IdealGasReactor(gas, energy=energy_flag, volume=V0)
    net = ct.ReactorNet([r])
    # Give the solver some breathing room; adjust if you still see tiny steps
    net.rtol = 1e-9
    net.atol = 1e-12

    rho0 = r.thermo.density
    cp0  = r.thermo.cp_mass
    cv0  = r.thermo.cv_mass
    T_start = r.T

    print("=== HOT BOX SAFE CHECK ===")
    print(f"Mode={MODE}  energy={energy_flag}")
    print(f"YAML={MECH}  phase={PHASE}")
    print(f"T0={T_start:.2f} K, P0={P0/ct.one_atm:.3f} atm, V={V0:.3e} m^3")
    print(f"rho0={rho0:.3f} kg/m^3, cp0={cp0:.1f} J/(kg·K), cv0={cv0:.1f} J/(kg·K)")

    # Instantaneous slope near t≈0 (robust to Δt=0)
    dTdt0 = step_until_advanced(net, r)
    if math.isfinite(dTdt0):
        print(f"Instantaneous dT/dt @ t≈0: {dTdt0:.3f} K/s")
    else:
        print("Instantaneous dT/dt @ t≈0: nan (no advancement after retries)")

    # Continue to T_END and report whole-window slope
    def advance_in_chunks(net, r, t_end, n_chunks=50):
        t0, T0 = net.time, r.T
        if t_end <= t0:
            return float('nan')
        dt = max((t_end - t0) / float(n_chunks), 1e-8)  # ensure nonzero step targets
        t = t0
        for _ in range(n_chunks):
            t_target = t + dt
            if t_target > t_end:
                t_target = t_end
            net.advance(t_target)       # guarantees advancement to target
            t = net.time
            if t >= t_end:
                break
        # compute whole-window slope
        return (r.T - T0) / (net.time - t0) if net.time > t0 else float('nan')

    dTdt_win = advance_in_chunks(net, r, T_END)
    print(f"Measured dT/dt = {dTdt_win:.3f} K/s")
    # --- end patch ---
if __name__ == "__main__":
    main()