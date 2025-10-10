# test_wall_heat_injection.py  (fixed)
import os, sys, faulthandler
faulthandler.enable()
os.environ["PYTHONFAULTHANDLER"] = "1"

import cantera as ct

MECH  = "A2NOx_hitest.yaml"
X0    = "N2:0.79, O2:0.21, e:1e-8"
T0    = 1200.0
THOT  = 1500.0
P0    = ct.one_atm
V0    = 1.0e-4

Awall = 0.01
Uwall = 50.0
T_END = 5e-2

CHEMISTRY_OFF = True

def main():
    gas = ct.Solution(MECH)
    gas.TPX = T0, P0, X0
    if CHEMISTRY_OFF:
        for i in range(gas.n_reactions):
            gas.set_multiplier(0.0, i)

    r = ct.IdealGasReactor(gas, energy="on", volume=V0)

    env_gas = ct.Solution(MECH)
    env_gas.TPX = THOT, P0, X0
    env = ct.Reservoir(env_gas)

    # Correct: use U (heat transfer coeff), connect env (left) -> reactor (right)
    w = ct.Wall(left=env, right=r, A=Awall, U=Uwall)

    net = ct.ReactorNet([r])

    rho0 = r.thermo.density
    cp0  = r.thermo.cp_mass
    T_start = r.T

    Qdot0 = Uwall * Awall * (THOT - T_start)          # W
    dTdt_pred = Qdot0 / (rho0 * cp0 * V0)             # K/s

    print("=== WALL HEAT INJECTION CHECK (fixed) ===")
    print(f"YAML={MECH}")
    print(f"T0={T_start:.2f} K, THOT={THOT:.2f} K, P0={P0/ct.one_atm:.3f} atm, V={V0:.3e} m^3")
    print(f"rho0={rho0:.3f} kg/m^3, cp0={cp0:.1f} J/(kg·K)")
    print(f"A={Awall:.3f} m^2, U={Uwall:.1f} W/(m^2*K)")
    print(f"Predicted dT/dt (cp) ≈ {dTdt_pred:.1f} K/s")

    t = 0.0
    t_log = []
    T_log = []
    try:
        while t < T_END:
            t = net.step()
            t_log.append(t)
            T_log.append(r.T)
    except BaseException as e:
        print(f"[CRASH/EXCEPTION] {type(e).__name__}: {e}")
        print("Last good time:", (t_log[-1] if t_log else 0.0))
        return

    if len(t_log) >= 2:
        dTdt_meas = (T_log[-1] - T_log[0]) / (t_log[-1] - t_log[0])
    else:
        dTdt_meas = float('nan')

    print(f"Measured dT/dt = {dTdt_meas:.1f} K/s")

if __name__ == "__main__":
    main()
