import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plasmaCombustor import PlasmaCombustor

# -----------------------------------------------
# Load NPSS cycle data for the CFM56 test
# -----------------------------------------------
NPSS_data = pd.read_csv('CFM56_5B_EDB.out', delimiter=r"\s+", skiprows=[0,1])

k = 2
dPqP_comb = 0.04  # Pressure drop fraction

P3  = NPSS_data['Pt3[kPa]'].values[0::k] * (1 - dPqP_comb)
T3  = NPSS_data['Tt3[K]'].values[0::k]
m_a = NPSS_data['W3[kg/s]'].values[0::k]
m_f = NPSS_data['Wf[kg/s]'].values[0::k]
mf = m_f

print(f"Loaded {len(P3)} operating points")

# -----------------------------------------------
# Design settings: approximate CFM56 PZ/SZ
# -----------------------------------------------
phi_mean = 0.23   # typical lean
phi_std = 0.02
V_psr = 0.005     # 5 liters per swirl cup
V_pfr = 0.05      # 50 liters liner
n_psr = 5         # swirl cups
n_pfr = 20        # PFR segments
EN_func_PZ = lambda t, i: 0.0   # Off for now, or set a short pulse if you want
EN_func_SZ = lambda t, i: 0.0

# -----------------------------------------------
# Storage for results
# -----------------------------------------------
n_points = len(P3)

EINOx = np.zeros(n_points)
EICO  = np.zeros(n_points)

NOx_g = np.zeros(n_points)
CO_g  = np.zeros(n_points)

PZ_tres = []
PZ_phi  = []

# -----------------------------------------------
# Loop each operating point
# -----------------------------------------------
for n, (P3_pt, T3_pt, mdot_air, mdot_fuel) in enumerate(zip(P3, T3, m_a, m_f)):

    print(f"\n>>> Running point {n+1}/{n_points} <<<")
    print(f"  P3 = {P3_pt/1e3:.1f} bar, T3 = {T3_pt:.1f} K, mdot_air = {mdot_air:.2f} kg/s, mdot_fuel = {mdot_fuel:.4f} kg/s")

    # Total mass flow = air + fuel
    mdot_total = mdot_air + mdot_fuel

    comb = PlasmaCombustor(
        gas_file='gri30_plasma_cpavan.yaml',
        n_psr=n_psr,
        n_pfr=n_pfr,
        phi_mean=phi_mean,
        phi_std=phi_std,
        V_psr=V_psr,
        V_pfr=V_pfr,
        mdot_total=mdot_total,
        T0=T3_pt,
        P0=P3_pt * 1e3,  # kPa to Pa
        EN_func_PZ=EN_func_PZ,
        EN_func_SZ=EN_func_SZ,
        debug=False
    )

    psr_states, (pfr_fast, pfr_slow, merged) = comb.run(
        t_end_psr=1e-4,
        t_end_pfr=1e-4
    )

    NOx, CO, EI_NOx, EI_CO, _, _ = comb.compute_emissions(merged)

    EINOx[n] = EI_NOx
    EICO[n]  = EI_CO
    NOx_g[n] = NOx
    CO_g[n]  = CO

    # Save PZ phi and residence times
    PZ_tres.append(comb.PZ_tres_list)
    PZ_phi.append(comb.PZ_phi_list)

# -----------------------------------------------
# Compare with NPSS deck (optional)
# -----------------------------------------------
# Example: plot EI vs fuel flow
fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

ax[0].plot(mf, EINOx, 'o-k', label='PlasmaCombustor NOx')
ax[0].set_xlabel('Fuel flow [kg/s]')
ax[0].set_ylabel('EI_NOx [g/kg fuel]')
ax[0].legend()

ax[1].plot(mf, EICO, 's-k', label='PlasmaCombustor CO')
ax[1].set_xlabel('Fuel flow [kg/s]')
ax[1].set_ylabel('EI_CO [g/kg fuel]')
ax[1].legend()

fig.tight_layout()
plt.show()