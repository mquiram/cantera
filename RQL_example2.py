import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import cantera as ct
from math import *
import scipy.integrate
import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

import plasmaCombustor as Combustor  # Assuming renamed from old 'Combustor' for compatibility

ct.add_directory('../reacMechs')

# Optional custom style (commented out since not available)
# plt.style.use(["/home/prash/prash.mplstyle"])

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

def compute_design_air_splits(PZ_phi_des, SZ_phi_des,
                               mdot_total=1.0, fuel_frac=0.03, FA_st=15.0):
    """
    Mimics des_setup() to compute PZ, SZ, and DZ air fractions for PlasmaCombustor.
    Inputs:
        PZ_phi_des: desired φ in PZ
        SZ_phi_des: desired φ in SZ
        fuel_name: species name of fuel (e.g., 'c12h26')
        fuel_MW: molecular weight of fuel [g/mol] (adjust if needed)
        mdot_total: total mass flow [kg/s]
        fuel_frac: fuel mass flow fraction (typically very small)
        FA_st: stoichiometric air-fuel ratio (mass basis), default ≈15 for hydrocarbons
    Returns:
        Tuple: (PZ_airfrac, SZ_airfrac, DZ_airfrac, mdot_fuel)
    """

    # Approximate fuel/air split
    mdot_fuel = mdot_total * fuel_frac
    mdot_air_total = mdot_total - mdot_fuel

    # Air required in PZ
    mdot_air_PZ = mdot_fuel * FA_st / PZ_phi_des

    # Residual fuel goes to SZ
    mdot_fuel_SZ = mdot_fuel  # assume all fuel goes through PZ
    mdot_air_SZ = mdot_fuel_SZ * FA_st / SZ_phi_des

    # Remaining air → dilution zone
    mdot_air_DZ = mdot_air_total - mdot_air_PZ - mdot_air_SZ

    if mdot_air_DZ < 0:
        raise ValueError("Negative air available for dilution — check phi values.")

    # Normalize to total air
    PZ_airfrac = mdot_air_PZ / mdot_air_total
    SZ_airfrac = mdot_air_SZ / mdot_air_total
    DZ_airfrac = mdot_air_DZ / mdot_air_total

    return PZ_airfrac, SZ_airfrac, DZ_airfrac, mdot_fuel


# Load NPSS data
NPSS_data = pd.read_csv('CFM56_5B_EDB.out', delimiter=r"\s+", skiprows=[0,1])

k = 2
dPqP_comb = 0.04
P3  = NPSS_data['Pt3[kPa]'].values[0::k] * (1 - dPqP_comb)
T3  = NPSS_data['Tt3[K]'].values[0::k]
m_a = NPSS_data['W3[kg/s]'].values[0::k]
m_f = NPSS_data['Wf[kg/s]'].values[0::k]
Tt4 = NPSS_data['T04[R]'].values[0::k] * 100 / 180
mf  = m_f

# Setup mechanism and geometry
#Combustor.setChem('nDodecane_ReitzNO_plasma.yaml','c12h26','O2:0.2078, N2:0.782, H2O:0.0101')
#Combustor.setGeometry(PZ_volume=0.00215, SZ_volume=0.01125, SZ_length=0.075)
#Combustor.setInletFlow(Pt3=P3[0]*1000, Tt3=T3[0], mdot_air=m_a[0], mdot_fuel=m_f[0])

PZ_airfrac, SZ_airfrac, DZ_airfrac, mdot_fuel = compute_design_air_splits(
    PZ_phi_des=2.37,
    SZ_phi_des=0.60,
    mdot_total=m_a[0] + m_f[0],
    fuel_frac=m_f[0] / (m_a[0] + m_f[0]),
    FA_st=15.0  # can be adjusted for Jet-A
)

comb = Combustor.PlasmaCombustor(
    gas_file='nDodecane_ReitzNO_plasma.yaml',
    n_psr=15,
    n_pfr=3,
    phi_mean=2.37,
    phi_std=0.01,  # small spread to match uniform phi assumption
    V_psr=0.00215/15,  # split evenly across PSRs
    V_pfr=0.01125,     # entire SZ volume
    mdot_total=m_a[0] + m_f[0],
    T0=T3[0],
    P0=P3[0]*1000,
    EN_func_PZ=lambda t, i: 0.0,
    EN_func_SZ=lambda t, i: 0.0,
    #DZ_frac=DZ_airfrac,
    SZ_fast_frac=0.5,
    DZ_fast_frac=0.5,
    fast_side_segments=[],#[0, 1],
    slow_side_segments=[],#list(range(18)),
    debug=False,
    PZ_airfrac=PZ_airfrac,
    SZ_airfrac=SZ_airfrac,
    DZ_airfrac=DZ_airfrac
)
print(f"Design air fractions: PZ={PZ_airfrac:.3f}, SZ={SZ_airfrac:.3f}, DZ={DZ_airfrac:.3f}")

Params = lambda: None
Params.PZ_phi_mean = np.mean(comb.phi_list)
Params.PZ_phi_lst = comb.phi_list
Params.PZ_tres_lst = comb.psr_res_times

psr_states, (pfr_fast, pfr_slow, merged), gas_mix = comb.run(t_end_psr=1e-4, t_end_pfr=1e-4)
NOx_g, CO_g, EI_NOx, EI_CO, *_ = comb.compute_emissions(pfr_fast, initial_thermo=gas_mix)
NOx_g, CO_g, EI_NOx, EI_CO, *_ = comb.compute_emissions(pfr_slow, initial_thermo=gas_mix)
NOx_g, CO_g, EI_NOx, EI_CO, *_ = comb.compute_emissions(merged, initial_thermo=gas_mix)

PZ_results = [None, None, None, NOx_g, CO_g]
states_SZ = merged

""" # Design configuration
phiPZdes = 2.37
phiSZdes = 0.60
dil_start = 0.
frac1 = 0.08
frac2 = 0.9
SZ_mdot_frac_outer = 0.5
SZ_A_frac_outer = 0.5
nPZ = 15

Combustor.des_setup(
    PZ_desPhi=phiPZdes, PZ_n_reactor=nPZ,
    SZ_desPhi=phiSZdes, SZ_dilution=1, dil_start=dil_start,
    frac_dil_length_outer=frac1, frac_dil_length_inner=frac2,
    SZ_mdot_frac_outer=SZ_mdot_frac_outer, SZ_A_frac_outer=SZ_A_frac_outer,
    debug=False)

# Run first point to initialize
#Params, PZ_results, states_SZ, EINOx, EICO, NOx, CO = Combustor.run(0)
psr_states, (pfr_fast, pfr_slow, merged) = comb.run(t_end_psr=1e-4, t_end_pfr=1e-4)
NOx_g, CO_g, EINOx, EICO, *_ = comb.compute_emissions(merged)
Params.PZ_phi_lst = comb.phi_list
Params.PZ_tres_lst = comb.psr_res_times
PZ_results = [None, None, None, NOx_g, CO_g]
states_SZ = merged """

""" # Load EDB comparison data
CFM56_5B = pd.read_csv('../../NPSS/CFM56-5B_simple2.csv')
print("Initial EI(NOx) Cantera vs EDB:", EINOx, CFM56_5B["EI(NOx)"].values[0])

# Run all points
nTot = len(P3)
PZ_phi_lst, PZ_tres_lst = [], []
EINOx = np.zeros(nTot)
EICO  = np.zeros(nTot)
PZ_NOx = np.zeros(nTot)
PZ_CO  = np.zeros(nTot)
NOx = np.zeros(nTot)
CO  = np.zeros(nTot)
PZphi = np.zeros(nTot)
Tt4_out = np.zeros(nTot)
NOxppm = np.zeros(nTot)

for n, P3_i, T3_i, m_a_i, m_f_i in zip(count(), P3, T3, m_a, m_f):
    Combustor.setInletFlow(Pt3=P3_i*1000, Tt3=T3_i, mdot_air=m_a_i, mdot_fuel=m_f_i)

    Combustor.setup(debug=False,
        PZ_airfrac=0.17709188820674876,
        SZ_airfrac=0.6995129584166577,
        DZ_airfrac=0.12339515337659357)

    Params, PZ_results, states_SZ, EINOx[n], EICO[n], NOx[n], CO[n] = Combustor.run(0)
    print(Params.PZ_airfrac, Params.SZ_airfrac)

    PZ_NOx[n] = PZ_results[3]
    PZ_CO[n]  = PZ_results[4]
    PZphi[n] = Params.PZ_phi_mean
    Tt4_out[n] = states_SZ.T
    PZ_phi_lst.append(Params.PZ_phi_lst)
    PZ_tres_lst.append(Params.PZ_tres_lst)

# Plot results
fig, ax = plt.subplots(1, 2, dpi=200, figsize=(8,5))

CFM56_5B.plot(ax=ax[0], kind='line', x='Wf', y='EI(NOx)', label='NO$_x$ EDB', logy=False, style='.-')
CFM56_5B.plot(ax=ax[1], kind='line', x='Wf', y='EI(CO)', label='CO EDB', logy=False, style='.-')

ax[0].plot(mf, EINOx, '.-k', label='Cantera')
ax[0].legend()
ax[0].set_ylabel('EI(NO$_x$) [g/kg]')

ax[1].plot(mf, EICO, '.-k', label='Cantera')
ax[1].legend()
ax[1].set_ylabel('EI(CO) [g/kg]')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
 """