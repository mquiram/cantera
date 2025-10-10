import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import cantera as ct
from math import *
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

#import hy2combustor as Combustor  # Assuming renamed from old 'Combustor' for compatibility
import hybridCombustorOld1 as Combustor

ct.add_directory('../reacMechs')

# Optional custom style (commented out since not available)
# plt.style.use(["/home/prash/prash.mplstyle"])

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

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
Tt4 = NPSS_data['T04[R]'].values[0::k] * 100 / 180  # Convert Rankine to Kelvin

# -----------------------------------------------
# Set up combustor chemistry and geometry
# -----------------------------------------------
""" Combustor.setChem(
    RM='nDodecane_ReitzNO_plasma.yaml',              # Use n-Dodecane plasma mechanism
    Fuel='c12h26',                         # Fuel species
    Oxidizer='O2:0.2078, N2:0.782, H2O:0.0101'  # Air composition (wet)
) """

Combustor.setChem(
    RM='A2NOx_hitest.yaml',              # Use n-Dodecane plasma mechanism
    Fuel='POSF10325',                         # Fuel species
    Oxidizer='O2:0.2078, N2:0.782, H2O:0.0101'  # Air composition (wet)
)

Combustor.setGeometry(
    PZ_volume=0.00215,                  # Primary zone volume [m^3]
    SZ_volume=0.01125,                  # Secondary zone volume [m^3]
    SZ_length=0.075                     # Secondary zone axial length [m]
)

Combustor.setInletFlow(
    Pt3=P3[0] * 1000,                   # Convert kPa to Pa
    Tt3=T3[0],                          # Inlet total temperature [K]
    mdot_air=m_a[0],                    # Air mass flow rate [kg/s]
    mdot_fuel=m_f[0]                    # Fuel mass flow rate [kg/s]
)

# -----------------------------------------------
# Design parameter setup
# -----------------------------------------------
ϕPZdes = 2.37
ϕSZdes = 0.60
dil_start = 0.0
frac_dil_outer = 0.08 #0.3 #0.08
frac_dil_inner = 0.9
SZ_mdot_frac_outer = 0.5
SZ_A_frac_outer = 0.5
nPZ = 15 #2

Combustor.des_setup(
    PZ_desPhi=ϕPZdes,
    PZ_n_reactor=nPZ,
    SZ_desPhi=ϕSZdes,
    SZ_dilution=1,
    dil_start=dil_start,
    frac_dil_length_outer=frac_dil_outer,
    frac_dil_length_inner=frac_dil_inner,
    SZ_mdot_frac_outer=SZ_mdot_frac_outer,
    SZ_A_frac_outer=SZ_A_frac_outer,
    debug=False
)

print("Combustor configuration complete.")

# -----------------------------------------------
# Run full RQL-style simulation
# -----------------------------------------------
Params, PZ_results, SZ_exitStream, EI_NOx, EI_CO, NOx, CO = Combustor.run(printResults=0)

# Print PZ exit diagnostics for troubleshooting
#Combustor.print_pz_diagnostics(Params, top_n=25)  # increase/decrease top_n as you like
# -----------------------------------------------
# Output summary
# -----------------------------------------------
""" print("\n--- Final Emission Indices ---")
print(f"EI NOx = {EI_NOx:.4f} g/kg fuel")
print(f"EI CO  = {EI_CO:.4f} g/kg fuel")
print(f"NOx produced = {NOx:.3f} g")
print(f"CO  produced = {CO:.3f} g") """