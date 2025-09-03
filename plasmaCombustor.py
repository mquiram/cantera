import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

from plasmaReactors import PSR_Plasma, PFR_Plasma

class PlasmaCombustor:
    def __init__(self, gas_file, n_psr, n_pfr, phi_mean, phi_std, V_psr, V_pfr,
                 mdot_total, T0, P0, EN_func_PZ, EN_func_SZ, SZ_fast_frac,
                 DZ_fast_frac, fast_side_segments=[], slow_side_segments=[], debug=False,
                 PZ_airfrac=None, SZ_airfrac=None, DZ_airfrac=None):
        """
        gas_file: mechanism file, e.g. 'gri30_plasma_cpavan.yaml'
        n_psr: number of PSRs in parallel
        n_pfr: number of PFR segments
        phi_mean, phi_std: mean and std. dev. for PZ equivalence ratio
        V_psr: volume of each PSR [m^3]
        V_pfr: total PFR volume [m^3]
        mdot_total: total inlet mass flow [kg/s]
        T0, P0: inlet T and P
        EN_func_PZ: EN function for PSRs: EN(t, i)
        EN_func_SZ: EN function for PFR: EN(t, i)
        """
        self.n_psr = n_psr
        self.n_pfr = n_pfr
        self.debug = debug

        self.gas_file = gas_file
        self.T0 = T0
        self.P0 = P0
        self.mdot_total = mdot_total

        if PZ_airfrac is not None:
            self.PZ_airfrac = PZ_airfrac
            self.SZ_airfrac = SZ_airfrac
            self.DZ_airfrac = DZ_airfrac
            if debug:
                print(f"[DEBUG] Using user-specified air splits: "
                      f"PZ={self.PZ_airfrac:.3f}, SZ={self.SZ_airfrac:.3f}, DZ={self.DZ_airfrac:.3f}")
        else:
            # Default air fractions if none provided
            self.PZ_airfrac = 0.2
            self.SZ_airfrac = 0.7
            self.DZ_airfrac = 0.1
            if debug:
                print(f"[DEBUG] Using default air splits: "
                      f"PZ={self.PZ_airfrac:.3f}, SZ={self.SZ_airfrac:.3f}, DZ={self.DZ_airfrac:.3f}")


        # ------------------------------
        # Primary Zone: PSR array
        # ------------------------------
        # Sample phi for each PSR
        np.random.seed(42)
        self.phi_list = np.random.normal(phi_mean, phi_std, n_psr)
        self.psrs = []
        self.fuel_mdot_total = 0.0
        self.psr_res_times = []

        mdot_psr = mdot_total / n_psr  # equal split for now

        for i, phi in enumerate(self.phi_list):
            gas = ct.Solution(gas_file)

            FA_st = 15 #17.2
            FA_actual = FA_st / phi

            #mdot_psr = mdot_total / n_psr  # total PSR flow for this PSR
            mdot_PZ = self.mdot_total * self.PZ_airfrac
            mdot_psr = mdot_PZ / self.n_psr
            mdot_air_psr = mdot_psr * FA_actual / (1 + FA_actual)
            mdot_fuel_psr = mdot_psr - mdot_air_psr
            self.fuel_mdot_total += mdot_fuel_psr

            #L_fuel = 510e3  # J/kg — replace with real value for Jet-A
            L_fuel = 3e5

            Q_vap_per_sec = mdot_fuel_psr * L_fuel
            Q_vap_per_kg_mixture = Q_vap_per_sec / mdot_psr

            air_fuel_ratio = FA_actual
            X = f'c12h26:1.0, O2:{18.5 * air_fuel_ratio}, N2:{69.06 * air_fuel_ratio}, e:1e-12'
            #X = f'CH4:1.0, O2:{2.0 * air_fuel_ratio}, N2:{7.52 * air_fuel_ratio}, e:1e-12'
            #X = f'CH4:1.0, O2:{2.0 * 0.2}, N2:{7.52 * 0.2}, e:1e-12'
            gas.TPX = T0, P0, X

            rho_psr = gas.density_mass
            t_res_psr = V_psr / (mdot_psr / rho_psr)
            self.psr_res_times.append(t_res_psr)

            if self.debug:
                print(f"[PSR {i}] Residence time: {t_res_psr:.6e} s")

            H_init = gas.enthalpy_mass
            H_adjusted = H_init - Q_vap_per_kg_mixture
            #print(f"[DEBUG] H_init = {H_init:.2f}, Q_vap_per_kg_mixture = {Q_vap_per_kg_mixture:.2f}, H_adj = {H_adjusted:.2f}")
            gas.HP = H_adjusted, P0
            gas.equilibrate('HP')

            psr = PSR_Plasma(gas, V=V_psr, mdot_in=mdot_psr,
                            T0=T0, P0=P0, EN_func=lambda t, idx=i: EN_func_PZ(t, idx),
                            debug=debug)
            self.psrs.append(psr)

        # ------------------------------
        # Secondary Zone: PFR chain
        # ------------------------------
        # We'll set the inlet mix later after PSR run
        self.pfr = None
        self.EN_func_PZ = EN_func_PZ
        self.EN_func_SZ = EN_func_SZ
        self.V_pfr = V_pfr
        self.V_psr = V_psr


        #self.DZ_frac = DZ_frac
        self.SZ_fast_frac = SZ_fast_frac
        self.DZ_fast_frac = DZ_fast_frac
        self.fast_side_segments = fast_side_segments
        self.slow_side_segments = slow_side_segments

    def run(self, t_end_psr, t_end_pfr, dt=1e-5, dt_EN=1e-5):
        """
        Run PSRs to steady → mix → feed PFR → run PFR
        """
        # ------------------------------
        # Run all PSRs in parallel
        # ------------------------------
        psr_states = []
        #mix_X = None
        mix_Y = None
        mix_T = []
        mix_mdot = []

        for psr in self.psrs:
            state = psr.run(t_end_psr, dt, dt_EN)
            psr_states.append(state)
            thermo = psr.reactor.thermo
            mix_T.append(thermo.T)
            mix_Y = thermo.Y if mix_Y is None else mix_Y + thermo.Y
            mix_mdot.append(psr.mdot_in)

        mix_Y /= self.n_psr
        T_mix = np.mean(mix_T)

        #gas_pfr = ct.Solution(self.gas_file)
        #gas_pfr.TPY = T_mix, self.P0, mix_Y

        gas_pfr_fast = ct.Solution(self.gas_file)
        gas_pfr_fast.TPY = T_mix, self.P0, mix_Y

        gas_pfr_slow = ct.Solution(self.gas_file)
        gas_pfr_slow.TPY = T_mix, self.P0, mix_Y

        # Compute side dilution air split
        #DZ_frac = 0.2
        #DZ_mdot_air = self.mdot_total * self.DZ_frac
        #SZ_core_mdot = self.mdot_total - DZ_mdot_air
        DZ_mdot_air = self.mdot_total * self.DZ_airfrac
        SZ_core_mdot = self.mdot_total * self.SZ_airfrac

        # User-controlled splits
        SZ_core_mdot_fast = SZ_core_mdot * self.SZ_fast_frac
        SZ_core_mdot_slow = SZ_core_mdot * (1 - self.SZ_fast_frac)

        DZ_mdot_air_fast = DZ_mdot_air * self.DZ_fast_frac
        DZ_mdot_air_slow = DZ_mdot_air * (1 - self.DZ_fast_frac)

        side_inlets_fast = [
            {"segment_idx": seg, "mdot": DZ_mdot_air_fast / len(self.fast_side_segments), "composition": "O2:1.0, N2:3.76"}
            for seg in self.fast_side_segments
        ]

        side_inlets_slow = [
            {"segment_idx": seg, "mdot": DZ_mdot_air_slow / len(self.slow_side_segments), "composition": "O2:1.0, N2:3.76"}
            for seg in self.slow_side_segments
        ]
        #side_inlets_fast = []
        #side_inlets_slow = []

        # Fast mode PFR
        self.pfr_fast = PFR_Plasma(
            gas_pfr_fast, V_total=self.V_pfr,
            n_reactors=self.n_pfr,
            mdot_in=SZ_core_mdot_fast,
            T0=T_mix, P0=self.P0,
            EN_func=self.EN_func_SZ,
            debug=self.debug,
            side_inlets=side_inlets_fast,
            L_total=0.075
        )

        # Slow mode PFR
        self.pfr_slow = PFR_Plasma(
            gas_pfr_fast, V_total=self.V_pfr,
            n_reactors=self.n_pfr,
            mdot_in=SZ_core_mdot_slow,
            T0=T_mix, P0=self.P0,
            EN_func=self.EN_func_SZ,
            debug=self.debug,
            side_inlets=side_inlets_slow,
            L_total=0.075
        )

        # Run both
        pfr_states_fast = self.pfr_fast.run(t_end_pfr, dt, dt_EN)
        pfr_states_slow = self.pfr_slow.run(t_end_pfr, dt, dt_EN)

        # Merge
        Y_fast = pfr_states_fast.Y[-1, :]
        Y_slow = pfr_states_slow.Y[-1, :]

        Y_merged = (
            Y_fast * SZ_core_mdot_fast + Y_slow * SZ_core_mdot_slow
        ) / (SZ_core_mdot_fast + SZ_core_mdot_slow)

        import types
        merged_states = types.SimpleNamespace()
        merged_states.Y = np.array([Y_merged])

        return psr_states, (pfr_states_fast, pfr_states_slow, merged_states), gas_pfr_fast

    def compute_emissions(self, pfr_states, initial_thermo=None):
        """
        Computes NOx, CO mass [g], EI [g/kg fuel] at PFR exit
        and prints detailed emissions change across the SZ.

        Parameters:
            pfr_states: the output states from the PFR
            initial_thermo: Cantera object representing pre-PFR gas state
        """
        gas = self.pfr_fast.gas
        NO_idx = gas.species_index("NO")
        NO2_idx = gas.species_index("NO2")
        CO_idx = gas.species_index("CO")

        # Final values (at PFR exit)
        Y_NO_end = pfr_states.Y[-1, NO_idx]
        Y_NO2_end = pfr_states.Y[-1, NO2_idx]
        Y_CO_end = pfr_states.Y[-1, CO_idx]

        # Start of SZ (average from PSRs or initial gas)
        if initial_thermo:
            Y_NO_start = initial_thermo.Y[NO_idx]
            Y_NO2_start = initial_thermo.Y[NO2_idx]
            Y_CO_start = initial_thermo.Y[CO_idx]
        else:
            raise ValueError("Must pass initial_thermo (post-PSR mix) to compute SZ deltas")

        MW_NO = 30.01  # g/mol
        MW_NO2 = 46.01
        MW_CO = 28.01

        rho_exit = gas.density_mass  # kg/m^3
        V_pfr = self.V_pfr
        m_exit = rho_exit * V_pfr  # kg gas mass in PFR

        # Start of SZ masses
        NOx_g_start = (Y_NO_start * MW_NO + Y_NO2_start * MW_NO2) * m_exit
        CO_g_start = Y_CO_start * MW_CO * m_exit

        # End of SZ masses
        NOx_g_end = (Y_NO_end * MW_NO + Y_NO2_end * MW_NO2) * m_exit
        CO_g_end = Y_CO_end * MW_CO * m_exit

        NOx_prod = NOx_g_end - NOx_g_start
        CO_prod = CO_g_end - CO_g_start

        EI_NOx = NOx_g_end / self.fuel_mdot_total
        EI_CO = CO_g_end / self.fuel_mdot_total

        # Mass balance check
        mdot_exit = self.pfr_fast.mdot_in + self.pfr_slow.mdot_in  # or full SZ outflow

        print("\n--------------------------- CO & NOx ------------------------------------")
        print(f"CO at start of SZ  = {CO_g_start:6.2E}\tCO  produced  = {CO_prod: .2E}\tCO end  = {CO_g_end:.3f}")
        print(f"NOx at start of SZ = {NOx_g_start:6.2E}\tNOx produced  = {NOx_prod: .2E}\tNOx end = {NOx_g_end:.3f}")
        #print(f"Mass Flow check: {mdot_exit:.3f} = {mdot_exit:.3f}")
        print(f"Total SZ mass flow (PFR exit): {mdot_exit:.6e} kg/s")
        print(f"Fast core: {self.pfr_fast.mdot_in:.6e}, Slow core: {self.pfr_slow.mdot_in:.6e}")
        print("-------------------------------------------------------------------------")

        return NOx_g_end, CO_g_end, EI_NOx, EI_CO

    def setInletFlow(Pt3, Tt3, mdot_air, mdot_fuel):
        global inlet_conditions
        inlet_conditions = {
            "Pt3": Pt3,
            "Tt3": Tt3,
            "mdot_air": mdot_air,
            "mdot_fuel": mdot_fuel
        }

    def setup(PZ_airfrac, SZ_airfrac, DZ_airfrac, debug=False):
        global air_fracs
        air_fracs = {
            "PZ": PZ_airfrac,
            "SZ": SZ_airfrac,
            "DZ": DZ_airfrac
        }
        if debug:
            print("Air fractions set:", air_fracs)

""" comb = PlasmaCombustor(
    gas_file='nDodecane_ReitzNO_plasma.yaml',   # Your plasma-enabled mechanism gri30_plasma_cpavan.yaml
    n_psr=2,                               # Small number of PSRs
    n_pfr=3,                               # Small number of PFR segments per chain
    phi_mean=2.5,                          # Typical lean combustor value  0.8
    phi_std=0.1,                          # Smaller spread → stable test  0.02
    V_psr=1e-6,                            # 5 cc per PSR
    V_pfr=1e-4,                            # 100 cc total PFR chain
    mdot_total=1e-3,                       # Small flow → easy to converge
    T0=1200,                                # Room temp
    P0=101325,                             # 1 atm
    EN_func_PZ=lambda t, i: 0.0, #190e-21 * np.exp(-((t-24e-9)**2)/(2*3e-9**2)),
    EN_func_SZ=lambda t, i: 0.0,           # Off for now → just test PSR coupling
    DZ_frac=0.4,
    SZ_fast_frac=0.6,
    DZ_fast_frac=0.5,
    fast_side_segments=[],
    slow_side_segments=[],
    debug=True                             #
) """

""" psr_states, pfr_states = comb.run(t_end_psr=1e-6, t_end_pfr=1e-6)

NOx_g, CO_g, EI_NOx, EI_CO, NOx_profile, axial_positions = comb.compute_emissions(pfr_states) """

""" psr_states, (pfr_states_fast, pfr_states_slow, merged_states) = comb.run(t_end_psr=1e-6, t_end_pfr=1e-6)

NOx_g, CO_g, EI_NOx, EI_CO, NOx_profile, axial_positions = comb.compute_emissions(merged_states)

print("\n=== Emissions summary ===")
print(f"NOx (g/s): {NOx_g:.6e}")
print(f"CO  (g/s): {CO_g:.6e}")
print(f"EI_NOx (g/kg fuel): {EI_NOx:.3f}")
print(f"EI_CO  (g/kg fuel): {EI_CO:.3f}")
print("\nAxial NOx profile [ppm]:")
for z, NO_ppm in zip(axial_positions, NOx_profile):
    print(f"Segment {z:.1f}: NO = {NO_ppm:.1f} ppm")

# Use the fast PFR for demonstration:
states = pfr_states_fast  # or merged, if you merged them

# Get unique segments:
segments = np.unique(states.segment_idx)
dz = comb.pfr_fast.L_total / comb.pfr_fast.n_reactors
z = segments * dz  # axial position

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# ---- Plasma species (example) ----
species1 = ['e', 'O2+', 'N2+', 'N2(A)', 'N2(B)', 'O']

for sp in species1:
    if sp in states.species_names:
        mole_fractions = []
        for seg in segments:
            idx = (states.segment_idx == seg)
            mole_fractions.append(states.X[idx, states.species_index(sp)][-1])
        ax[0].plot(z, mole_fractions, label=sp)

ax[0].set_yscale('log')
ax[0].set_ylim([1e-14, 1e-3])
ax[0].set_ylabel('Mole fraction [-]')
ax[0].legend()
ax[0].set_title('Plasma Species along Combustor Length')

# ---- Major species and Temperature ----
species2 = ['CH4', 'O2', 'CO', 'CO2', 'OH', 'H', 'H2O']

for sp in species2:
    if sp in states.species_names:
        mole_fractions = []
        for seg in segments:
            idx = (states.segment_idx == seg)
            mole_fractions.append(states.X[idx, states.species_index(sp)][-1])
        ax[1].plot(z, mole_fractions, label=sp)

axT = ax[1].twinx()
T_profile = []
for seg in segments:
    idx = (states.segment_idx == seg)
    T_profile.append(states.T[idx][-1])

axT.plot(z, T_profile, 'k--', label='T [K]')
axT.set_ylabel('T [K]')

ax[1].set_xlabel('Axial Length [m]')
ax[1].set_ylabel('Mole fraction [-]')
ax[1].legend(loc='upper left')
axT.legend(loc='upper right')
ax[1].set_title('Main Species and Temperature along Combustor Length')

plt.tight_layout()
plt.show() """