import cantera as ct
import numpy as np

from plasmaReactors import PSR_Plasma, PFR_Plasma

class PlasmaCombustor:
    def __init__(self, gas_file, n_psr, n_pfr, phi_mean, phi_std, V_psr, V_pfr,
                 mdot_total, T0, P0, EN_func_PZ, EN_func_SZ, debug=False):
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

        # ------------------------------
        # Primary Zone: PSR array
        # ------------------------------
        # Sample phi for each PSR
        np.random.seed(42)
        self.phi_list = np.random.normal(phi_mean, phi_std, n_psr)
        self.psrs = []

        mdot_psr = mdot_total / n_psr  # equal split for now

        for i, phi in enumerate(self.phi_list):
            gas = ct.Solution(gas_file)

            FA_st = 17.2
            FA_actual = FA_st / phi

            mdot_psr = mdot_total / n_psr  # total PSR flow for this PSR
            mdot_air_psr = mdot_psr * FA_actual / (1 + FA_actual)
            mdot_fuel_psr = mdot_psr - mdot_air_psr

            L_fuel = 510e3  # J/kg — replace with real value for Jet-A

            Q_vap_per_sec = mdot_fuel_psr * L_fuel
            Q_vap_per_kg_mixture = Q_vap_per_sec / mdot_psr

            air_fuel_ratio = FA_actual
            X = f'CH4:1.0, O2:{2.0 * air_fuel_ratio}, N2:{7.52 * air_fuel_ratio}, e:1e-12'
            gas.TPX = T0, P0, X

            H_init = gas.enthalpy_mass
            H_adjusted = H_init - Q_vap_per_kg_mixture
            gas.HP = H_adjusted, P0

            psr = PSR_Plasma(gas, V=V_psr, mdot_in=mdot_psr,
                            T0=T0, P0=P0, EN_func=lambda t, idx=i: EN_func_PZ(t, idx),
                            debug=debug)
            self.psrs.append(psr)

        # ------------------------------
        # Secondary Zone: PFR chain
        # ------------------------------
        # We'll set the inlet mix later after PSR run
        self.pfr = None
        self.EN_func_SZ = EN_func_SZ
        self.V_pfr = V_pfr
        self.mdot_total = mdot_total

    def run(self, t_end_psr, t_end_pfr, dt=1e-9, dt_EN=1e-9):
        """
        Run PSRs to steady → mix → feed PFR → run PFR
        """
        # ------------------------------
        # Run all PSRs in parallel
        # ------------------------------
        psr_states = []
        mix_X = None
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

        gas_pfr = ct.Solution(self.gas_file)
        gas_pfr.TPY = T_mix, self.P0, mix_Y

        # ------------------------------
        # Now chain to PFR
        # ------------------------------
        self.pfr = PFR_Plasma(gas_pfr, V_total=self.V_pfr,
                              n_reactors=self.n_pfr, mdot_in=self.mdot_total,
                              T0=T_mix, P0=self.P0, EN_func=self.EN_func_SZ,
                              debug=self.debug)

        pfr_states = self.pfr.run(t_end_pfr, dt, dt_EN)

        return psr_states, pfr_states

comb = PlasmaCombustor(
    gas_file='gri30_plasma_cpavan.yaml',
    n_psr=2,
    n_pfr=3,
    phi_mean=0.8,
    phi_std=0.05,
    V_psr=5e-6,
    V_pfr=1e-4,
    mdot_total=1e-3,
    T0=300,
    P0=101325,
    EN_func_PZ=lambda t, i: 190e-21 * np.exp(-((t-24e-9)**2)/(2*3e-9**2)),
    EN_func_SZ=lambda t, i: 0.0,  # no SZ plasma yet
    debug=True
)

psr_states, pfr_states = comb.run(t_end_psr=5e-7, t_end_pfr=5e-7)