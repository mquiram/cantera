import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

class PSR_Plasma2:
    def __init__(self, gas, V, mdot_in, T0, P0, EN_func=None, debug=False):
        """
        PSR using Cantera IdealGasReactor with native plasma EEDF updates.

        gas: Cantera plasma Solution with EN support
        V: reactor volume [m^3]
        mdot_in: inlet mass flow [kg/s]
        T0: inlet temp [K]
        P0: inlet pressure [Pa]
        EN_func: function of time, EN(t) in Td
        """
        self.gas = gas
        self.V = V
        self.mdot_in = mdot_in
        self.T0 = T0
        self.P0 = P0
        self.EN_func = EN_func
        self.debug = debug
        #self.gas_file = gas.name if gas.name else 'gri30_plasma_cpavan.yaml'

        # Inlet
        """ self.inlet = ct.Reservoir(gas)
        # Reactor
        self.reactor = ct.IdealGasReactor(gas, volume=V)
        # Outlet
        self.outlet = ct.Reservoir(gas)

        # Mass flow
        self.mfc = ct.MassFlowController(self.inlet, self.reactor, mdot=mdot_in)
        self.outlet_mfc = ct.MassFlowController(self.reactor, self.outlet, mdot=mdot_in)
        #self.valve = ct.PressureController(self.reactor, self.outlet, K=1e-5)

        # Reactor network
        self.sim = ct.ReactorNet([self.reactor]) """

        # Ensure reactor starts at (T0, P0) so the held pressure is P0
        self.gas.TP = T0, P0

        # Inlet & outlet reservoirs (constant thermodynamic states)
        # Note: Using the same Solution is fine as long as we don't mutate it after creating reservoirs.
        self.inlet = ct.Reservoir(self.gas)
        self.outlet = ct.Reservoir(self.gas)

        # --- Constant-pressure reactor (isobaric by construction) ---
        # Volume adjusts internally so that the reactor pressure remains equal to the initial pressure
        # of the contained gas (we set TP above), i.e., ~P0.
        self.reactor = ct.IdealGasConstPressureReactor(self.gas, energy='on', volume=V)

        # Inlet/outlet devices: match mass flow so reactor mass stays ~constant
        self.mfc_in = ct.MassFlowController(self.inlet, self.reactor, mdot=mdot_in)
        self.mfc_out = ct.MassFlowController(self.reactor, self.outlet, mdot=mdot_in)

        # Reactor network
        self.sim = ct.ReactorNet([self.reactor])

    """ def disable_two_temp_plasma_reactions(gas, verbose=True):
        kin = gas.kinetics
        n_disabled = 0
        for i, rxn in enumerate(kin.reactions()):
            rtype = getattr(rxn, "reaction_type", "")
            # robust fallback if reaction_type is missing:
            rate = getattr(rxn, "rate", None)
            is_two_temp = (rtype == "two-temperature-plasma") or \
                        (rate is not None and rate.__class__.__name__.lower().startswith("twotemp"))
            if is_two_temp:
                kin.set_multiplier(0.0, i)
                n_disabled += 1
                if verbose:
                    try:
                        eqn = rxn.equation
                    except Exception:
                        eqn = "<equation unavailable>"
                    print(f"disabled [{i:03d}] {rtype:>24s} : {eqn}")
        if verbose:
            print(f"Total disabled two-temp-plasma reactions: {n_disabled}") """

    def run(self, t_end, dt=1e-5, dt_EN=1e-5):
        """
        Run the PSR to t_end [s], updating EN at every dt_EN.
        """
        t = 0.0
        states = ct.SolutionArray(self.gas, extra=['t'])
        states = ct.SolutionArray(self.reactor.thermo, extra=['t'])

        """ while t < t_end:
            EN_now = self.EN_func(t)
            self.gas.EN = EN_now
            self.gas.update_EEDF()
            self.sim.reinitialize()

            # advance by small timestep
            self.sim.advance(t + dt)
            states.append(self.reactor.thermo.state, t=self.sim.time)

            if self.debug:
                print(f"t={self.sim.time:.3e} s | T={self.reactor.T:.1f} K | EN={EN_now:.3e} Td")

            t += dt_EN """

        while t < t_end:
            #EN_now = self.EN_func(t)
            #self.gas.EN = EN_now
            #self.gas.update_EEDF()
            #self.sim.reinitialize()

            # advance by small timestep
            #self.sim.advance(t + dt)
            #t_old = t
            t = self.sim.step()
            #print(f"dt_step = {t - t_old:.3e} s")
            states.append(self.reactor.thermo.state, t=t)

            if self.debug:
                print(f"t={self.sim.time:.3e} s | T={self.reactor.T:.1f} K | EN={EN_now:.3e} Td")

        return states

class PSR_Plasma:
    def __init__(self, gas_feed, V, mdot_in, T0, P0, EN_func=None, debug=False, gas0=None, mdot_air=None, mdot_fuel=None, L_fuel_inlet=0.0, fuelStr=None, oxidizerStr=None):
        """
        Constant-P PSR. 'gas_feed' is the fresh inlet mixture at (T0,P0).
        'gas0' (optional) is the reactor's initial state (e.g., HP pre-eq).
        """
        import cantera as ct

        self.V = V
        self.mdot_in = mdot_in
        self.T0 = T0
        self.P0 = P0
        self.EN_func = EN_func
        self.debug = debug

        # --- Inlet (fresh) ---
        gas_feed.TP = T0, P0   # keep feed at supply conditions   #1000 or T0
        self.inlet = ct.Reservoir(gas_feed)

        # --- Reactor (constant P) ---
        self.rgas = gas0 if gas0 is not None else gas_feed  # if gas0 provided, do NOT reset its TP
        self.reactor = ct.IdealGasConstPressureReactor(self.rgas, energy='on', volume=V)

        # --- Outlet reference reservoir at (T0,P0) with same mechanism ---
        mech = getattr(gas_feed, "source", None) or getattr(gas_feed, "name", None)
        env = ct.Solution(mech, transport_model='None')
        env.TPX = T0, P0, gas_feed.X
        self.outlet = ct.Reservoir(env)

        # --- Inlet(s) ---
        if (mdot_air is not None) and (mdot_fuel is not None) and (fuelStr is not None) and (oxidizerStr is not None):
            gas_air = ct.Solution(mech, transport_model='None');  gas_air.TPX = T0, P0, oxidizerStr
            gas_f   = ct.Solution(mech, transport_model='None');  gas_f.TPX   = T0, P0, {fuelStr: 1.0}
            if L_fuel_inlet > 0.0:
                # lower the fuel stream enthalpy by latent (sets a slightly cooler T)
                gas_f.HP = gas_f.enthalpy_mass - L_fuel_inlet, P0

            self.inlet_air  = ct.Reservoir(gas_air)
            self.inlet_fuel = ct.Reservoir(gas_f)
            self.mfc_air  = ct.MassFlowController(self.inlet_air,  self.reactor, mdot=mdot_air)
            self.mfc_fuel = ct.MassFlowController(self.inlet_fuel, self.reactor, mdot=mdot_fuel)
            self.mfco     = ct.MassFlowController(self.reactor,    self.outlet,  mdot=(mdot_air + mdot_fuel))
        else:
            # fall back: single premixed inlet, as you had
            gas_feed.TP = T0, P0
            self.inlet = ct.Reservoir(gas_feed)
            self.mfc   = ct.MassFlowController(self.inlet,  self.reactor, mdot=mdot_in)
            self.mfco  = ct.MassFlowController(self.reactor, self.outlet, mdot=mdot_in)

        self.sim = ct.ReactorNet([self.reactor])

    def run(self, t_end, dt=1e-5, dt_EN=1e-5):
        import cantera as ct
        t = 0.0
        states = ct.SolutionArray(self.reactor.thermo, extra=['t'])
        while t < t_end:
            # if self.EN_func: (keep your EN/EEDF updates here if needed)
            t = self.sim.step()
            states.append(self.reactor.thermo.state, t=t)

        """ EN_peak = 190 * 1e-21  # Td
        c0        = 24e-9             # first pulse center [s]
        period    = 1e-4              # 0.1 ms interpulse [s]
        sigma     = 3e-9              # Gaussian std dev [s]
        pre_win   = 30e-9             # fine window start before center [s]
        post_win  = 70e-9             # fine window end after center [s]

        def gaussian_about_center(t, center):
            return EN_peak * np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))

        # Next pulse center c_next >= t (starts at c0, then c0 + n*period)
        def next_pulse_center(t):
            if t <= c0:
                return c0
            n = np.ceil((t - c0) / period)
            return c0 + n * period

        t = 0.0
        dt_fine        = 1e-10
        dt_chunk_fine  = 1e-9
        dt_coarse      = 1e-5
        while t < t_end:
            c_next = next_pulse_center(t)
            fine_start = c_next - pre_win
            fine_end   = c_next + post_win
            print(t)

            if t < fine_start:
                # Interpulse region: coarse march to (c_next - 30 ns)
                t_chunk = min(t_end, fine_start)
                dt_use  = dt_coarse

                # EN ~ 0 far from pulse; skip EEDF work
                self.rgas.EN = 0.0
                # Reinitialize once for the upcoming coarse segment (piecewise-constant source)
                self.sim.reinitialize()

                # Integrate to t_chunk with big steps
                while self.sim.time < t_chunk:
                    target = min(t + dt_use, t_chunk)
                    self.sim.advance(target)
                    t = self.sim.time
                    states.append(self.reactor.thermo.state, t=t)

                # Move to next loop; EN/EEDF will be updated when we enter fine window
                continue

            # We are inside the fine window: resolve the Gaussian
            t_chunk = min(t_end, fine_end)
            dt_use  = dt_fine

            # Use small chunks (5 ns) with piecewise-constant EN per chunk
            while self.sim.time < t_chunk:
                # compute the current chunk end (<= 5 ns ahead, not past t_chunk)
                local_chunk_end = min(t + dt_chunk_fine, t_chunk)

                # march through this small chunk with small dt
                while self.sim.time < local_chunk_end:
                    target = min(t + dt_use, local_chunk_end)
                    self.sim.advance(target)
                    t = self.sim.time
                    states.append(self.reactor.thermo.state, t=t)

                # set EN for this chunk based on the *next pulse center*
                EN_now = gaussian_about_center(t, c_next)
                self.rgas.EN = EN_now
                self.rgas.update_EEDF()
                self.sim.reinitialize() """

        return states

class PFR(object):

    def __init__(self,gas):
        self.gas = gas
        self.MW  = gas.molecular_weights

    def __call__(self, t, y, ODEParams):

        # Secondary zone parameters:
            # SZ_A   :  Secondary zone area [m2]
            # SZ_mf  :  Total SZ_mf at the point of integration [kg/s]
            # beta   :  Mass flow of air per m [kg/s/m]
            # Y_in_a :  Mass fractions air [-]
            # h_a_in :  Specific enthalpy air [J/kg]
        #print('t,y_values = ',t,y[0], y[1])
        SZ_A   = ODEParams.SZ_A

        Y_in_a = ODEParams.Y_in_a
        h_in   = ODEParams.h_in
        P0     = ODEParams.P0
        x_dil_start = ODEParams.dil_start
        x_dil_end   = ODEParams.dil_end

        x_DZ_dil_start = ODEParams.DZ_dil_start

        ### Definitions

        self.gas.set_unnormalized_mass_fractions(y[2:])  # Set the mass fractions without normalizing to force sum(Y) == 1.0
        self.gas.TP = y[1], P0

        rho = self.gas.density                           # Gas density [kg/m^3]
        wdot = self.gas.net_production_rates             # Production rates of the species [kmol/m^3/s]
        M = y[0]                                         # Mass flow rate in reactor [kg/s]

        ### Equations of the secondary zone (Ideal Gas Plug Flow Reactor)

        # Conservation of mass
        # --------------------
        """ if (t >= x_dil_start) & (t < x_dil_end):
            beta = ODEParams.beta               # Massflow of air per m [kg/s/m]
        else:
            beta = 0                           # Massflow of air per m [kg/s/m]

        if (t>= x_DZ_dil_start):
            beta = ODEParams.beta_DZ """

        beta = 0
        if (t >= x_dil_start) & (t < x_dil_end):
            beta += ODEParams.beta               # Massflow of air per m [kg/s/m]
        if (t >= x_DZ_dil_start):
            beta += ODEParams.beta_DZ            # Massflow of air per m [kg/s/m]

        dMdz = beta

        # Conservation of species
        # -----------------------
        dYdz = (wdot*self.MW*SZ_A + beta*(Y_in_a-self.gas.Y))/M

        # Conservation of energy
        # ----------------------
        dTdz = 1/(self.gas.cp_mass*M)*(beta*(h_in - np.dot(Y_in_a, self.gas.partial_molar_enthalpies/self.MW)) \
        -SZ_A*np.dot(self.gas.partial_molar_enthalpies, wdot))

        ### Outputs
        return np.hstack((dMdz,dTdz,dYdz))

