import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

class PSR_Plasma:
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
        self.inlet = ct.Reservoir(gas)
        # Reactor
        self.reactor = ct.IdealGasReactor(gas, volume=V)
        # Outlet
        self.outlet = ct.Reservoir(gas)

        # Mass flow
        self.mfc = ct.MassFlowController(self.inlet, self.reactor, mdot=mdot_in)
        self.outlet_mfc = ct.MassFlowController(self.reactor, self.outlet, mdot=mdot_in)
        #self.valve = ct.PressureController(self.reactor, self.outlet, K=1e-5)

        # Reactor network
        self.sim = ct.ReactorNet([self.reactor])

    def run(self, t_end, dt=1e-5, dt_EN=1e-5):
        """
        Run the PSR to t_end [s], updating EN at every dt_EN.
        """
        t = 0.0
        states = ct.SolutionArray(self.gas, extra=['t'])

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

            #t += dt_EN

        return states

    def final_stream(self):
        q = ct.Quantity(self.reactor.thermo, constant='HP')
        q.mass = self.mdot_in
        return q

class PFR_Plasma:
    def __init__(self, gas, V_total, n_reactors, mdot_in, T0, P0, EN_func=None, debug=False, beta=0.0, beta_DZ=0.0,
                 dil_start=0.0, dil_end=0.0, DZ_dil_start=0.0,
                 Y_in_a=None, h_in=None, SZ_length=None):
        """
        PFR using a chain of IdealGasReactors with optional plasma actuation.

        gas: Cantera plasma Solution with EN support
        V_total: total PFR volume [m^3]
        n_reactors: number of reactor segments
        mdot_in: inlet mass flow [kg/s]
        T0: inlet temp [K]
        P0: inlet pressure [Pa]
        EN_func: function of time and axial position, EN(t, i)
        """
        self.gas = gas
        self.V_total = V_total
        self.n_reactors = n_reactors
        self.mdot_in = mdot_in
        self.T0 = T0
        self.P0 = P0
        self.EN_func = EN_func
        self.debug = debug
        self.beta = beta
        self.beta_DZ = beta_DZ
        self.dil_start = dil_start
        self.dil_end = dil_end
        self.DZ_dil_start = DZ_dil_start
        #self.Y_in_a = Y_in_a if Y_in_a else gas.Y
        self.Y_in_a = Y_in_a if Y_in_a is not None else gas.Y
        #self.h_in = h_in if h_in else gas.enthalpy_mass
        self.h_in = h_in if h_in is not None else gas.enthalpy_mass
        self.SZ_length = SZ_length
        self.gas_file = 'nDodecane_ReitzNO_plasma.yaml' #'gri30_plasma_cpavan.yaml'

        self.segment_volume = V_total / n_reactors

        # Inlet reservoir
        inlet_gas = ct.Solution(self.gas_file)
        inlet_gas.TPX = T0, P0, gas.X
        self.inlet = ct.Reservoir(inlet_gas)

        # Outlet reservoir
        self.outlet = ct.Reservoir(gas)

        # Create chain of reactors
        self.reactors = [ct.IdealGasReactor(gas, volume=self.segment_volume)
                         for _ in range(n_reactors)]

        # Create MFC chain: inlet -> reactor[0] -> reactor[1] -> ... -> outlet
        self.mfcs = []

        # Inlet to first reactor
        self.mfcs.append(ct.MassFlowController(self.inlet, self.reactors[0], mdot=mdot_in))

        # Link each reactor to the next
        for i in range(n_reactors - 1):
            self.mfcs.append(ct.MassFlowController(self.reactors[i], self.reactors[i+1], mdot=mdot_in))

        # Outlet
        self.mfcs.append(ct.MassFlowController(self.reactors[-1], self.outlet, mdot=mdot_in))

        # Side dilution jets here
        self.side_reservoirs = []
        self.side_mfcs = []

        self.side_inlets = []

        # If length is defined, convert continuous beta injection into segment-wise side jets
        if self.SZ_length and (self.beta > 0 or self.beta_DZ > 0):
            dz = self.SZ_length / self.n_reactors
            DZ_end = self.SZ_length

            # Compute total mass flow targets for each region
            total_dil_length = max(1e-12, self.dil_end - self.dil_start)
            total_DZ_length = max(1e-12, DZ_end - self.DZ_dil_start)

            target_dil_mdot = self.beta * total_dil_length
            target_DZ_mdot = self.beta_DZ * total_DZ_length

            # Compute per-segment overlap and normalized weights
            segment_mdot = []
            for i in range(self.n_reactors):
                seg_start = i * dz
                seg_end = seg_start + dz

                overlap_dil = max(0.0, min(self.dil_end, seg_end) - max(self.dil_start, seg_start))
                overlap_DZ = max(0.0, min(DZ_end, seg_end) - max(self.DZ_dil_start, seg_start))

                frac_dil = overlap_dil / total_dil_length
                frac_DZ = overlap_DZ / total_DZ_length

                mdot = target_dil_mdot * frac_dil + target_DZ_mdot * frac_DZ
                segment_mdot.append(mdot)

            for i, mdot in enumerate(segment_mdot):
                if mdot > 0:
                    self.side_inlets.append({
                        "segment_idx": i,
                        "mdot": mdot,
                        "composition": self.Y_in_a
                    })

            for jet in self.side_inlets:
                idx = jet["segment_idx"]
                mdot = jet["mdot"]
                comp = jet["composition"]
                print(f"[Side Jet] segment {idx} | mdot = {mdot:.4f} kg/s | beta = {self.beta:.2f} | beta_DZ = {self.beta_DZ:.2f}")

                # Create a separate reservoir for each side jet (air)
                air = ct.Solution(self.gas_file)
                air.TPY = self.T0, self.P0, comp
                res = ct.Reservoir(air)

                mfc = ct.MassFlowController(res, self.reactors[idx], mdot=mdot)

                self.side_reservoirs.append(res)
                self.side_mfcs.append(mfc)

        # Create ReactorNet
        #self.sim = ct.ReactorNet(self.reactors)

        # Compute geometry & axial velocity if L_total was given
        if self.SZ_length:
            rho_in = self.gas.density_mass  # kg/m^3 at inlet
            self.A_pfr = self.V_total / self.SZ_length
            self.u_axial = self.mdot_in / (rho_in * self.A_pfr)

            if self.debug:
                print(f"[PFR] SZ_length = {self.SZ_length:.4f} m | A_pfr = {self.A_pfr:.4e} m^2 | rho_in = {rho_in:.3f} kg/m^3")
                print(f"[PFR] Mean axial velocity u = {self.u_axial:.2f} m/s")

    def run(self, t_end, dt=1e-5, dt_EN=1e-5):
        """
        Run the PFR chain for t_end [s], updating EN at every dt_EN.
        """
        t = 0.0
        gas = ct.Solution(self.gas_file)
        gas.TPX = self.T0, self.P0, self.gas.X
        h = self.h_in

        states = ct.SolutionArray(self.gas, extra=['t', 'segment_idx'])

        for i, reactor in enumerate(self.reactors):
            # Set inlet state for this segment
            gas.HPX = h, self.P0, gas.X
            reactor.syncState()  # copy to current reactor

            # Apply EN function if available
            sim = ct.ReactorNet([reactor])
            t_local = 0.0

            while t_local < t_end:
                #if self.EN_func is not None:
                    #EN_now = self.EN_func(t + t_local, i)
                    #reactor.thermo.EN = EN_now
                    #reactor.thermo.update_EEDF()

                #sim.reinitialize()

                #sim.advance(t_local + dt)
                t_local = sim.step()  # use step to ensure consistent time advancement
                #t_local += dt_EN
                """ if t_local >= next_log_time:
                    states.append(reactor.thermo.state, t=t + t_local, segment_idx=i)
                    next_log_time += dt_log """

                states.append(reactor.thermo.state, t=t + t_local, segment_idx=i)

            # Prepare for next segment
            h = reactor.thermo.enthalpy_mass
            gas.X = reactor.thermo.X
            t += t_local

            if self.debug:
                print(f"[PFR Segment {i}] T_end = {reactor.T:.1f} K | t = {t:.3e} s")

        return states
        """ t = 0.0
        #states = [ct.SolutionArray(self.gas, extra=['t']) for _ in range(self.n_reactors)]
        states = ct.SolutionArray(self.gas, extra=['t', 'segment_idx'])

        while t < t_end:
            for i, r in enumerate(self.reactors):
                if self.EN_func is not None:
                    EN_now = self.EN_func(t, i)
                    r.thermo.EN = EN_now
                    r.thermo.update_EEDF()

            self.sim.reinitialize()  # rebuild with new rates

            self.sim.advance(t + dt)

            for i, r in enumerate(self.reactors):
                states.append(r.thermo.state, t=self.sim.time, segment_idx=i)

            if self.debug:
                print(f"t={self.sim.time:.3e} s | T_end={self.reactors[-1].T:.1f} K")

            t += dt_EN

        return states """

    def final_stream(self):
        total_mdot = self.mdot_in + sum(jet["mdot"] for jet in self.side_inlets)
        #q = ct.Quantity(self.reactors[-1].thermo, constant='HP')
        #q.mass = total_mdot
        gas = ct.Solution(self.gas_file)
        reactor = self.reactors[-1]
        h = reactor.thermo.enthalpy_mass
        X = reactor.thermo.X

        # Force consistent pressure for both SZ_inner and SZ_outer
        gas.HPX = h, self.P0, X

        q = ct.Quantity(gas, constant='HP')
        q.mass = total_mdot
        print(f"[Final Stream] mdot_in = {self.mdot_in:.5f}, side_inlet_total = {sum(j['mdot'] for j in self.side_inlets):.5f}")
        print(f"[Final Stream] Assigned streamQuantity.mass = {q.mass:.5f}")
        return q

# Load your plasma mechanism
""" gas = ct.Solution('gri30_plasma_cpavan.yaml')
gas.TPX = 300., 101325., 'CH4:0.1, O2:0.2, N2:0.7, e:1e-11'

# Example EN profile: Gaussian pulse
EN_peak = 190e-21
pulse_center = 24e-9
pulse_width = 3e-9
EN_func = lambda t: EN_peak * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))

psr = PSR_Plasma(gas, V=5e-6, mdot_in=1e-3, T0=300, P0=101325, EN_func=EN_func, debug=True)

result = psr.run(t_end=90e-9, dt=1e-10, dt_EN=1e-9)

gas = psr.gas  # reuse same Solution
states = result

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot electrons, ions, excited states
species1 = ['e', 'O2+', 'N2+', 'N2(A)', 'N2(B)', 'O']

for sp in species1:
    if sp in gas.species_names:
        ax[0].plot(states.t, states.X[:, gas.species_index(sp)], label=sp)

ax[0].set_yscale('log')
ax[0].set_ylim([1e-14, 1e-3])
ax[0].set_ylabel('Mole fraction [-]')
ax[0].legend()
ax[0].set_title('Plasma Species')

# Plot major species and T
species2 = ['CH4', 'O2', 'CO', 'CO2', 'OH', 'H']

for sp in species2:
    if sp in gas.species_names:
        ax[1].plot(states.t, states.X[:, gas.species_index(sp)], label=sp)

axT = ax[1].twinx()
axT.plot(states.t, states.T, 'k--', label='T [K]')
axT.set_ylabel('T [K]')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Mole fraction [-]')
ax[1].legend(loc='upper left')
axT.legend(loc='upper right')
ax[1].set_title('Main Species and Temperature')

plt.tight_layout()
plt.show() """

""" gas = ct.Solution('gri30_plasma_cpavan.yaml')
gas.TPX = 300., 101325., 'CH4:0.1, O2:0.2, N2:0.7, e:1e-11'

# EN function: same pulse for every segment
EN_peak = 190e-21
pulse_center = 24e-9
pulse_width = 3e-9
EN_func = lambda t, i: EN_peak * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))

pfr = PFR_Plasma(gas, V_total=1e-4, n_reactors=10, mdot_in=1e-3,
                 T0=300, P0=101325, EN_func=EN_func, debug=True)

result = pfr.run(t_end=90e-9, dt=1e-10, dt_EN=1e-9)

gas = pfr.gas  # reuse same Solution
states = result

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot electrons, ions, excited states
species1 = ['e', 'O2+', 'N2+', 'N2(A)', 'N2(B)', 'O']

for sp in species1:
    if sp in gas.species_names:
        for i, seg in enumerate(states):
            ax[0].plot(seg.t, seg.X[:, gas.species_index(sp)], label=f'{sp} - seg {i}')

ax[0].set_yscale('log')
ax[0].set_ylim([1e-14, 1e-3])
ax[0].set_ylabel('Mole fraction [-]')
ax[0].legend()
ax[0].set_title('Plasma Species')

# Plot major species and T
species2 = ['CH4', 'O2', 'CO', 'CO2', 'OH', 'H']

for sp in species2:
    if sp in gas.species_names:
        for i, seg in enumerate(states):
            ax[1].plot(seg.t, seg.X[:, gas.species_index(sp)], label=f'{sp} - seg {i}')


ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Mole fraction [-]')
ax[1].legend(loc='upper left')
ax[1].set_title('Main Species and Temperature')

plt.tight_layout()
plt.show()
 """