import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

class PSR_Plasma:
    def __init__(self, gas, V, mdot_in, T0, P0, EN_func, debug=False):
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

    def run(self, t_end, dt=1e-9, dt_EN=1e-9):
        """
        Run the PSR to t_end [s], updating EN at every dt_EN.
        """
        t = 0.0
        states = ct.SolutionArray(self.gas, extra=['t'])

        while t < t_end:
            EN_now = self.EN_func(t)
            self.gas.EN = EN_now
            self.gas.update_EEDF()
            self.sim.reinitialize()

            # advance by small timestep
            self.sim.advance(t + dt)
            states.append(self.reactor.thermo.state, t=self.sim.time)

            if self.debug:
                print(f"t={self.sim.time:.3e} s | T={self.reactor.T:.1f} K | EN={EN_now:.3e} Td")

            t += dt_EN

        return states

class PFR_Plasma:
    def __init__(self, gas, V_total, n_reactors, mdot_in, T0, P0, EN_func=None, debug=False):
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

        self.segment_volume = V_total / n_reactors

        # Inlet reservoir
        inlet_gas = gas
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

        # Create ReactorNet
        self.sim = ct.ReactorNet(self.reactors)

    def run(self, t_end, dt=1e-9, dt_EN=1e-9):
        """
        Run the PFR chain for t_end [s], updating EN at every dt_EN.
        """
        t = 0.0
        states = [ct.SolutionArray(self.gas, extra=['t']) for _ in range(self.n_reactors)]

        while t < t_end:
            for i, r in enumerate(self.reactors):
                if self.EN_func is not None:
                    EN_now = self.EN_func(t, i)
                    r.thermo.EN = EN_now
                    r.thermo.update_EEDF()

            self.sim.reinitialize()  # rebuild with new rates

            self.sim.advance(t + dt)

            for i, r in enumerate(self.reactors):
                states[i].append(r.thermo.state, t=self.sim.time)

            if self.debug:
                print(f"t={self.sim.time:.3e} s | T_end={self.reactors[-1].T:.1f} K")

            t += dt_EN

        return states

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
plt.show() """
