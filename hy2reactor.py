import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

def run_psr_constP(
    gas_feed,          # Cantera Solution defining INLET mixture (Y_in, h_in, P)
    gas_ic,            # initial reactor contents (HP-equilibrated copy)
    V,                 # reactor volume [m^3]
    mdot_in,           # inlet mass flow [kg/s]
    P0,                # operating pressure [Pa]
    t_end,             # integration time [s]
    dt=1e-5,           # time step cap [s]
    debug=False
):
    """
    Constant-pressure PSR built from Cantera primitives.
    - Inlet reservoir: composition = gas_feed (premixed), enthalpy set via HP
    - Reactor IC: gas_ic (post HP-equilibration of same H,P)
    - Outlet: same mdot as inlet to hold mass nearly constant
    - Integrates to t_end and returns (thermo, reactor)
    """
    # Clone feed for inlet reservoir, ensure (h_in, P0) is exactly imposed
    mech = getattr(gas_feed, "source", None) or getattr(gas_feed, "name", None)
    inlet_gas = ct.Solution(mech, transport_model='None')
    inlet_gas.TPX = gas_feed.T, P0, gas_feed.X
    inlet_gas.HP = gas_feed.enthalpy_mass, P0  # exact h_in, P0

    inlet = ct.Reservoir(inlet_gas)

    # Environment reservoir at same P (composition irrelevant for const-P reactor)
    env_gas = ct.Solution(mech, transport_model='None')
    env_gas.TPX = gas_feed.T, P0, gas_feed.X
    environment = ct.Reservoir(env_gas)

    # Reactor at constant pressure with IC = equilibrated state at (H,P)
    reactor = ct.IdealGasConstPressureReactor(gas_ic, energy='on', volume=V)

    # Mass flow controllers (PSR assumption: mdot_out = mdot_in)
    mfc_in  = ct.MassFlowController(inlet,   reactor, mdot=mdot_in)
    mfc_out = ct.MassFlowController(reactor, environment, mdot=mdot_in)

    net = ct.ReactorNet([reactor])

    t = 0.0
    while t < t_end:
        t_next = min(t + dt, t_end)
        net.advance(t_next)
        t = t_next
        if debug and int(t / dt) % 100 == 0:
            print(f"[PSR] t={t:.4e}s  T={reactor.T:.1f}K  P={reactor.thermo.P/1e5:.3f}bar")

    return reactor.thermo, reactor

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
        if (t >= x_dil_start) & (t < x_dil_end):
            beta = ODEParams.beta               # Massflow of air per m [kg/s/m]
        else:
            beta = 0                           # Massflow of air per m [kg/s/m]

        if (t>= x_DZ_dil_start):
            beta = ODEParams.beta_DZ

        """ beta = 0
        if (t >= x_dil_start) & (t < x_dil_end):
            beta += ODEParams.beta               # Massflow of air per m [kg/s/m]
        if (t >= x_DZ_dil_start):
            beta += ODEParams.beta_DZ            # Massflow of air per m [kg/s/m] """

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

