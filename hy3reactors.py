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
    mults = [9.0, 6.5, 5.5, 7.0, 5.0, 5.5, 7.5]
    EN_peak = 190 * 1e-21  # Td
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
    gas_ic.EN = gaussian_about_center(0, c0)
    gas_ic.update_EEDF()
    reactor = ct.IdealGasConstPressureReactor(gas_ic, energy='on', volume=V)

    # Mass flow controllers (PSR assumption: mdot_out = mdot_in)
    mfc_in  = ct.MassFlowController(inlet,   reactor, mdot=mdot_in)
    mfc_out = ct.MassFlowController(reactor, environment, mdot=mdot_in)

    sim = ct.ReactorNet([reactor])

    """ S = (EN_peak/(190 * 1e-21))**0.5
    print(S)
    S = max(0.5, min(2.0, S))  # clip
    print(S)

    for i in range(7):
        M_final = 1.0 + (mults[i] - 1.0) * S  # or just mults[i] if no scaling
        gas_ic.set_multiplier(M_final, i)
        print(f"Applied M={M_final:.3g}") """

    t = 0.0
    dt_fine        = 1e-10
    dt_chunk_fine  = 1e-9
    dt_coarse      = 1e-5
    while t < t_end:
        c_next = next_pulse_center(t)
        fine_start = c_next - pre_win
        fine_end   = c_next + post_win

        if t < fine_start:
            # Interpulse region: coarse march to (c_next - 30 ns)
            t_chunk = min(t_end, fine_start)
            dt_use  = dt_coarse

            # EN ~ 0 far from pulse; skip EEDF work
            gas_ic.EN = 0.0
            # Reinitialize once for the upcoming coarse segment (piecewise-constant source)
            sim.reinitialize()

            # Integrate to t_chunk with big steps
            while sim.time < t_chunk:
                target = min(t + dt_use, t_chunk)
                sim.advance(target)
                t = sim.time

            # Move to next loop; EN/EEDF will be updated when we enter fine window
            continue

        # We are inside the fine window: resolve the Gaussian
        t_chunk = min(t_end, fine_end)
        dt_use  = dt_fine

        # Use small chunks (5 ns) with piecewise-constant EN per chunk
        while sim.time < t_chunk:
            # compute the current chunk end (<= 5 ns ahead, not past t_chunk)
            local_chunk_end = min(t + dt_chunk_fine, t_chunk)

            # march through this small chunk with small dt
            while sim.time < local_chunk_end:
                target = min(t + dt_use, local_chunk_end)
                sim.advance(target)
                t = sim.time

            # set EN for this chunk based on the *next pulse center*
            EN_now = gaussian_about_center(t, c_next)
            gas_ic.EN = EN_now
            gas_ic.update_EEDF()
            sim.reinitialize()

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