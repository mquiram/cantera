import numpy as np
import cantera as ct
import CombUtils
from plasmaReactors2 import PSR_Plasma
from plasmaReactors2 import PFR_Plasma
import time
from functools import reduce

class PlasmaParameters:
    """
    The PlasmaParameters class stores relevant input parameters to the combustor model,
    structured identically to the original architecture for compatibility.
    """

    debug = False
    # Chemical Properties
    RM = 'gri30_plasma_cpavan.yaml'             # Reaction mechanism [-]
    fuelStr = 'CH4:1'            # Fuel string [-]
    Oxidizer = 'O2:1.0, N2:3.76' # Oxidizer string [-]
    FA_st = 0.0581465            # Stoichiometric fuel-air ratio [-]
    LHV = 50748183.2665071       # Lower heating value [J/kg]

    # Combustor geometric properties
    PZ_volume = 0.005            # Volume of PZ [m^3]
    SZ_volume = 0.00025          # Volume of SZ [m^3]
    SZ_length = 0.20             # Secondary zone length [m]
    frac_dil_length = 0.1        # Ratio of dilution zone length to SZ length [-]

    # Simulation settings
    PZ_tsim = 100e-3 / 4         # Simulation time for PZ reactor [s]
    SZ_dz = 1e-3                 # Step size in SZ [m]
    SZ_dilution = 1             # Whether to model SZ dilution air [1:yes, 0:no]
    PZ_n_reactor = 1             # Number of PZ reactors
    PZ_k_pressure = 1           # Pressure controller constant in PSR

    def __init__(self):
        self.mdot_air  = 19      # Total air mass flow rate [kg/s]
        self.mdot_fuel = 0.40    # Total fuel mass flow rate [kg/s]

    def calcGeom(self):
        """Calculate SZ cross-sectional area and total combustor volume."""
        self.V_combustor = self.PZ_volume + self.SZ_volume
        self.SZ_A = self.SZ_volume / self.SZ_length

    def des_splitMassFlow(self, PZ_desPhi, SZ_desPhi):
        """Split air mass flow based on PZ and SZ design equivalence ratios."""
        self.PZ_desPhi = PZ_desPhi
        self.PZ_phi_mean = PZ_desPhi
        self.PZ_phi_sigma = 2.16 * np.exp(-2 * self.PZ_phi_mean)

        self.PZ_mdot_air = self.mdot_fuel / (PZ_desPhi * self.FA_st)
        self.PZ_airfrac = self.PZ_mdot_air / self.mdot_air
        self.PZ_mdot_in = self.PZ_mdot_air + self.mdot_fuel

        self.SZ_desPhi = SZ_desPhi
        self.SZ_airfrac = (self.mdot_fuel / self.mdot_air) / (SZ_desPhi * self.FA_st)
        self.SZ_mdot_air = self.mdot_air * self.SZ_airfrac

        self.DZ_airfrac = 1 - (self.SZ_airfrac + self.PZ_airfrac)
        self.DZ_mdot_air = self.mdot_air * self.DZ_airfrac

        if self.DZ_mdot_air < 0:
            print("WARNING NO DILUTION AIR: DZ_mdot_air =", self.DZ_mdot_air)
            self.SZ_mdot_air = self.mdot_air - self.PZ_mdot_air
            self.SZ_airfrac = self.SZ_mdot_air / self.mdot_air
            self.DZ_mdot_air = 0
            self.DZ_airfrac = 0
            print(f"\tSZ mdot air set to {self.SZ_mdot_air:.2f},\n"
                  f"\tSZ_airfrac = {self.SZ_airfrac:.5f}\n\t..and setting 0.0 DZ air")

        print(f"Design point values:\n\tDes. PZ_airfrac = {self.PZ_airfrac}\n"
              f"\tDes. SZ_airfrac = {self.SZ_airfrac}\n\tDes. DZ_airfrac = {self.DZ_airfrac}")

    def splitMassFlow(self):
        """Split total air mass flow using precomputed fractions."""
        self.PZ_mdot_air = self.mdot_air * self.PZ_airfrac
        self.PZ_phi_mean = (self.mdot_fuel / self.PZ_mdot_air) / self.FA_st
        self.PZ_phi_sigma = 2.16 * np.exp(-2 * self.PZ_phi_mean)
        self.PZ_mdot_in = self.PZ_mdot_air + self.mdot_fuel

        self.SZ_mdot_air = self.mdot_air * self.SZ_airfrac
        self.DZ_mdot_air = self.mdot_air * self.DZ_airfrac

def CombustorPZ(Params, gas):
    """
    Run Cantera-based PSRs in parallel for the Primary Zone (PZ).
    Replaces ODE integration with native Cantera reactor tools.
    """
    PZ_start_time = time.time()

    # Unpack parameters
    PZ_volume       = Params.PZ_volume
    PZ_tsim         = Params.PZ_tsim
    P0              = Params.P0
    T0              = Params.T0
    PZ_n_reactor    = Params.PZ_n_reactor
    PZ_k_pressure   = Params.PZ_k_pressure

    # Split PZ into parallel reactors and calculate flow splits
    splitReactor(Params)

    # Initialize output arrays
    Params.PZ_states = []
    Params.PZ_exitStream = []
    Params.PZ_tres_lst = []
    streamQuantityList = []

    for i in range(PZ_n_reactor):
        # Create a new gas object for each reactor
        gas_i = ct.Solution(Params.RM)
        gas_i.TP = T0, P0

        # Set initial composition for this reactor
        gas_i.set_equivalence_ratio(Params.PZ_phi_lst[i], Params.fuelStr, Params.Oxidizer)

        # Create the PSR reactor with pressure control
        reactor = PSR_Plasma(gas_i,
                      V=Params.PZ_V_lst[i],
                      mdot_in=Params.PZ_mdot_in_lst[i],
                      T0=T0,
                      P0=P0,
                      EN_func=lambda t: 0.0,  # No external energy input
                      #k=Params.PZ_k_pressure,
                      #name=f"PZ{i+1}"
                      )

        # Integrate until steady state
        state = reactor.run(t_end=PZ_tsim, dt=1e-5, dt_EN=1e-5)  # Can adapt dt if needed

        # Store results
        Params.PZ_states.append(state)
        Params.PZ_exitStream.append(reactor.reactor.thermo)
        gas_i.HPX = reactor.reactor.thermo.enthalpy_mass, P0, reactor.reactor.thermo.X
        streamQuantity = ct.Quantity(gas_i, constant='HP')
        streamQuantity.mass = Params.PZ_mdot_in_lst[i]
        streamQuantityList.append(streamQuantity)
        tres = Params.PZ_V_lst[i] / Params.PZ_mdot_in_lst[i]
        Params.PZ_tres_lst.append(tres)

        if Params.debug:
            print(f"[PZ {i+1}] T = {state.T:.1f} K | phi = {Params.PZ_phi_lst[i]:.3f} | t_res = {tres:.4f} s")

    print("Primary zone simulation complete. Elapsed time: {:.3f} s".format(time.time() - PZ_start_time))

    PZ_exitStream = reduce(lambda a, b: a + b, streamQuantityList)
    Params.PZ_exitStream = PZ_exitStream
    gas = PZ_exitStream.phase

    NOx = 1000 * (gas.Y[gas.species_index('NO')] * 46 / 30 +
                  gas.Y[gas.species_index('NO2')]) * PZ_exitStream.mass
    CO = 1000 * gas.Y[gas.species_index('CO')] * PZ_exitStream.mass

    if Params.debug:
        print(f"PZ exit: T = {gas.T:.3f}, P = {gas.P:.3f}")
        print(f"PZ exit NOx = {NOx:.3e}")
        print(f"PZ exit CO  = {CO:.3e}")
        print("\n\n ------------------------------------------------------------ ")

    PZ_results = [gas.T, gas.P, gas.density, NOx, CO, gas.X, gas.cp_mass - gas.cv_mass]

    return PZ_results

def CombustorSZ(Params, gas, air):

    SZ_start_time = time.time()

    def SZ_subroutine(mdot_in, A, air, mdot_air, frac_dil_length_i, mdot_DZ):
        print(f"\n--- SZ Subroutine Debug ---")
        print(f"mdot_in = {mdot_in:.5f} kg/s")
        print(f"A = {A:.5f} m²")
        print(f"mdot_air (dilution) = {mdot_air:.5f} kg/s")
        print(f"frac_dil_length_i = {frac_dil_length_i:.5f}")
        print(f"mdot_DZ = {mdot_DZ:.5f} kg/s")
        print(f"-----------------------------\n")
        gas = Params.PZ_exitStream.phase  # Reset to exit state of PZ
        P0 = Params.P0
        SZ_length = Params.SZ_length
        dx_SZ = SZ_length * Params.SZ_dz

        # Set up dilution parameters
        if Params.SZ_dilution == 1:
            beta = mdot_air / (frac_dil_length_i * SZ_length)
        else:
            beta = 0.0

        beta_DZ = mdot_DZ / (Params.DZ_frac_dil_length * SZ_length)
        print(f"beta = {beta:.5f}, beta_DZ = {beta_DZ:.5f}")

        x_dil_start = Params.dil_start * SZ_length
        x_dil_end = x_dil_start + frac_dil_length_i * SZ_length
        x_DZ_start = SZ_length * (1 - Params.DZ_frac_dil_length)
        print(f"x_dil_start = {x_dil_start:.5f} m, x_dil_end = {x_dil_end:.5f} m, x_DZ_start = {x_DZ_start:.5f} m")

        # Configure inlet air properties
        Y_in_a = air.Y
        h_in_a = air.enthalpy_mass

        # Run SZ_PFR reactor model

        #propertySolver_stats = reactornet.solverStats
        n_seg = max(1, int(SZ_length / dx_SZ))
        print("SZ length:", SZ_length)
        print("dx_SZ:", dx_SZ)
        print("n_seg ",n_seg)
        n_seg = 10 #3
        pfr = PFR_Plasma(
            gas,
            V_total=A * SZ_length,
            n_reactors=n_seg,
            mdot_in=mdot_in,
            #A=A,
            T0=Params.T0,
            P0=P0,
            EN_func=lambda t, i: 0.0,
            debug=Params.debug,
            #length=SZ_length,
            #dx=dx_SZ,
            beta=beta,
            beta_DZ=beta_DZ,
            dil_start=x_dil_start,
            dil_end=x_dil_end,
            DZ_dil_start=x_DZ_start,
            Y_in_a=Y_in_a,
            h_in=h_in_a,
            SZ_length=SZ_length
        )

        rho = gas.density
        u_axial = mdot_in / (rho * A)
        t_end1 = SZ_length / u_axial
        states_SZ = pfr.run(t_end=t_end1, dt=1e-5, dt_EN=1e-5)
        streamQuantity = pfr.final_stream()

        # Optional debug output
        #if Params.debug:
        CO_in = 1000 * states_SZ.Y[0, states_SZ.species_index('CO')] * mdot_in
        CO_out = 1000 * states_SZ.Y[-1, states_SZ.species_index('CO')] * streamQuantity.mass
        NOx_in = 1000 * (states_SZ.Y[0, states_SZ.species_index('NO')]*46/30 + states_SZ.Y[0, states_SZ.species_index('NO2')]) * mdot_in
        NOx_out = 1000 * (states_SZ.Y[-1, states_SZ.species_index('NO')]*46/30 + states_SZ.Y[-1, states_SZ.species_index('NO2')]) * streamQuantity.mass
        print('--------------------------- CO & NOx ------------------------------------')
        print(f'CO at start of SZ  = {CO_in:.2E}\tCO  produced  = {CO_out - CO_in:.2E}\tCO end  = {CO_out:.3f}')
        print(f'NOx at start of SZ = {NOx_in:.2E}\tNOx produced  = {NOx_out - NOx_in:.2E}\tNOx end = {NOx_out:.3f}')
        print(f"Initial mdot_in: {mdot_in:.3f}")
        print(f"Expected final mass flow: {mdot_in + mdot_air + mdot_DZ:.3f}")
        print(f"Computed final mass flow: {streamQuantity.mass:.3f}")
        print('-------------------------------------------------------------------------')

        return streamQuantity

    print(f"[Outer SZ] mdot_frac = {Params.SZ_mdot_frac_outer:.5f}, A_frac = {Params.SZ_A_frac_outer:.5f}, "
        f"frac_dil_length = {Params.frac_dil_length_outer:.5f}")
    print(f"[Inner SZ] mdot_frac = {Params.SZ_mdot_frac_inner:.5f}, A_frac = {Params.SZ_A_frac_inner:.5f}, "
        f"frac_dil_length = {Params.frac_dil_length_inner:.5f}")

    # Compute contributions from inner and outer SZ subdomains
    SZ_outer = SZ_subroutine(
        Params.PZ_mdot_in * Params.SZ_mdot_frac_outer,
        Params.SZ_A * Params.SZ_A_frac_outer,
        air,  # Clone to avoid modifying original air object
        Params.SZ_mdot_air / 2,
        Params.frac_dil_length_outer,
        Params.DZ_mdot_air
    )

    SZ_inner = SZ_subroutine(
        Params.PZ_mdot_in * Params.SZ_mdot_frac_inner,
        Params.SZ_A * Params.SZ_A_frac_inner,
        air,  # Clone to avoid modifying original air object
        Params.SZ_mdot_air / 2,
        Params.frac_dil_length_inner,
        0.0  # Inner SZ has no DZ air
    )

    print("Secondary zone simulation complete. Elapsed time: {:.3f} s".format(time.time() - SZ_start_time))

    # Total exit stream from secondary zone
    SZ_exitStream = SZ_outer + SZ_inner
    if Params.debug:
        print(f"Total SZ mass exit = {SZ_exitStream.mass:.3f} kg/s | Expected = {(Params.mdot_air + Params.mdot_fuel):.3f} kg/s")

    return SZ_exitStream

Params = PlasmaParameters()

def splitReactor(Params):
    """Splits the primary zone into multiple reactors and calculates corresponding fuel and air mass flow rates."""

    def NormalDistribution(mu, sigma, x):
        return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if Params.PZ_n_reactor == 1:
        Params.PZ_phi_lst        = np.array([Params.PZ_phi_mean])
        Params.PZ_mdot_in_lst    = np.array([Params.PZ_mdot_in])
        Params.PZ_V_lst          = np.array([Params.PZ_volume])
        Params.PZ_mdot_fuel_lst  = Params.PZ_phi_lst * Params.FA_st * Params.PZ_mdot_air

    else:
        mdot_air = Params.PZ_mdot_air

        # Redefine sigma based on phi mean (if desired)
        Params.PZ_phi_sigma = 0.37 * Params.PZ_phi_mean  # or keep preset if already done

        # Set phi bounds
        phi_min = 0.025 / Params.FA_st
        phi_max = 4.0
        x1 = abs(Params.PZ_phi_mean - phi_min) / Params.PZ_phi_sigma
        x2 = abs(phi_max - Params.PZ_phi_mean) / Params.PZ_phi_sigma
        x = np.min([x1, x2, 3])

        # Linearly spaced phi values for each reactor
        Params.PZ_phi_lst = np.linspace(
            Params.PZ_phi_mean - x * Params.PZ_phi_sigma,
            Params.PZ_phi_mean + x * Params.PZ_phi_sigma,
            Params.PZ_n_reactor
        )

        # Use Gaussian distribution to determine air split
        fraction_lst = NormalDistribution(Params.PZ_phi_mean, Params.PZ_phi_sigma, Params.PZ_phi_lst)
        fraction_lst /= np.sum(fraction_lst)

        mdot_air_lst   = mdot_air * fraction_lst
        mdot_fuel_lst  = Params.PZ_phi_lst * Params.FA_st * mdot_air_lst
        Params.PZ_mdot_fuel_lst = mdot_fuel_lst

        if Params.debug:
            print("Fuel flows  =", sum(mdot_fuel_lst), "(input:", Params.mdot_fuel, ")")
            print("Air flows   =", sum(mdot_air_lst), "(input:", mdot_air, ")")
            print("Air split   =", mdot_air_lst)
            print("Fuel split  =", mdot_fuel_lst)
            print("Phi values  =", mdot_fuel_lst / mdot_air_lst / Params.FA_st)

        # Consistency check
        if abs(sum(mdot_fuel_lst) - Params.mdot_fuel) > 1e-10:
            print("Error in mass of fuel = ", sum(mdot_fuel_lst) - Params.mdot_fuel)

        # Equal volume distribution
        Params.PZ_V_lst = (Params.PZ_volume / Params.PZ_n_reactor) * np.ones(Params.PZ_n_reactor)

        # Total gas mass flow per reactor
        Params.PZ_mdot_in_lst = mdot_air_lst + mdot_fuel_lst

    if Params.debug:
        print('Primary zone phi dist    =', Params.PZ_phi_lst)
        print('Primary zone volume dist =', Params.PZ_V_lst)

def setChem(RM, Fuel, Oxidizer):
    """
    Initializes Cantera gas objects for the main gas and dilution air using the specified
    reaction mechanism and fuel/oxidizer strings. Stores results in the Params object.

    Inputs
    ------
    RM       : Reaction mechanism file (e.g. 'gri30.cti')
    Fuel     : Fuel composition string (e.g. 'CH4:1')
    Oxidizer : Oxidizer composition string (e.g. 'O2:1.0, N2:3.76')
    Params   : Instance of PlasmaParameters class
    """

    Params.Tref = 298.15    # Reference temperature [K]
    Params.Pref = 101325.0  # Reference pressure [Pa]

    Params.RM = RM
    Params.fuelStr  = Fuel
    Params.Oxidizer = Oxidizer

    # Compute stoichiometric fuel-air ratio and LHV from utilities
    Params.FA_st = CombUtils.calc_Stoichiometric_FAR(RM, Fuel, Oxidizer)
    Params.LHV   = CombUtils.calcLHV(RM, Fuel, Oxidizer)

    # Create Cantera Solution objects without transport properties
    Params.gas     = ct.Solution(RM, transport_model='None')
    Params.dil_air = ct.Solution(RM, transport_model='None')

    # Set standard enthalpy of the fuel
    Params.gas.TPX = Params.Tref, Params.Pref, Fuel + ":1"
    Params.fuel_std_enthalpy_mass = Params.gas.enthalpy_mass

### ----------------------------------------------------------------------
### Define the setInletFlow function
### ----------------------------------------------------------------------
def setInletFlow(mdot_air, mdot_fuel, Pt3, Tt3):
    """
    Sets inlet mass flow and thermodynamic state for air and fuel streams.

    Inputs
    -------
    mdot_air  : core air mass flow [kg/s]
    mdot_fuel : baseline fuel mass flow [kg/s]
    Pt3       : Total pressure at combustor inlet [Pa]
    Tt3       : Total temperature at combustor inlet [K]
    """
    Params.mdot_air  = mdot_air

    # Scale fuel flow to maintain same heating power as NPSS baseline
    LHV_NPSS = 43.5e6  # [J/kg]
    Params.fuel_scaler = (LHV_NPSS / Params.LHV)
    Params.mdot_fuel = mdot_fuel * Params.fuel_scaler
    Params.fuel_deficit = mdot_fuel * (1 - Params.fuel_scaler)

    Params.P0 = Pt3          # Static pressure for Cantera inlet state [Pa]
    Params.T0 = Tt3          # Static temperature for Cantera inlet state [K]
    Params.T_air = Tt3       # Assume same T for air inlet
    Params.P_air = Pt3       # Assume same P for air inlet

### ----------------------------------------------------------------------
### Define the setGeometry function
### ----------------------------------------------------------------------
def setGeometry(PZ_volume, SZ_volume, SZ_length):
    """
    Sets combustor volume and secondary zone geometry.

    Inputs
    -------
    PZ_volume : Primary zone volume [m^3]
    SZ_volume : Secondary zone volume [m^3]
    SZ_length : Secondary zone length [m]
    """
    Params.PZ_volume = PZ_volume
    Params.SZ_volume = SZ_volume
    Params.SZ_length = SZ_length

def des_setup(PZ_desPhi, PZ_n_reactor,
              SZ_desPhi, SZ_dilution, dil_start,
              frac_dil_length_outer, frac_dil_length_inner,
              SZ_mdot_frac_outer, SZ_A_frac_outer,
              DZ_frac_dil_length = 0.5, #0.05,
              debug = False):
    """
    Sets up the combustor at design point based on equivalence ratios and dilution configuration.

    Inputs
    -------
    PZ_desPhi              : Mean equivalence ratio in PZ
    PZ_n_reactor           : Number of reactors to discretize PZ
    SZ_desPhi              : Secondary zone equivalence ratio
    SZ_dilution            : Enable SZ dilution (1 = yes, 0 = no)
    dil_start              : Start of SZ dilution region (fraction of SZ length)
    frac_dil_length_outer  : Fractional length of outer dilution
    frac_dil_length_inner  : Fractional length of inner dilution
    SZ_mdot_frac_outer     : Fraction of SZ air for outer injection
    SZ_A_frac_outer        : Fraction of SZ area for outer injection
    DZ_frac_dil_length     : Fractional length for final DZ dilution (default 0.05)
    debug                  : Enable debug printouts
    """

    Params.debug = debug
    Params.PZ_n_reactor = PZ_n_reactor
    Params.DZ_frac_dil_length = DZ_frac_dil_length

    # Split air/fuel based on design point Φ
    Params.des_splitMassFlow(PZ_desPhi, SZ_desPhi)
    print(f"[Air Split Check] SZ_airfrac = {Params.SZ_airfrac:.5f}, "
      f"DZ_airfrac = {Params.DZ_airfrac:.5f}, "
      f"PZ_airfrac = {Params.PZ_airfrac:.5f}")

    # Update geometry and dilution config
    Params.calcGeom()
    Params.SZ_dilution = SZ_dilution

    Params.frac_dil_length_outer = frac_dil_length_outer
    Params.frac_dil_length_inner = frac_dil_length_inner

    Params.SZ_mdot_frac_outer = SZ_mdot_frac_outer
    Params.SZ_mdot_frac_inner = 1.0 - SZ_mdot_frac_outer

    Params.SZ_A_frac_outer = SZ_A_frac_outer
    Params.SZ_A_frac_inner = 1.0 - SZ_A_frac_outer

    Params.dil_start = dil_start

def setup(debug=False, PZ_airfrac=None, SZ_airfrac=None, DZ_airfrac=None):
    """
    Allows manual air mass split configuration (off-design) if not using design-point setup.

    Inputs
    -------
    debug        : Enable debug output
    PZ_airfrac   : Fraction of air to PZ (optional)
    SZ_airfrac   : Fraction of air to SZ (optional)
    DZ_airfrac   : Fraction of air to DZ (optional)
    """
    if PZ_airfrac is not None:
        print("\nSetting Airfracs based on provided inputs\n")
        Params.PZ_airfrac = PZ_airfrac
        Params.SZ_airfrac = SZ_airfrac
        Params.DZ_airfrac = DZ_airfrac

    # Check for required design-point variables
    try:
        Params.PZ_airfrac
    except AttributeError:
        raise RuntimeError("MUST SETUP DESIGN POINT BEFORE OFF-DESIGN CALCULATIONS! "
                           "\nUse des_setup(...) to configure first.")

    # Proceed with flow split and geometry update
    Params.debug = debug
    Params.splitMassFlow()
    Params.calcGeom()

def run(printResults):
    if printResults == 1:
        CombUtils.print_run_conditions()

    # Initialize diluent air (secondary zone air)
    Params.dil_air.TPX = Params.T_air, Params.P_air, Params.Oxidizer

    # --- Primary Zone ---
    PZ_results = CombustorPZ(Params, Params.gas)

    # Optional use of PZ output to update parameters (e.g., thermal diffusivity or flame radius scaling)
    R = PZ_results[-1]  # This could be cp_mass - cv_mass, or used to estimate diffusivity

    # --- Secondary Zone ---
    SZ_exitStream = CombustorSZ(Params, Params.gas, Params.dil_air)

    # --- Emissions Calculation ---
    NOx = 1000 * (
        SZ_exitStream.Y[SZ_exitStream.species_index('NO')] * 46 / 30 +
        SZ_exitStream.Y[SZ_exitStream.species_index('NO2')]
    ) * SZ_exitStream.mass

    CO = 1000 * SZ_exitStream.Y[SZ_exitStream.species_index('CO')] * SZ_exitStream.mass

    # Emission Indices (mass-based emissions per unit fuel mass)
    EI_NOx = NOx / (Params.mdot_fuel / Params.fuel_scaler)
    EI_CO  = CO / (Params.mdot_fuel / Params.fuel_scaler)

    if printResults == 1:
        CombUtils.print_results()

    return Params, PZ_results, SZ_exitStream, EI_NOx, EI_CO, NOx, CO