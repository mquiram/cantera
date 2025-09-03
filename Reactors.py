import numpy as np
import cantera as ct
from math import *
import cantera

class PSR(object): 
    
    ### Description: Parameters of the ODE system and auxiliary data are stored in the ReactorOde object.
    
    def __init__(self,gas):
        self.gas = gas
        self.MW  = gas.molecular_weights
        
    def __call__(self, t, y, ODEParams):
        
        # t:           Time [s]
        # y:           Output values: {mass, temperature, gas mass fractions} [kg,K,-]
        
        # Primary zone parameters:
            # Y_in:    Incoming mass fractions primary zone [-]
            # mdot_in: Incoming gas mass flow primary zone [kg/s]
            # h_in:    Incoming gas enthalpy primary zone [J/kg]
            # V:       Primary zone volume [m^3]
            # P0:      Reference pressure [Pa]
            # k:       Pressure reactor constant [-]

         ### Unpack parameters
        Y_in    = ODEParams.Y_in
        mdot_in = ODEParams.mdot_in
        h_in    = ODEParams.h_in
        V       = ODEParams.V
        P0      = ODEParams.P0
        k       = ODEParams.k

        ### Definitions 
        self.gas.set_unnormalized_mass_fractions(y[2:])  # Set the mass fractions without normalizing to force sum(Y) == 1.0
        m = y[0]                                         # Set mass of reactor [kg]
      
        rho = m/V                                        # Gas density [kg/m^3]
        #print(rho)
        if rho<0: 
            print('M = ',m, 'p-p = ',self.gas.P-P0, 'm_in = ', mdot_in)
            print(vars(Params))
            
        self.gas.TD = y[1], rho                          # Set temperature and density of the gas
        P = self.gas.P                                   # Get pressure [Pa]
        wdot = self.gas.net_production_rates             # Production rates of the species [kmol/m^3/s] 
        mgen = V*wdot*self.gas.molecular_weights         # Mass generation term
        mdot_out = mdot_in + (P-P0)*k                      # Mass flow out [kg/s]
        
        
        ### Equations of the primary zone (Ideal Gas Constant Pressure Reactor)
        
        # Conservation of mass
        # --------------------
        dmdt = mdot_in - mdot_out
        
        # Conservation of energy 
        # ----------------------
        dTdt = (mdot_in/(m*self.gas.cp_mass))*\
        (h_in - np.dot(self.gas.partial_molar_enthalpies/self.MW, Y_in)) \
        - np.dot(mgen, self.gas.partial_molar_enthalpies/self.MW)/(m*self.gas.cp_mass)
        
        # Conservation of species
        # -----------------------
        dYdt = mgen/(m) + mdot_in*(Y_in - self.gas.Y)/(m)
        
        ### Outputs stacked
        return np.hstack((dmdt,dTdt,dYdt))
    
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
        
