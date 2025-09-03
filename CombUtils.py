import cantera as ct
import numpy as np
from itertools import count, repeat, product
from functools import reduce
import matplotlib.pyplot as plt
import scipy
#from termcolor import colored, cprint

# Helper functions for constructing a Combustor
### Define the
def calc_Stoichiometric_FAR(RM,Fuel,Oxidizer):
    # Add reaction mechanisms folder
    ct.add_directory('./reacMechs')

    # Create solution
    gas = ct.Solution(RM)
    gas.set_equivalence_ratio(1,Fuel,Oxidizer)

    PZ_fuelY = (1-sum(gas['O2','N2', 'H2O'].Y))
    PZ_airY = (sum(gas['O2','N2','H2O'].Y))

    # Print output
    FAR_st = PZ_fuelY/PZ_airY

    return FAR_st

# Define the calcLHV function
def calcLHV(RM,fuel,oxidizer):
    # Create gas
    gas = ct.Solution(RM)

    # Define gas properties
    gas.TP = 298, ct.one_atm
    gas.set_equivalence_ratio(1.0, fuel, oxidizer)

    # Get reactant properties
    h1 = gas.enthalpy_mass

    # Extract fuel components from the fuel string
    fuel_comp = [];
    count = 0
    for i in fuel:
        if i == ':':
            count = count + 1
    if count == 0:
        fuel_comp.append(fuel)
    else:
        n = 0
        for i in np.arange(len(fuel)):
            if fuel[i] == ':':
                fuel_comp.append(fuel[n:i])
            if fuel[i] == ' ':
                n = i + 1

    Y_fuel = 0
    for i in fuel_comp:
        Y_fuel += gas[i].Y[0]

    # Complete combustion products
    X_products = {'CO2': gas.elemental_mole_fraction('C'),
                  'H2O': 0.5 * gas.elemental_mole_fraction('H'),
                  'N2': 0.5 * gas.elemental_mole_fraction('N'),}

    gas.TPX = None, None, X_products
    Y_H2O = gas['H2O'].Y[0]

    # Get product properties
    h2 = gas.enthalpy_mass
    LHV = -(h2-h1)/Y_fuel

    return LHV




def PSR_steadystate_check(states, mdot_in):
    from termcolor import colored
    print('Steady state check:');
    #print('\tSteady state T = ', states[-1].T,
    #      'K, ', colored('5 t-step diff: %.3E'%(states[-1].T - states[-5].T), 'red'))
    print('\tResidence time = ', round(1000*states.m[-1]/mdot_in,3), ' ms')

def PSR_plotter(states, n_reactor):
            fig,ax = plt.subplots(5,1,figsize=(10,10), dpi=100,sharex = True)

            ax[0].plot(states.t, states.T)
            ax[0].set_ylabel('Temp [K]')

            ax[1].plot(states.t,1000*(states.Y[:,states.species_index('NO')]*46/30 +
                             states.Y[:,states.species_index('NO2')])*(states.m))
            ax[1].set_ylabel('NO$_x$ [g]')

            ax[2].plot(states.t,1000*states.Y[:,states.species_index('CO')]*(states.m))
            ax[2].set_ylabel('CO[g]')

            ax[3].plot(states.t, states.m)
            ax[3].set_ylabel('Reac Mass [kg]')

            ax[4].plot(states.t, states.density)
            ax[4].set_ylabel('Reac density [kg/m3]')

            plt.xlabel('t [s]')
            title = 'Primary Zone Reactor #' + str(n_reactor)
            plt.suptitle(title)
            for axis in ax:
                axis.grid(linewidth = 0.5, linestyle = '--', alpha = 0.8)

def warning(text):
    print('\n-------------------------- W A R N I N G -----------------------','red', attrs=['reverse'])
    #cprint("{0:^80s}".format(text), 'red', attrs=['reverse'])
    #cprint('----------------------------------------------------------------','red', attrs=['reverse'])


def print_run_conditions():
    ### Print simulation details for confirmation of model run
    print ("{0:^80s}\n".format('|----------------------------- SIMULATION DETAILS ----------------------------|'))

    print("{0:<20s}{1:^3s}{2:<20s}".format('Reaction Mech',':', Params.RM))
    print("{0:<20s}{1:^3s}{2:<30s}".format('Fuel composition',':', Params.fuelStr))
    print("{0:<20s}{1:^3s}{2:<30s}\n".format('Oxidizer composition',':', Params.Oxidizer))

    print("{0:<30s}{1:^3s}{2: >7.3f}\t{3:<20s}{4:^3s}{5: >8.3f}".format('Core Mass Flow [kg/s]',':',
                                                                        Params.mdot_air,'Tt3 [K]',':',Params.Tt3))
    print("{0:<30s}{1:^3s}{2: >7.3f}\t{3:<20s}{4:^3s}{5: >8.3f}\n".format('Equiv. Fuel Flow rate [kg/s]',':',
                                                                          Params.mdot_fuel,'Pt3 [kPa]',':',Params.Pt3/1000))

    print("{0:^40s} {1:^40s}".format('---------- Primary Zone ----------','--------- Secondary Zone ---------'))
    print ("{0:>41s}".format('|'))
    print("{0:<25s}{1: >5.2f}{2:>11s}".format('PZ equivalance ratio :',Params.PZ_phi_mean,'|'))
    print("{0:<25s}{1: >5.4f}{2:>10s}".format('PZ eq ratio std. dev.:',Params.PZ_phi_sigma,'|'))
    print("{0:<25s}{1: >5.2f}{2:>11s}".format('Number of PZ Reactors:',Params.PZ_n_reactor,'|'))
    print ("{0:>41s}".format('|'))

    print("{0:<25s}{1: >5.4f}{2:>11s}{3:<25s}{4: >5.4f}".format('PZ air flow [kg/s] : ',
                                                                Params.PZ_mdot_air,' | ','SZ air flow [kg/s] : ',Params.SZ_mdot_air))
    print("{0:<25s}{1: >5.2f}{2:>11s}{3:<25s}{4: >5.2f}".format('PZ air frac [%]     : ',
                                                                100*Params.PZ_airfrac,'  | ','SZ airfrac [%]     : ',100*Params.SZ_airfrac))

    print("{0:<25s}{1: >5.4f}{2:>11s}{3:<25s}{4: >5.4f}".format('PZ Volume    [m3]   : ',
                                                                Params.PZ_volume,' | ',   'SZ Volume    [m3]  : ',Params.SZ_volume))

    print("{0:>42s}{1:<25s}{2: >5.2f}".format(' | ','DilZone [% SZ length] :',Params.frac_dil_length*100))
    print("{0:>42s}{1:<25s}{2: >5.2f}".format(' | ','SZ Area   [m2] : ',Params.SZ_A))
    print("{0:>42s}{1:<25s}{2: >5.2f}\n".format(' | ','SZ Area   [m2] : ',Params.SZ_A))

    print("{0:^80s}\n".format('|------------------------------------------------------------------------------|'))


def print_results():
    ### Print model results
    print ("{0:^80s}".format('|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|'))

    print(" {0:<35s}{1: >8.3f} ms\n".format('Residence time in Primary Zone = ',
                                            Params.PZ_volume*PZ_results[2]/ Params.PZ_mdot_in*1000))

    print(" {0:<35s}{1: >8.3f} K".format('PZ Exit Temp = ', PZ_results[0]))
    print(" {0:<35s}{1: >8.3f} K".format('Tt4 (Comb Exit) = ', states_SZ.T[-1]))
    print(" {0:<35s}{1: >8.3f} kg/s\n".format('Mass flow rate = ', states_SZ.M_SZ[-1]))

    print(" {0:<35s}{1: >8.3f} g/kg".format('EI NOx = ', EI_NOx))
    print(" {0:<35s}{1: >8.3f} g/kg".format('EI CO  = ', EI_CO))
    print("{0:^80s}\n\n".format('|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|'))

