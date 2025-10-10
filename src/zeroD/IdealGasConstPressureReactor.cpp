//! @file ConstPressureReactor.cpp A constant pressure zero-dimensional reactor

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/zeroD/IdealGasConstPressureReactor.h"
#include "cantera/zeroD/FlowDevice.h"
#include "cantera/zeroD/ReactorNet.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/base/utilities.h"
#include "cantera/thermo/PlasmaPhase.h"

namespace Cantera
{

void IdealGasConstPressureReactor::setThermo(ThermoPhase& thermo)
{
    //! @todo: Add a method to ThermoPhase that indicates whether a given
    //! subclass is compatible with this reactor model
    if (thermo.type() != "ideal-gas") {
        throw CanteraError("IdealGasConstPressureReactor::setThermo",
                           "Incompatible phase type provided");
    }
    Reactor::setThermo(thermo);
}

void IdealGasConstPressureReactor::getState(double* y)
{
    if (m_thermo == 0) {
        throw CanteraError("IdealGasConstPressureReactor::getState",
                           "Error: reactor is empty.");
    }
    m_thermo->restoreState(m_state);

    // set the first component to the total mass
    y[0] = m_thermo->density() * m_vol;

    // set the second component to the temperature
    y[1] = m_thermo->temperature();

    // set components y+2 ... y+K+1 to the mass fractions Y_k of each species
    m_thermo->getMassFractions(y+2);

    // set the remaining components to the surface species
    // coverages on the walls
    getSurfaceInitialConditions(y + m_nsp + 2);
}

void IdealGasConstPressureReactor::initialize(double t0)
{
    ConstPressureReactor::initialize(t0);
    m_hk.resize(m_nsp, 0.0);
}

void IdealGasConstPressureReactor::updateState(double* y)
{
    // The components of y are [0] the total mass, [1] the temperature,
    // [2...K+2) are the mass fractions of each species, and [K+2...] are the
    // coverages of surface species on each wall.
    m_mass = y[0];
    m_thermo->setMassFractions_NoNorm(y+2);
    m_thermo->setState_TP(y[1], m_pressure);
    m_vol = m_mass / m_thermo->density();
    updateConnected(false);
    updateSurfaceState(y + m_nsp + 2);
}

void IdealGasConstPressureReactor::eval(double time, double* LHS, double* RHS)
{
    double& dmdt = RHS[0]; // dm/dt (gas phase)
    double& mcpdTdt = RHS[1]; // m * c_p * dT/dt
    double* mdYdt = RHS + 2; // mass * dY/dt

    dmdt = 0.0;
    mcpdTdt = 0.0;

    evalWalls(time);

    m_thermo->restoreState(m_state);
    const vector<double>& mw = m_thermo->molecularWeights();
    const double* Y = m_thermo->massFractions();

    evalSurfaces(LHS + m_nsp + 2, RHS + m_nsp + 2, m_sdot.data());
    double mdot_surf = dot(m_sdot.begin(), m_sdot.end(), mw.begin());
    dmdt += mdot_surf;

    m_thermo->getPartialMolarEnthalpies(&m_hk[0]);

    if (m_chem) {
        m_kin->getNetProductionRates(&m_wdot[0]); // "omega dot"
    }

    // external heat transfer
    mcpdTdt += m_Qdot;
    // --- ADD: plasma power terms (heavy-gas heating) ---
    if (m_energy && m_vol > 0) {
        if (const auto* plasma = dynamic_cast<const PlasmaPhase*>(m_thermo)) {
            const double qJ = plasma->jouleHeatingPower_noexcept();  // W/m^3
            const double qE = plasma->elasticPowerLoss_noexcept();   // W/m^3
            const double q_total = qJ + qE;
            if (std::isfinite(q_total) && q_total != 0.0) {
                mcpdTdt += q_total * m_vol; // [W/m^3]*[m^3] = W → into m*cp*dT/dt (works for CV and CP)
            }
        }
    }
    /* if (auto* plasma = dynamic_cast<PlasmaPhase*>(m_thermo)) {
        const double u_e = plasma->electronMobility();             // [m^2/(V·s)]
        const double q_j = plasma->jouleHeatingPower_noexcept();   // [W/m^3]
        const double q_e = plasma->elasticPowerLoss_noexcept();    // [W/m^3]
        const double mQ  = (q_j + q_e) * m_vol;                    // total power [W]
        const double cp  = m_thermo->cp_mass();                    // J/(kg·K)
        const double rho = m_thermo->density();                    // kg/m^3
        const double mcp = rho * cp * m_vol;                       // [J/K]
        const double mcpdTdt_pred = mQ;                            // since Q = mcp * dT/dt

        writelog("=== PLASMA DEBUG ===\n");
        writelog(fmt::format("u_e      = {:g}  [m^2/(V·s)]\n", u_e));
        writelog(fmt::format("q_j      = {:g}  [W/m^3]\n", q_j));
        writelog(fmt::format("q_e      = {:g}  [W/m^3]\n", q_e));
        writelog(fmt::format("m_Qdot   = {:g}  [W]\n", mQ));
        writelog(fmt::format("mcpdTdt  = {:g}  [W] (should match m_Qdot if balanced)\n", mcpdTdt_pred));
        writelog("====================\n");
    } */

    // --- ADD: plasma power terms (heavy-gas heating) ---
/*     if (auto* plasma = dynamic_cast<PlasmaPhase*>(m_thermo)) {
        // Volumetric Joule heating (σE^2) and elastic e→gas transfer (both are W/m^3)
        const double qJ       = plasma->jouleHeatingPower();
        const double qElastic = plasma->elasticPowerLoss();

        // Convert to total power (multiply by volume) and add as *heating* (positive)
        const double q_total = (qJ + qElastic) * m_vol;
        if (std::isfinite(q_total)) {
            mcpdTdt += q_total;
        }
    } */

    for (size_t n = 0; n < m_nsp; n++) {
        // heat release from gas phase and surface reactions
        mcpdTdt -= m_wdot[n] * m_hk[n] * m_vol;
        mcpdTdt -= m_sdot[n] * m_hk[n];
        // production in gas phase and from surfaces
        mdYdt[n] = (m_wdot[n] * m_vol + m_sdot[n]) * mw[n];
        // dilution by net surface mass flux
        mdYdt[n] -= Y[n] * mdot_surf;
        //Assign left-hand side of dYdt ODE as total mass
        LHS[n+2] = m_mass;
    }

    // add terms for outlets
    for (auto outlet : m_outlet) {
        dmdt -= outlet->massFlowRate(); // mass flow out of system
    }

    // add terms for inlets
    for (auto inlet : m_inlet) {
        double mdot = inlet->massFlowRate();
        dmdt += mdot; // mass flow into system
        mcpdTdt += inlet->enthalpy_mass() * mdot;
        for (size_t n = 0; n < m_nsp; n++) {
            double mdot_spec = inlet->outletSpeciesMassFlowRate(n);
            // flow of species into system and dilution by other species
            mdYdt[n] += mdot_spec - mdot * Y[n];
            mcpdTdt -= m_hk[n] / mw[n] * mdot_spec;
        }
    }

    if (m_energy) {
        LHS[1] = m_mass * m_thermo->cp_mass();
    } else {
        RHS[1] = 0.0;
    }
}

size_t IdealGasConstPressureReactor::componentIndex(const string& nm) const
{
    size_t k = speciesIndex(nm);
    if (k != npos) {
        return k + 2;
    } else if (nm == "mass") {
        return 0;
    } else if (nm == "temperature") {
        return 1;
    } else {
        return npos;
    }
}

string IdealGasConstPressureReactor::componentName(size_t k) {
    if (k == 1) {
        return "temperature";
    } else {
        return ConstPressureReactor::componentName(k);
    }
}

}
