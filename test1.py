
import cantera as ct
#ct.CanteraError.set_stack_trace_depth(10)

import numpy as np
import matplotlib.pyplot as plt

# Gaussian pulse parameters
EN_peak = 190 * 1e-21  # Td
pulse_center = 24e-9
pulse_width = 3e-9    # standard deviation in ns

def gaussian_EN(t):
    return EN_peak * np.exp(-((t - pulse_center)**2) / (2 * pulse_width**2))

# setup
gas = ct.Solution('A2NOx_plasma.yaml')