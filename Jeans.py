#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:41:29 2018

@author: tianqin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.constants as cs

def f(hh, y, params):
    h,g = hh  # unpack current values of hh
    a1, a0 = params  # unpack parameters
    derivs = [g, -2*np.divide(g,y)-2*np.divide(a1,y)-np.divide(a0,np.power(1-y,4))*np.exp(h)]
    # list of dy/dt=f functions
    return derivs
# List of constants:
rho0 = 80 # GeV/cm^3
r0 = 2.7 # kpc
sigma0 = 165 # km/s

# Parameters
a1 = 4*np.pi*cs.G*rho0*r0**2/sigma0**2   # -\Phi_B(0)/\sigma_0^2
a0= (rho0/2.2)*(r0/2.7)**2*(150/rho0)**2  # 4\pi G\rho_0 r_0^2/\sigma_0^2


# Initial values
h0 = 0.0     # DM density at the core
g0 = -a1     # ensure the existance of a core

# Bundle parameters for ODE solver
params = [a1, a0]

# Bundle initial conditions for ODE solver
y0 = [h0, g0]

# Make time array for solution
yStop = 1.0 # at most 1.0
yInc = 0.0001
y = np.arange(0.01, yStop, yInc)

# Call the ODE solver
psoln = odeint(f, y0, y, args=(params,))
h = psoln[:,0]

# transform the solution
r = r0*np.divide(y,1-y)
rho = rho0*np.exp(h)

# Plot the solution
fig = plt.figure(1, figsize=(8,8))

ax1 = fig.add_subplot(211)
ax1.loglog(r,rho,label = 'Jeans Solution')
ax1.set_xlabel(r'log_Radius r (kpc)')
ax1.set_ylabel(r'log_Density $(GeV/cm^3)$')

# construct the NFW profile 
rho_NFW = np.divide(rho0, r/r0*(1+r/r0)**2)
ax1.loglog(r,rho_NFW, label = 'NFW')
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.loglog(r[0:90],rho[0:90],label = 'Jeans Solution')
ax2.set_xlabel(r'log_Radius r (kpc)')
ax2.set_ylabel(r'log_Density $(GeV/cm^3)$')

# construct the NFW profile 
rho_NFW = np.divide(rho0, r/r0*(1+r/r0)**2)
ax2.loglog(r[0:90],rho_NFW[0:90], label = 'NFW')
ax2.legend()
plt.savefig('Jeans_1')
plt.show()