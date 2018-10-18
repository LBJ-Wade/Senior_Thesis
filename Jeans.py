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
import matplotlib


def Jeans_Solver(rho0,r0,sigma0,f):
    # List of constants:
    #rho0 = 80 # GeV/cm^3
    #r0 = 1.7 # kpc
    #sigma0 = 165 # km/s
    
    # Parameters
    a1 = 4*np.pi*cs.G*rho0*r0**2/sigma0**2   # -\Phi_B(0)/\sigma_0^2
    a0= (rho0/2.2)*(r0/2.7)**2*(150/sigma0)**2  # 4\pi G\rho_0 r_0^2/\sigma_0^2
    
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
    y = np.arange(10**-8, yStop, yInc)
    
    # Call the ODE solver
    psoln = odeint(f, y0, y, args=(params,))
    h = psoln[:,0]
    
    # transform the solution
    r = r0*np.divide(y,1-y)
    rho = rho0*np.exp(h)
    
    return r, rho

    
def NFW(r,rho0, r0):
    # construct the NFW profile 
    rho_NFW = np.divide(rho0, r/r0*(1+r/r0)**2)
    return r,rho_NFW
    
# =============================================================================
# main
# =============================================================================
def f(hh, y, params):
    h,g = hh  # unpack current values of hh
    a1, a0 = params  # unpack parameters
    derivs = [g, -2*np.divide(g,y)-2*np.divide(a1,y)-np.divide(a0,np.power(1-y,4))*np.exp(h)]
    # list of dy/dt=f functions
    return derivs
# Adiabatic 
rho0 = 80 # GeV/cm^3
r0 = 1.5 # kpc
sigma0 = 165 # km/s
r, rho_adiabatic = Jeans_Solver(rho0,r0,sigma0,f)
_, rho_adiabatic_nfw = NFW(r,rho0,r0)

# Non-adiabatic
rho0 = 14 # GeV/cm^3
r0 = 4.0 # kpc
sigma0 = 165 # km/s
r, rho_nonadiabatic = Jeans_Solver(rho0,r0,sigma0,f)
_, rho_nonadiabatic_nfw = NFW(r,rho0,r0)


# Plot the solution
fig = plt.figure(1, figsize=(8,8))

ax1 = fig.add_subplot(211)
ax1.loglog(r,rho_adiabatic,label = 'SIDM',color = 'k')
ax1.loglog(r,rho_nonadiabatic,label = 'SIDM',color = 'r')
ax1.set_xlabel(r'log_Radius r (kpc)')
ax1.set_ylabel(r'log_Density $(GeV/cm^3)$')


ax1.loglog(r,rho_adiabatic_nfw,'--', label = 'NFW with AC',color = 'k')
ax1.loglog(r,rho_nonadiabatic_nfw, '--',label = 'NFW',color = 'r')
ax1.legend()

beg = 400
end = 8000
ax2 = fig.add_subplot(212)
ax2.loglog(r[beg:end],rho_adiabatic[beg:end],label = 'SIDM',color = 'k')
ax2.loglog(r[beg:end],rho_nonadiabatic[beg:end],label = 'SIDM',color = 'r')
ax2.set_xlabel(r'log_Radius r (kpc)')
ax2.set_ylabel(r'log_Density $(GeV/cm^3)$')

# construct the NFW profile 
ax2.loglog(r[beg:end],rho_adiabatic_nfw[beg:end],'--', label = 'NFW with AC',color = 'k')
ax2.loglog(r[beg:end],rho_nonadiabatic_nfw[beg:end], '--',label = 'NFW',color = 'r')
ax2.legend()
ax2.set_xticks([0.2,0.5,1.0,2.0,5.0,10.0])
ax2.set_yticks([0.1,1.0,10,100])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig('Jeans_1')
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.show()