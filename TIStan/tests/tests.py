#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:46:00 2019

@author: R. Wesley Henderson
"""

import numpy as np
from numpy import matlib
from .. import TIStan
import dill
import matplotlib.pyplot as plt
import os


# Tell the module where the data files are
package_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(package_directory, 'sinusoid_data_20161004.dill')
stanfile = os.path.join(package_directory, 'sinusoids.stan')


def sin_energy(alpha, data):
    """Sinsuoids energy function."""
    m = data['m']
    t = data['t']
    A = (data['Amax'] - data['Amin']) * alpha[0:m] + data['Amin']
    B = (data['Amax'] - data['Amin']) * alpha[m:2*m] + data['Amin']
    f = (data['freqmax'] - data['freqmin']) * alpha[2*m:3*m] + data['freqmin']
    A = matlib.reshape(A, [m, 1])
    B = matlib.reshape(B, [m, 1])
    f = matlib.reshape(f, [1, m])
    g = ((np.cos(t.dot(2 * np.pi * f)).dot(A)) +
         (np.sin(t.dot(2 * np.pi * f)).dot(B)))
    E = np.sum((g - data['dat']) ** 2) / (2 * data['sigma'] ** 2)
    return E


def full_test():
    """
    This test function will take a long time to run, and it should produce
    valid results.
    """
    # Setup
    with open(filename, 'rb') as f:
        dat = dill.load(f)
        t = dill.load(f)
        dill.load(f)
        fs = dill.load(f)
    print("Done loading")
    n = len(t)
    t_stan = t[:, 0]
    t = matlib.reshape(t, [n, 1])
    dat_stan = dat[:, 0]
    dat = matlib.reshape(dat, [n, 1])
    data1 = {'n': n, 'm': 1, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data2 = {'n': n, 'm': 2, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data3 = {'n': n, 'm': 3, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data4 = {'n': n, 'm': 4, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    obj1 = TIStan(sin_energy, 3, stan_file=stanfile)
    obj2 = TIStan(sin_energy, 6, stan_file=stanfile)
    obj3 = TIStan(sin_energy, 9, stan_file=stanfile)
    obj4 = TIStan(sin_energy, 12, stan_file=stanfile)
    # Run
    out1 = obj1.run(data=data1, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True)
    out2 = obj2.run(data=data2, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True)
    out3 = obj3.run(data=data3, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True)
    out4 = obj4.run(data=data4, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True)
    # Plots results
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [out2[0]/out1[0], out3[0]/out1[0], out4[0]/out1[0]])
    ax.set_ylabel('Log odds vs model 1')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Model 2', 'Model 3', 'Model 4'])
    plt.show()
    return out1, out2, out3, out4


def dummy_test():
    """
    This test function will end each run of TI-Stan early. It will not produce
    valid results. Its purpose is to make sure the package and all dependencies
    are installed correctly.
    """
    # Setup
    with open(filename, 'rb') as f:
        dat = dill.load(f)
        t = dill.load(f)
        dill.load(f)
        fs = dill.load(f)
    print("Done loading")
    n = len(t)
    t_stan = t[:, 0]
    t = matlib.reshape(t, [n, 1])
    dat_stan = dat[:, 0]
    dat = matlib.reshape(dat, [n, 1])
    data1 = {'n': n, 'm': 1, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data2 = {'n': n, 'm': 2, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data3 = {'n': n, 'm': 3, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    data4 = {'n': n, 'm': 4, 't': t, 'Amax': 2.0, 'Amin': -2.0,
             'freqmax': fs/10, 'freqmin': 0, 'dat': dat, 'sigma': 0.1,
             'beta': 1.0, 'dat_stan': dat_stan, 't_stan': t_stan}
    obj1 = TIStan(sin_energy, 3, stan_file=stanfile)
    obj2 = TIStan(sin_energy, 6, stan_file=stanfile)
    obj3 = TIStan(sin_energy, 9, stan_file=stanfile)
    obj4 = TIStan(sin_energy, 12, stan_file=stanfile)
    # Run
    out1 = obj1.run(data=data1, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True, max_iter=2)
    out2 = obj2.run(data=data2, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True, max_iter=2)
    out3 = obj3.run(data=data3, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True, max_iter=2)
    out4 = obj4.run(data=data4, num_mcmc_iter=20, num_chains=32,
                    wmax_over_wmin=1.05, serial=False, smooth=False,
                    verbose=True, max_iter=2)
    # Plots results
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [out2[0]/out1[0], out3[0]/out1[0], out4[0]/out1[0]])
    ax.set_ylabel('Log odds vs model 1')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Model 2', 'Model 3', 'Model 4'])
    plt.show()
    return out1, out2, out3, out4
