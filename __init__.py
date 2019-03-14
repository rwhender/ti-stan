# -*- coding: utf-8 -*-
"""
2016-07-13 13:51
2018-09-12 New file
Implement STAN in TI

Thermodynamic Integration, after Goggans and Chi, which is after Skilling
@author: Wesley Henderson
"""

from multiprocessing import Pool
from itertools import starmap
from numpy import log, exp, mean, zeros, argsort, Inf, arange, array
from numpy.random import rand, seed
from copy import copy
import pystan
import cProfile


class TIStan(object):
    def __init__(self, energy, stan_file, num_params):
        """
        Initialize TIStan object for thermodynamic integration with PyStan.

        Parameters
        ----------
        energy : function handle
            handle to energy function
        stan_file : string
            name of stan file with model details to load
        """
        self.energy = energy
        self.sm = pystan.StanModel(file=stan_file)
        self.num_params = num_params

    def run(self, data, num_mcmc_iter=200, num_chains=16, wmax_over_wmin=1.05,
            serial=False, smooth=True, verbose=False, profile=False):
        """
        Run thermodynamic integration with given settings (or use defaults).
        All parameters after the first (data) are optional.

        Parameters
        ----------
        data : dictionary
            dictionary of data values to pass to Stan model and energy function
        num_mcmc_iter : int
            number of MCMC iterations to do per beta step
        num_chains : int
            number of parallel MCMC chains to run
        wmax_over_wmin : float
            ratio used to set rate constant for delta beta schedule
        serial : bool
            if True, run chain evolution serially. If false, run chain
            evolution in parallel using multiprocessing module
        smooth : bool, optional
            default True, if True, smooth ee-beta curve by discarding some samples
        verbose : bool, optional
            default False, if True, print messages about progress
        profile : bool, optional
            default False, if True, and serial is False, profile parallel parts
        
        Returns
        -------
        model_log_likelihood : float
        num_chains_removed : array-like
            array of chains removed for each beta
        beta_list : list
            list of beta values
        expected_energy : list
            list of expected energy values for each beta value
        alpha : array-like
            array of final alpha values for each chain
        EstarX : array-like
            array of final energy values for each chain
        energy_count : int
            number of times the energy function was called
        """
        out = ti(energy=self.energy, num_params=self.num_params,
                 num_mcmc_iter=num_mcmc_iter, num_chains=num_chains,
                 wmax_over_wmin=wmax_over_wmin, sm=self.sm, data=data,
                 serial=serial, smooth=smooth, verbose=verbose,
                 profile=profile)
        self.model_logL = out[0]
        self.num_chains_removed = out[1]
        self.beta_list = out[2]
        self.expected_energy = out[3]
        self.alpha = out[4]
        self.EstarX = out[5]
        return out


def profile_worker(pipe, num_iter, sm, energy, num):
    cProfile.runctx('stan_worker(pipe, num_iter, sm, energy)', globals(),
                    locals(), 'prof%d.prof' % num)


def stan_worker(dat, num_iter, sm, energy, alpha):
    """
    Worker function that invokes HMC from Stan

    Parameters
    ----------
    dat : dict
        Dictionary containing everything specified in the 'data' block of the
        stan model. beta inverse temp is included here
    num_iter : int
        number of HMC iterations
    sm : pystan.StanModel
        The StanModel object
    energy : function handle
        energy function
    alpha : array-like
        array of parameters. These should be scaled such that the
        maximum value is 1 and the minimum is 0.

    Returns
    -------
    alpha : array-like
        most recent chain sample
    EstarX : float
        energy of most recent chain sample
    energy_count : int

    """
    seed()
    energy_count = 0
    fit = sm.sampling(iter=num_iter, chains=1, algorithm='HMC',
                      init=[{'alpha': alpha}, ], n_jobs=1, data=dat,
                      check_hmc_diagnostics=False, refresh=0)
    fitout = fit.extract()
    alpha = fitout['alpha'][-1]
    if isinstance(alpha, float):
        alpha = array([alpha, ])
    # EstarX = -1 * fitout['lp__'][-1]
    EstarX = energy(alpha, dat)
    return alpha, EstarX, energy_count + 1


def chain_resample(weight, alpha, EstarX, num_chains_removed, num_chains):
    """Resample the chains according to their weights.

    Parameters
    ----------
    weight : array-like
        num_chains-length array of weights
    alpha : array-like
        num_params x num_chains array of integer parameters. Changed by
        function
    EstarX : array-like
        num_chains-length array of energy values
    num_chains_removed : list
        list of numbers of chains removed for each beta value. Changed by
        function
    num_chains : int
        number of chains

    Returns
    -------
    None : None
    """
    I = argsort(weight)
    weight.sort()
    alpha = alpha[:, I]
    EstarX = EstarX[I]
    weight = (num_chains / weight.sum()) * weight
    randu = rand()
    weight = weight.cumsum()
    samplej = 0
    chain_before = -1
    ncr = 0
    for m in range(num_chains):
        while weight[m] > randu:
            alpha[:, samplej] = alpha[:, m]
            EstarX[samplej] = EstarX[m]
            if chain_before == m:
                ncr += 1
            chain_before = m
            randu += 1
            samplej += 1
    num_chains_removed.append(ncr)


def ti(energy, num_params, num_mcmc_iter, num_chains, wmax_over_wmin, sm,
       data, serial=False, smooth=True, verbose=False, profile=False):
    """Thermodynamic integration, after Goggans and Chi, 2004

    Parameters
    ----------
    energy : function handle
        function handle to energy function
    num_params : int
        number of model parameters
    num_mcmc_iter : int
        number of mcmc steps to take per chain
    num_chains : int
        number of parallel MCMC chains to run
    wmax_over_wmin : float
        ratio used to set rate constant for delta beta schedule
    sm : pystan.StanModel
        StanModel object
    data : dict
        data dictionary for sm and for energy function
    serial : bool
        Default false. If True, use the non-parallelized worker function
    smooth : bool, optional
        default True, if True, smooth ee-beta curve by discarding some samples
    verbose : bool, optional
        default False, if True, print messages about progress

    Returns
    -------
    model_log_likelihood : float
    num_chains_removed : array-like
        array of chains removed for each beta
    beta_list : list
        list of beta values
    expected_energy : list
        list of expected energy values for each beta value
    alpha : array-like
        array of final alpha values for each chain
    EstarX : array-like
        array of final energy values for each chain
    energy_count : int
        number of times the energy function was called
    """
    energy_count = 0
    rate_constant = log(wmax_over_wmin)
    expected_energy = []
    beta_list = []
    # Create some initial points to get an initial beta value.
    alpha = zeros((num_params, num_chains))
    EstarX = zeros(num_chains)
    for m in range(num_chains):
        alpha[:, m] = rand(num_params)
        EstarX[m] = energy(alpha[:, m], data)
        energy_count += 1
    expected_energy.append(mean(EstarX))
    deltabeta = min(rate_constant / (max(EstarX) - min(EstarX)), 1)
    beta_list.append(deltabeta)

    # Resample the initial chains. Probably need to break this into its own
    # function!
    weight = exp(-deltabeta * EstarX)
    num_chains_removed = []
    chain_resample(weight, alpha, EstarX, num_chains_removed, num_chains)

    # Set up input lists for stan worker
    stan_input_gen = [data, num_mcmc_iter, sm, energy]
    # Start pool
    with Pool() as p:
        # Start beta loop
        step = 0
        beta = beta_list[-1]
        while beta > 0 and beta < 1 and step <= Inf:
            # MCMC loop
            if verbose:
                print("                                                  ", end='\r')
                print("beta =", beta, "ee =", expected_energy[-1], end='\r')
            data['beta'] = copy(beta)
            # Send current step off to chains
            stan_inputs = [stan_input_gen for i in range(num_chains)]
            for m in range(num_chains):
                stan_inputs[m] = stan_inputs[m] + [alpha[:, m], ]
            # Evolve chains
            stan_outputs = p.starmap(stan_worker, stan_inputs)
            for m in range(num_chains):
                alpha[:, m] = stan_outputs[m][0]
                EstarX[m] = stan_outputs[m][1]
                energy_count += stan_outputs[m][2]
            # Get the expected energy at this value of beta
            expected_energy.append(EstarX.mean())
            # Compute new beta value
            delta_beta = rate_constant / (max(EstarX) - min(EstarX))
            beta = min(beta + delta_beta, 1)
            beta_list.append(beta)
            if beta_list[-2] + delta_beta > 1:
                delta_beta = 1 - beta_list[-2]
            weight = exp(-delta_beta * EstarX)
            # Resample chains
            chain_resample(weight, alpha, EstarX, num_chains_removed,
                           num_chains)
            step += 1
        # Compute model log likelihood, but first smooth expected energy
        if smooth:
            beta_list, expected_energy, _ = ee_smooth(beta_list,
                                                      expected_energy)
        area = 0.0
        beta_length = len(beta_list)
        for i in arange(1, beta_length):
            area += ((1/2) * (expected_energy[i] + expected_energy[i-1]) *
                     (beta_list[i] - beta_list[i-1]))
        model_log_likelihood = -1 * area
    # Clears status printing carriage return biz
    # print('')
    return (model_log_likelihood, num_chains_removed, beta_list,
            expected_energy, alpha, EstarX, energy_count)


def ee_smooth(betas, ees):
    """Smooth the expected energy vs beta function. EE should be non-increasing
    with beta.

    Parameters
    ----------
    betas : list
        list of beta values
    ees : list
        list of expected energy values

    Returns
    -------
    ctable : list
        smoothed list of beta values
    Ltable : list
        smoothed list of expected energy values
    ntable : list
        list of counts of how many energy values were averaged to produce this
        value at this beta
    """
    ctable = [0.0]
    Ltable = [0.0]
    ntable = [0]
    for beta, ee in zip(betas, ees):
        if beta > ctable[-1]:
            ctable.append(beta)
            Ltable.append(ee)
            ntable.append(1)
        else:
            Ltable[-1] = (ntable[-1] * Ltable[-1] + ee) / (ntable[-1] + 1)
            ntable[-1] += 1
        while len(ctable) > 2 and len(Ltable) > 2 and Ltable[-2] <= Ltable[-1]:
            crecent = ctable.pop()
            Lrecent = Ltable.pop()
            nrecent = ntable.pop()
            ctable[-1] = crecent
            Ltable[-1] = ((ntable[-1] * Ltable[-1] + nrecent * Lrecent) /
                          (ntable[-1] + nrecent))
            ntable[-1] += nrecent
    return ctable, Ltable, ntable
