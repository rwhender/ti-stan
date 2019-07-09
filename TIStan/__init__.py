# -*- coding: utf-8 -*-
"""
2016-07-13 13:51
2018-09-12 New file
Implement STAN in TI

Thermodynamic Integration, after Goggans and Chi, which is after Skilling
@author: R. Wesley Henderson
"""

from numpy import log, exp, mean, zeros, argsort, Inf, arange, array
from numpy.random import rand, seed
from copy import copy
import pystan
import cProfile
import pickle


class TIStan(object):
    """
    A class meant to facilitate model evidence estimation using thermodyanmic
    integration with Stan, AKA TI-Stan.

    Attributes
    ----------
    energy : function
        a function that returns an energy value
    num_params : int
        number of parameters in the mathematical model being evaluated
    sm : object
        PyStan.StanModel type object. Used to perform NUTS to refresh sample
        population.
    model_logL : float
        model evidence, result
    num_chains_removed : array-like
        array of chains removed for each beta, result
    beta_list : list
        list of beta values, result
    expected_energy : list
        list of expected energy values for each beta value
    alpha : array-like
        array of final alpha values for each chain
    EstarX : array-like
        array of final energy values for each chain

    Methods
    -------
    run(data, num_mcmc_iter=200, num_chains=16, wmax_over_wmin=1.05,
        num_workers=None, serial=False, smooth=False, verbose=False,
        profile=False, max_iter=Inf)
        AFter the object is initialized, this method initializes the TI
        process. Results are returned by this method as well as stored within
        the object.
    """
    def __init__(self, energy, num_params, stan_file=None, stan_pickle=None):
        """
        Initialize TIStan object for thermodynamic integration with PyStan.

        To compile a model from scratch, pass a valid path to a stan file
        in 'stan_file'.

        To use a pickled pre-compiled model, pass a valid path in
        'stan_pickle'.

        Parameters
        ----------
        energy : function handle
            handle to energy function
        num_params : int
            number of model parameters
        stan_file : string
            name of stan file with model details to load
        stan_pickle : string
            location of pickled stan model if available. will be used instead
            of stan file
        """
        self.energy = energy
        if stan_pickle:
            self.sm = pickle.load(open(stan_pickle, 'rb'))
        elif stan_file:
            self.sm = pystan.StanModel(file=stan_file)
        else:
            raise(ValueError("You must pass either stan_file or stan_pickle"))
        self.num_params = num_params

    def run(self, data, num_mcmc_iter=200, num_chains=16, wmax_over_wmin=1.05,
            num_workers=None, serial=False, smooth=False, verbose=False,
            profile=False, max_iter=Inf):
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
        num_workers : int
            Default None. Number of processors to allow to use. None -> the
            number of CPUs in machine
        serial : bool
            if True, run chain evolution serially. If false, run chain
            evolution in parallel using multiprocessing module
        smooth : bool, optional
            default False, if True, smooth ee-beta curve by discarding some
            samples
        verbose : bool, optional
            default False, if True, print messages about progress
        profile : bool, optional
            default False, if True, and serial is False, profile parallel parts
        max_iter : int
            maximum number of temperature iterations

        Returns
        -------
        model_logL : float
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
        """
        out = ti(energy=self.energy, num_params=self.num_params,
                 num_mcmc_iter=num_mcmc_iter, num_chains=num_chains,
                 wmax_over_wmin=wmax_over_wmin, sm=self.sm, data=data,
                 num_workers=num_workers, serial=serial, smooth=smooth,
                 verbose=verbose, profile=profile, max_iter=max_iter)
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
    sorted_idxs = argsort(weight)
    weight.sort()
    alpha = alpha[:, sorted_idxs]
    EstarX = EstarX[sorted_idxs]
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
       data, num_workers, serial, smooth, verbose, profile, max_iter):
    """
    Thermodynamic integration, after Goggans and Chi, 2004

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
    num_workers : int
        Number of processors to allow to use. None -> the number of CPUs
    serial : bool
        Default false. If True, use the non-parallelized worker function
    smooth : bool, optional
        default True, if True, smooth ee-beta curve by discarding some samples
    verbose : bool, optional
        default False, if True, print messages about progress
    max_iter : int
        maximum number of temperature iterations

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
    if num_workers is None:
        num_workers = -1
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

    # Resample the initial chains
    weight = exp(-deltabeta * EstarX)
    num_chains_removed = []
    chain_resample(weight, alpha, EstarX, num_chains_removed, num_chains)

    # Start beta loop
    step = 0
    beta = beta_list[-1]
    while beta > 0 and beta < 1 and step <= max_iter:
        # MCMC loop
        if verbose:
            print("                                                  ", end='\r')
            print("beta =", beta, "ee =", expected_energy[-1], end='\r')
        data['beta'] = copy(beta)
        # Send current step off to chains
        stan_init = [{'alpha': alpha[:, m]} for m in range(num_chains)]
        fit = sm.sampling(iter=num_mcmc_iter, chains=num_chains,
                          algorithm='NUTS',
                          init=stan_init, data=data,
                          check_hmc_diagnostics=False, refresh=0,
                          n_jobs=num_workers)
        fitout = fit.extract(permuted=False)
        alpha = (fitout[-1, :, :num_params]).T
        if isinstance(alpha, float):
            alpha = array([alpha, ])
        for m in range(num_chains):
            EstarX[m] = energy(alpha[:, m], data)
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
    # Compute model log likelihood, but first maybe smooth expected energy
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
