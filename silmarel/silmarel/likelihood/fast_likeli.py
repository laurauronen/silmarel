from typing import Any

import os

import numpy as np
import matplotlib.pyplot as plt
import emcee

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot

from .data.gw_data import GWData
from .data.em_data import ImageData

class FastLikelihood():

    def __init__(self,
                priors : dict,
                em_data : Any,
                gw_data : Any,
                models : dict,
                outdir : str,
                rewrite : bool = False):

        self.em_prior = priors
        self.em_data = em_data
        self.gw_data = gw_data
        self.models = models

        self.outdir = outdir

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        elif os.path.exists(outdir) and rewrite is False:
            outdir = f"{outdir}_2"
            print(f"Outdir exists already. Writing results to {outdir}")
            os.mkdir(outdir)
        elif os.path.exists(outdir) and rewrite is True: 
            print('WARNING: Overwriting outdir.')

        self.kwargs_likelihood = {'source_marg': False}

        self.multi_band_list = [
            [em_data.kwargs_data,
             em_data.kwargs_psf,
             em_data.kwargs_numerics]
             ]

        self.kwargs_data_joint = {
            'multi_band_list': self.multi_band_list,
            'multi_band_type': 'single-band'
            }

        self.kwargs_constraints = {'linear_solver': False}

        self.fitting_seq = FittingSequence(self.kwargs_data_joint,
                                      models,
                                      self.kwargs_constraints,
                                      self.kwargs_likelihood,
                                      self.em_prior)

        self.chain_list = None
        self.kwargs_result = None
        self.lens_plot = None

        self.em_posterior = None
        self.em_params = None
        self.gw_prior = None

        return

    def run_lens_reconstruction(self,
                                fit_seq : list):

        self.chain_list = self.fitting_seq.fit_sequence(fit_seq)
        self.kwargs_result = self.fitting_seq.best_fit()

        return

    def plot_reconstruction(self):

        self.lens_plot = ModelPlot(self.multi_band_list,
                              self.models,
                              self.kwargs_result,
                              arrow_size=0.02,
                              cmap_string="bone",
                              linear_solver=self.kwargs_constraints.get('linear_solver', True))

        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

        self.lens_plot.data_plot(ax=axes[0,0])
        self.lens_plot.model_plot(ax=axes[0,1])
        self.lens_plot.normalized_residual_plot(ax=axes[0,2])
        self.lens_plot.source_plot(ax=axes[1, 0],
                                   deltaPix_source=0.01,
                                   numPix=100,
                                   v_min=-5,
                                   v_max=0)
        self.lens_plot.convergence_plot(ax=axes[1, 1],
                                        v_max=1)
        self.lens_plot.magnification_plot(ax=axes[1, 2])
        f.tight_layout()
        f.subplots_adjust(left=None,
                          bottom=None,
                          right=None,
                          top=None,
                          wspace=0.,
                          hspace=0.05)
        plt.savefig(self.outdir+'/fit.png')

        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

        self.lens_plot.decomposition_plot(ax=axes[0,0],
                                          text='Lens light',
                                          lens_light_add=True,
                                          unconvolved=True)
        self.lens_plot.decomposition_plot(ax=axes[1,0],
                                          text='Lens light convolved',
                                          lens_light_add=True)
        self.lens_plot.decomposition_plot(ax=axes[0,1],
                                          text='Source light',
                                          source_add=True,
                                          unconvolved=True)
        self.lens_plot.decomposition_plot(ax=axes[1,1],
                                          text='Source light convolved',
                                          source_add=True)
        self.lens_plot.decomposition_plot(ax=axes[0,2],
                                          text='All components',
                                          source_add=True,
                                          lens_light_add=True,
                                          unconvolved=True)
        self.lens_plot.decomposition_plot(ax=axes[1,2],
                                          text='All components convolved',
                                          source_add=True,
                                          lens_light_add=True,
                                          point_source_add=True)
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        plt.savefig(self.outdir+'/decomposition.png')

        sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = self.chain_list[1]
        for i in range(len(self.chain_list)):
            chain_plot.plot_chain_list(self.chain_list, i)
        plt.savefig(self.outdir+'/chains.png')

        self.em_posterior = samples_mcmc
        self.em_params = param_mcmc

        return

    def run_gw_localisation(self,
                            prior : dict,
                            sampler : str = 'emcee',
                            n_iter : int = 1000
                            ):

        self.gw_prior = prior

        if sampler == 'emcee':
            ndim = len(prior.keys())
            nwalkers = 50

            start = []

            for key in prior.keys():
                min, max = prior[key]['min'], prior[key]['max']
                start.append((min + max) / 2)

            pos = [start + 0.1*np.random.randn(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim, self.log_prob,
                                            args=self.gw_data)
            sampler.run_mcmc(pos, n_iter, progress=True)

        return

    def log_prob(self, theta):

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    def log_prior(self, theta):

        x, y = theta[0], theta[1]
        if self.gw_prior['x']['min'] < x < self.gw_prior['x']['max'] and \
            self.gw_prior['y']['min'] < y < self.gw_prior['y']['max']:
            prior_xy = 0.0
        else: 
            prior_xy = -np.inf

        if 'z_s' in self.gw_prior.keys():
            z_s = theta[2]
            if self.gw_prior['z_s']['min'] < z_s < self.gw_prior['z_s']['max']:
                prior_z = 0.0
            else:
                prior_z = -np.inf
        else:
            prior_z = 1.

        if 'H0' in self.gw_prior.keys():
            H0 = theta[3]
            if self.gw_prior['H0']['min'] < H0 < self.gw_prior['H0']['max']:
                prior_h0 = 0.0
            else:
                prior_h0 = -np.inf
        else:
            prior_h0 = 1.

        if -np.inf in [prior_xy, prior_z, prior_h0]:
            return -np.inf

        return 0.0

    def log_likelihood(self):
        return
