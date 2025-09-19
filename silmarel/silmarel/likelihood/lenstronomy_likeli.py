"""
Module to run lenstronomy-based fast multi-messenger inference.
"""

from typing import Any, Optional

import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import emcee
from corner import corner

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
from lenstronomy.LensModel.lens_model import LensModel

from ..data.gw_data import GWData
from ..utils.gw_lensing import *
from ..data.em_data import ImageData, EMPosteriors

class LenstronomyLikelihood():
    """
    Lenstronomy-based fast lens reconstruction.

    Attributes:
    ===========
    em_prior:     dict
        Dictionary of EM lens parameter priors.
    gw_prior:     dict
        Dictionary of GW parameter priors.
    em_data:    Any
        Data object for lens reconstruction.
    gw_data:    Any
        GWData() object for GW localisation.
    models:     dict
        List of EM lens/source models.
    outdir:     str
        Path to outdir. 
    rewrite:    bool
        Whether outdir can be safely overwritten.

    Methods:
    ========
    run_lens_reconstruction
    plot_reconstruction
    run_gw_localisation
    """

    def __init__(self,
                em_data : Any,
                gw_data : Any,
                em_prior : dict,
                gw_prior : dict,
                models : dict,
                outdir : str,
                rewrite : bool = False):

        self.em_data = em_data
        self.gw_data = gw_data
        self.em_prior = em_prior
        self.gw_prior = gw_prior
        self.models = models

        self.outdir = outdir

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        elif os.path.exists(outdir): 
            print('WARNING: Overwriting outdir.')
            subprocess.run(f"rm -r {outdir}", shell=True)
            os.mkdir(outdir)

        if isinstance(em_data, ImageData):
            # make lens reconstruction inference settings
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
                                        self.models,
                                        self.kwargs_constraints,
                                        self.kwargs_likelihood,
                                        self.em_prior)
            # set all remaining/postprocessing parameters
            self.chain_list = None
            self.kwargs_result = None
            self.lens_plot = None
            self.em_posterior = None
            self.em_params = None

        elif isinstance(em_data, EMPosteriors):
            self.em_posterior = 1
            self.em_params = 1

        return

    def run_lens_reconstruction(self,
                                fit_seq : list):
        """
        Runs the lens reconstruction using the defined settings.

        ARGS
        ====
        fit_seq :   list
            lenstronomy-appropriate list with selected samplers & 
            sampler settings.
        """

        self.chain_list = self.fitting_seq.fit_sequence(fit_seq)
        self.kwargs_result = self.fitting_seq.best_fit()

        return

    def plot_reconstruction(self):
        """
        Plot lens reconstruction results.
        """

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
                            z_l : float,
                            z_s : Optional[float],
                            H0 : float = 70,
                            sampler : str = 'emcee',
                            n_iter : int = 1000,
                            ):

        prior = self.gw_prior
        self.z_l = z_l
        self.z_s = z_s
        self.H0 = H0

        if 'z_s' in self.gw_prior.keys():
            self.sample_z = True
        else: 
            self.sample_z = z_s

        if self.sample_z is False and z_s is None:
            print('WARNING: No z_s prior or value set. Sampling cannot proceed.')

        if sampler == 'emcee':
            ndim = len(prior.keys())
            nwalkers = 50

            start = []

            for key in prior.keys():
                mmin, mmax = prior[key]['min'], prior[key]['max']
                start.append((mmin + mmax) / 2)

            pos = [start + 0.1*np.random.randn(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim, self.log_prob)
            sampler.run_mcmc(pos, n_iter, progress=True)

            flat_samples = sampler.get_chain(discard=1, thin=1, flat=True)

            self.gw_posterior = flat_samples
            self.gw_params = list(self.gw_prior.keys())

            labels = self.gw_params

            if self.gw_data.sim is not None:
                truths = [self.gw_data.sim['source_ra'],
                          self.gw_data.sim['source_dec']]
                if self.sample_z is True:
                    truths.append(self.gw_data.sim['z_source'])

                fig = corner(flat_samples,
                             labels=labels,
                             truths=truths)
            else:
                fig = corner(flat_samples,
                             labels=labels)
            plt.savefig(self.outdir+'/bbh_corner.png')
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

        if self.sample_z is True:
            z_s = theta[2]
            if self.gw_prior['z_s']['min'] < z_s < self.gw_prior['z_s']['max']:
                prior_z = 0.0
            else:
                prior_z = -np.inf
        else:
            prior_z = 1.

        if -np.inf in [prior_xy, prior_z]:
            return -np.inf

        return 0.0

    def log_likelihood(self, theta):

        if self.gw_data.sim is not None:
            sim = True
        else: 
            sim = False

        rand_idx = np.random.randint(0,len(self.em_posterior))
        rand_sample = self.em_posterior[rand_idx]

        pointmodel = None
        lens_model_list = self.models['lens_model_list']

        sample_list = []

        for j in range(len(lens_model_list)):
            label = f"_lens{j}"
            sample_dict = {}

            for i, key in enumerate(self.em_params):
                if label in key: 
                    key = key.removesuffix(label)
                    sample_dict[key] = rand_sample[i]

            sample_list.append(sample_dict)

        if self.sample_z is True:
            z_s = theta[2]
        else: 
            z_s = self.z_s

        lensModel = LensModel(lens_model_list=lens_model_list, z_source=z_s, z_lens=self.z_l)

        gw_model_dict = lens_gw(pointmodel,
                                {'ra':theta[0],
                                'dec':theta[1]},
                                lensModel,
                                sample_list,
                                self.z_l,
                                z_s,
                                self.H0,
                                n_images = self.gw_data.n_images)
        gw_model_data = GWData(self.gw_data.n_images,
                               sim=gw_model_dict,
                               sim_sigma={'delta_t' : 0.,
                                          'relative_magnification' : 0.,
                                          'luminosity_distance' : 0.})

        log_l = 0

        if sim is True:
            for k, param in enumerate(self.gw_data.posteriors):
                log_l += -0.5 * (gw_model_data.posteriors[k] - param)**2 / self.gw_data.sigma[k]**2
        #elif sim is False: 

        return log_l
