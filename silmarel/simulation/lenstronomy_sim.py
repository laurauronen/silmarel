"""
This module provides simple-to-use functions to wrap around 
lenstronomy/herculens and include lensed GW simulation for 
complete lensed MM simulations.

Author: Laura E. Uronen
Date: 03-Sept-2025
"""

# standard imports
import os
from typing import Any, Optional
from dataclasses import dataclass

# third party imports
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from lenstronomy.Util import image_util, data_util
from lenstronomy.ImSim.image_model import ImageModel as lens_ImageModel
from lenstronomy.Data.imaging_data import ImageData as lens_ImageData
from lenstronomy.Data.pixel_grid import PixelGrid as lens_PixelGrid
from lenstronomy.Data.psf import PSF as lens_PSF
from lenstronomy.Plots import lens_plot

# local imports 
from ..utils.gw_lensing import *

class LenstronomySim():
    """
    Attributes:
    ===========

    Methods:
    ========
    lenstronomy_image

    herculens_image
    """

    def __init__(self,
                 models: list[Any],
                 kwargs_models : list[list],
                 kwargs_settings : list[dict],
                 outdir : str,
                 gw_kwargs : Optional[dict] = None,
                 H0 : float = 70,):

        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        # private setup steps
        self._setup_models(models, kwargs_models)
        self._setup_data(kwargs_settings)
        if gw_kwargs:
            self._setup_gw(gw_kwargs, H0)

        self.pixelgrid = lens_PixelGrid(**self.settings.kwargs_pixel)
        self.psf = lens_PSF(**self.settings.kwargs_psf)
        self.image_data = self.lenstronomy_image()
        self.caustic_plot()

    def lenstronomy_image(self):
        """
        Create lenstronomy observed image in lenstronomy.

        RETURNS
        ======= 
        image_real: array
            Image data array.
        """

        imageModel = lens_ImageModel(data_class=self.pixelgrid,
                        psf_class=self.psf,
                        lens_model_class=self.models.LensMass,
                        source_model_class=self.models.SourceLight,
                        lens_light_model_class=self.models.LensLight,
                        point_source_class=None,
                        kwargs_numerics=self.settings.kwargs_numerics)

        imageLens = imageModel.image(kwargs_lens=self.models.mass_kwargs,
                                     kwargs_source=self.models.source_kwargs,
                                     kwargs_lens_light=self.models.lenslight_kwargs,
                                     kwargs_ps=None)

        if self.settings.kwargs_data['background_rms'] is None:
            background_rms = np.sqrt(np.mean((imageLens[0:10,0:10])**2))
            self.settings.kwargs_data['background_rms'] = background_rms

        self.settings.update_image(imageLens)
        data_class = lens_ImageData(**self.settings.kwargs_data)

        poisson = image_util.add_poisson(imageLens, 
                                         exp_time=self.settings.kwargs_data['exposure_time'])
        bkg = image_util.add_background(imageLens, 
                                        sigma_bkd=self.settings.kwargs_data['background_rms'])

        image_real = imageLens + poisson + bkg
        data_class.update_data(image_real)
        self.settings.update_image(image_real)

        return image_real

    def caustic_plot(self):

        if hasattr(self, 'gw_data'):
            source_ra = self.gw_data['source_ra']
            source_dec = self.gw_data['source_dec']
        else:
            source_ra, source_dec = None, None

        _ , ax = plt.subplots(1, 1, figsize=(10, 10))
        lens_plot.lens_model_plot(ax, lensModel=self.models.LensMass,
                                  kwargs_lens=self.models.mass_kwargs,
                                  sourcePos_x=source_ra,
                                  sourcePos_y=source_dec,
                                  point_source=True, with_caustics=True,
                                  fast_caustic=True)
        plt.savefig(self.outdir+'/lens_plot.png')
        plt.close()

    def _setup_models(self, models, kwargs_models):
        lensmass, sourcelight, lenslight = models
        mass_kwargs, source_kwargs, lenslight_kwargs = kwargs_models
        self.models = ModelParams(lensmass,
                                  sourcelight,
                                  lenslight,
                                  mass_kwargs,
                                  source_kwargs,
                                  lenslight_kwargs)

    def _setup_data(self, kwargs_settings):
        kwargs_data, kwargs_psf, kwargs_pixel, kwargs_numerics = kwargs_settings
        self.settings = DataParams(kwargs_data, 
                                   kwargs_psf, 
                                   kwargs_pixel, 
                                   kwargs_numerics)

    def _setup_gw(self, gw_kwargs, H0):
        self.gw_data = lens_gw(pointmodel=None, 
                               pointkwargs=gw_kwargs,
                               massmodel=self.models.LensMass,
                               lenskwargs=self.models.mass_kwargs,
                               z_l = self.models.LensMass.z_lens,
                               z_s = self.models.LensMass.z_source, 
                               H0 = H0)

@dataclass
class ModelParams:
    """
    Attributes:
    ===========
    LensMass (Any):
        LensModel/MassModel object.
    SourceLight (Any):
        LightModel object for source.
    LensLight (Any):
        LightModel object for lens.

    mass_kwargs (list):
        List of dictionaries for lens mass.
    source_kwargs (list):
        List of dictionaries for source.
    lenslight_kwargs (list):
        List of dictionaries for lens light.
    """
    LensMass : Any
    SourceLight : Any
    LensLight : Any

    mass_kwargs : list
    source_kwargs : list
    lenslight_kwargs : list

@dataclass
class DataParams:
    """
    Attributes
    ==========
    kwargs_data (dict):
        Data parameters.
    kwargs_psf (dict):
        PSF settings.
    kwargs_pixel (dict):
        Pixel grid settings.
    kwargs_numerics (dict):
        Additional settings.
    """

    kwargs_data : dict
    kwargs_psf : dict
    kwargs_pixel : dict
    kwargs_numerics : dict

    def update_image(self, image: np.ndarray):
        self.kwargs_data['image_data'] = image