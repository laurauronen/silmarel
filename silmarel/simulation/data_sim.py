"""
This module provides simple-to-use functions to wrap around 
lenstronomy/herculens and include lensed GW simulation for 
complete lensed MM simulations.

Author: Laura E. Uronen
Date: 03-Sept-2025
"""

# standard imports
from typing import Any, Optional
from dataclasses import dataclass

# third party imports
import numpy as np

from lenstronomy.Util import image_util, data_util
from lenstronomy.ImSim.image_model import ImageModel as lens_ImageModel
from lenstronomy.Data.imaging_data import ImageData as lens_ImageData
from lenstronomy.Data.pixel_grid import PixelGrid as lens_PixelGrid
from lenstronomy.Data.psf import PSF as lens_PSF

# local imports 
#from ..utils.herculens_gw import *
from ..utils.lenstronomy_gw import lens_gw

class ModelSim():
    """
    Attributes:
    ===========
    pixelgrid (Any):
    psf (Any):

    gw_kwargs (dict):
    gw_data (dict):

    image_data (array):
        'Observed' simulated optical image.

    Methods:
    ========
    lenstronomy_image

    herculens_image
    """

    def __init__(self,
                 models: list[Any, Any, Any],
                 kwargs_models : list[list],
                 kwargs_settings : list[dict],
                 gw_kwargs : Optional[dict] = None,
                 likeli: str = 'lenstronomy'):

        lensmass, sourcelight, lenslight = models
        kwargs_mass, kwargs_source, kwargs_llight = kwargs_models
        self.models = ModelParams(lensmass,
                                  sourcelight,
                                  lenslight,
                                  kwargs_mass,
                                  kwargs_source,
                                  kwargs_llight)

        kwargs_data, kwargs_psf, kwargs_pixel, kwargs_numerics = kwargs_settings
        self.settings = DataParams(kwargs_data, kwargs_psf, kwargs_pixel, kwargs_numerics)

        self.pixelgrid = None
        self.psf = None

        if likeli == 'lenstronomy':
            if gw_kwargs: 
                self.gw_data = lens_gw(gw_kwargs['ra'],
                                    gw_kwargs['dec'],
                                    lensmodel = self.models.LensMass,
                                    kwargs_lens = self.models.mass_kwargs)

            self.pixelgrid = lens_PixelGrid(**self.settings.kwargs_pixel)
            self.psf = lens_PSF(**self.settings.kwargs_psf)

            self.image_data = self.lenstronomy_image()


        elif likeli == 'herculens':

            print("WIP.")

        else:
            print('Please provide either "lenstronomy" or "herculens".')

    def lenstronomy_image(self):
        """
        Create lenstronomy observed image in lenstronomy.

        ARGS
        ====
        None

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

        self.settings.kwargs_data['image_data'] = imageLens
        data_class = lens_ImageData(**self.settings.kwargs_data)

        poisson = image_util.add_poisson(imageLens, exp_time=self.settings.kwargs_data['exposure_time'])
        bkg = image_util.add_background(imageLens, sigma_bkd=self.settings.kwargs_data['background_rms'])

        image_real = imageLens + poisson + bkg
        data_class.update_data(image_real)
        self.settings.kwargs_data['image_data'] = image_real

        return image_real

#    def herculens_image(self):
#        return

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