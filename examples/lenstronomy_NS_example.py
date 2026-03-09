import numpy as np 

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

import silmarel
from silmarel.simulation.lenstronomy_sim import LenstronomySim as ModelSim
from silmarel.data.em_data import ImageData
from silmarel.data.gw_data import GWData
from silmarel.likelihood.lenstronomy_likeli import LenstronomyLikeli as LL

import pymultinest
import pickle

outdir = 'outdir'

# lens characteristics
center_x, center_y = 0, 0
# source (EM) characteristics
source_x, source_y = 0.2, -0.05
z_lens = 0.7
z_source = 1.5

lens_model_list = ['EPL', 'SHEAR']
lens_light_model_list = ['SERSIC_ELLIPSE'] 
source_light_model_list = ['SERSIC_ELLIPSE']

kwargs_model = {'lens_model_list': lens_model_list,
                'lens_light_model_list': lens_light_model_list,
                'source_light_model_list': source_light_model_list}

kwargs_epl = {'theta_E': 1.2, 'center_x': center_x, 'center_y': center_y, 'e1': 0, 'e2': 0.1, 'gamma': 2.0}
kwargs_shear = {'gamma1': 0.1, 'gamma2': 0.}
kwargs_lens = [kwargs_epl, kwargs_shear]

# lens light
kwargs_lens_light_mag = [{'amp': 50,
                          'R_sersic': .2, 
                          'n_sersic': 4, 
                          'e1': 0, 
                          'e2': 0.1, 
                          'center_x': center_x, 
                          'center_y': center_y}]
# source light
kwargs_source_mag = [{'amp': 250,
                      'R_sersic': 0.04, 
                      'n_sersic': 1, 
                      'e1': -0.1, 
                      'e2': 0.2, 
                      'center_x': source_x, 
                      'center_y': source_y}]

lensModel = LensModel(lens_model_list=lens_model_list, z_source=z_source, z_lens=z_lens)
lensLightModel = LightModel(light_model_list=lens_light_model_list)
sourceLightModel = LightModel(light_model_list=source_light_model_list)

numPix = 40
deltaPix = 0.1
shift = numPix * deltaPix / 2
ra_start, dec_start = center_x - shift, center_y - shift
transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix

fwhm = 0.067 #HST/JWST: 0.067, TMT/ELT: 0.01 in IR

background_rms = None
exposure_time = 2200

kwargs_pixel = {'nx': numPix,
                'ny': numPix, 
                'ra_at_xy_0': ra_start, 
                'dec_at_xy_0': dec_start, 
                'transform_pix2angle': transform_pix2angle}
kwargs_psf = {'psf_type': 'GAUSSIAN', 
              'fwhm': fwhm, 
              'pixel_size': deltaPix}
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
kwargs_data = {'image_data': None, 'background_rms': background_rms, 'exposure_time': exposure_time, 'ra_at_xy_0': ra_start, 'dec_at_xy_0': dec_start, 'transform_pix2angle': transform_pix2angle}
gw_kwargs = {'ra' : source_x, 'dec' : source_y}

models = [lensModel, sourceLightModel, lensLightModel]
kwargs_models = [kwargs_lens, kwargs_source_mag, kwargs_lens_light_mag]
kwargs_settings = [kwargs_data, kwargs_psf, kwargs_pixel, kwargs_numerics]

model = ModelSim(models, kwargs_models, kwargs_settings, outdir, gw_kwargs)

gw_unc = {'delta_t' : 0.05, 'relative_magnification' : 0.2, 'luminosity_distance' : 500}

gwdata = GWData(n_images = 4, sim = model.gw_data, sim_sigma = gw_unc)
with open('true_gw.pkl', 'wb') as f:
    pickle.dump(gwdata, f)

imagedata = ImageData(kwargs_data = model.settings.kwargs_data, 
                        kwargs_psf = model.settings.kwargs_psf,
                        kwargs_numerics = model.settings.kwargs_numerics)

model.caustic_plot()
model.lenstronomy_image()

##############################
# SET UP LENS RECONSTRUCTION #
##############################
# lens models
fixed_lens = []
kwargs_lens_init = []
kwargs_lens_sigma = []
kwargs_lower_lens = []
kwargs_upper_lens = []

fixed_lens.append({})  # for this example, we fix the power-law index of the lens model to be isothermal
kwargs_lens_init.append({'theta_E': 1.0, 'e1': 0., 'e2': 0.,
                         'center_x': 0., 'center_y': 0., 
                         'gamma': 2.0})
kwargs_lens_sigma.append({'theta_E': .5, 'e1': 0.1, 'e2': 0.1,
                         'center_x': 0.1, 'center_y': 0.1, 
                         'gamma': 0.5})
kwargs_lower_lens.append({'theta_E': 0.5, 'e1': -0.5,
                          'e2': -0.5, 'center_x': -0.1, 
                          'center_y': -0.1, 'gamma': 1.0})
kwargs_upper_lens.append({'theta_E': 1.5, 'e1': 0.5,
                          'e2': 0.5, 'center_x': 0.1, 
                          'center_y': 0.1, 'gamma': 3.0})

fixed_lens.append({'ra_0': 0, 'dec_0': 0})
kwargs_lens_init.append({'gamma1': 0., 'gamma2': 0.})
kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
kwargs_lower_lens.append({'gamma1': -0.5, 'gamma2': -0.5})
kwargs_upper_lens.append({'gamma1': 0.5, 'gamma2': 0.5})

lens_params = [kwargs_lens_init, kwargs_lens_sigma, fixed_lens, 
               kwargs_lower_lens, kwargs_upper_lens]

fixed_source = []
kwargs_source_init = []
kwargs_source_sigma = []
kwargs_lower_source = []
kwargs_upper_source = []

fixed_source.append({})
kwargs_source_init.append({'R_sersic': 0.05, 'n_sersic': 1, 
                           'e1': 0, 'e2': 0, 'center_x': 0., 
                           'center_y': 0.0, 'amp': 10})
kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.02, 
                            'e1': 0.5, 'e2': 0.5, 'center_x': 0.2, 
                            'center_y': 0.2, 'amp': 500})
kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 
                            'R_sersic': 0.01, 'n_sersic': .5, 
                            'center_x': -0.1, 'center_y': -0.1, 
                            'amp': 0.1})
kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 
                            'R_sersic': 0.5, 'n_sersic': 5., 
                            'center_x': 0.1, 'center_y': 0.1, 
                            'amp': 1000})

source_params = [kwargs_source_init, kwargs_source_sigma, 
                 fixed_source, kwargs_lower_source, kwargs_upper_source]

fixed_lens_light = []
kwargs_lens_light_init = []
kwargs_lens_light_sigma = []
kwargs_lower_lens_light = []
kwargs_upper_lens_light = []

fixed_lens_light.append({})
kwargs_lens_light_init.append({'R_sersic': 0.2, 'n_sersic': 4,
                               'e1': 0.0, 'e2': 0.0, 'center_x': 0., 
                               'center_y': 0, 'amp': 10})
kwargs_lens_light_sigma.append({'n_sersic': 1, 'R_sersic': 0.1,
                                'e1': 0.5, 'e2': 0.5, 'center_x': 0.1, 
                                'center_y': 0.1, 'amp': 5})
kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5,
                                'R_sersic': 0.01, 'n_sersic': .5, 
                                'center_x': -0.1, 'center_y': -0.1, 
                                'amp': 0.1})
kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5,
                                'R_sersic': 1, 'n_sersic': 5., 
                                'center_x': 0.1, 'center_y': 0.1, 
                                'amp': 1000})

lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma,
                     fixed_lens_light, kwargs_lower_lens_light,
                     kwargs_upper_lens_light]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params,
                }

kwargs_params_gw = {'x' : {'min': -0.3, 'max': 0.3},
                    'y' : {'min': -0.3, 'max': 0.3},
                     #'z_s': {'prior': 'gaussian', 'mean': 2.0, 'sigma': 0.5}
                     }

fastlikelihood = LL(
    em_data = imagedata,
    gw_data = gwdata,
    em_prior = kwargs_params,
    gw_prior = kwargs_params_gw,
    models = kwargs_model,
    z_l = 0.7,
    z_s = 1.5,
    outdir = outdir,
    overwrite = False
)

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 10,
                                'n_iterations': 5000}],
                      ['MCMC', {'n_burn': 2000, 'n_run': 100000,
                                 'n_walkers': 50, 'sigma_scale': .1}]
                       ]

fastlikelihood.run_lens_reconstruction(fitting_kwargs_list)
fastlikelihood.plot_reconstruction()
fastlikelihood.run_gw_localisation(sampler = 'MultiNest')
