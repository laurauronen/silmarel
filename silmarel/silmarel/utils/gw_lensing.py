"""
Tools to lens gravitational wave using Lenstronomy setup/models.
"""
from typing import Any, Optional

import numpy as np
import jax.numpy as jnp

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants

from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source import PointSource

def tdd (z_l: float,
         z_s: float,
         cosmo: Any) -> Any:
    """
    Obtain time delay distance constant for given parameters. 

    ARGS
    ====
    z_lens      Lens redshift
    z_source    Source redshift
    H0          Hubble constant for chosen cosmology

    RETURNS
    =======
                Time delay distance value.

    """

    DL = cosmo.angular_diameter_distance(z=z_l).to(u.m)
    DS = cosmo.angular_diameter_distance(z=z_s).to(u.m)
    DLS = cosmo.angular_diameter_distance_z1z2(z1=z_l, z2=z_s).to(u.m)
    c = constants.c

    return (1 + z_l) * (DL * DS / DLS / c).to(u.d)

def lens_gw(pointmodel: Optional[Any],
            pointkwargs: list[dict],
            massmodel: Any,
            lenskwargs: list[dict],
            z_l : float,
            z_s : float,
            H0 : float = 70,
            n_images : Optional[float] = None,
            jax : bool = False) -> dict:

    """
    Function to simulate a lensed gravitational wave.

    ARGS
    ====
    pointmodel          Herculens PointSource model for GW.
    pointkwargs         List-wrapped dictionary with GW ra, dec.
    massmodel           Herculens MassModel/Lenstronomy LensModel object.
    lenskwargs          Herculens/Lenstronomy lens parameters.
    z_l                 Lens redshift.
    z_s                 Source redshift.
    H0                  Hubble constant.
    n_images            Number of images to compare against.
                        If None:    will return the normal GW dictionary.
                        If float:   will check if length of GW parameters 
                                    matches given data.
                                    If not, returns a dummy dictionary without values.
    jax                 If True, uses Herculens, else uses Lenstronomy.

    RETURNS
    =======
    gw_dictionary       Dictionary with time delays, relative magnifications,
                        image positions, source position, effective 
                        luminosity distance. 
    """

    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    luminosity_distance = cosmo.luminosity_distance(z_s)

    if jax is True:
        arcsec = 2 * np.pi / 360 / 3600

        theta_ra, theta_dec = \
            pointmodel.image_positions(pointkwargs, kwargs_lens=lenskwargs)

        size = jnp.size(theta_ra) - 1
        unique_x = jnp.unique(theta_ra, return_index=True, size=size)[1]
        theta_ra = theta_ra[unique_x]
        theta_dec = theta_dec[unique_x]

        fermat_potential = \
            massmodel.fermat_potential(y_image=theta_dec, x_image=theta_ra,
            kwargs_lens=lenskwargs,
            x_source=pointkwargs['ra'],
            y_source=pointkwargs['dec'])

        fermat_potential = fermat_potential * (arcsec ** 2)
        dt_distance = tdd(z_l, z_s, cosmo).value
        arrival_times = jnp.array(dt_distance * fermat_potential)
        magnifications = \
            jnp.array(massmodel.magnification(theta_ra, theta_dec, lenskwargs))
    else: 
        solver = LensEquationSolver(lensModel = massmodel)
        theta_ra, theta_dec = \
            solver.image_position_from_source(pointkwargs['ra'],
                                                pointkwargs['dec'],
                                                lenskwargs)

        magnifications = massmodel.magnification(theta_ra, theta_dec, lenskwargs)
        arrival_times = massmodel.arrival_time(theta_ra, theta_dec, lenskwargs)

    eff_luminosity_distance = luminosity_distance / magnifications[0]

    time_delays = (arrival_times - arrival_times[0])[1:]
    relative_magnifications = (magnifications / magnifications[0])[1:]

    if n_images:
        if len(time_delays) == n_images - 1:
            gw_dictionary = {
                            'image_ra': theta_ra,
                            'image_dec': theta_dec,
                            'delta_t' : time_delays,
                            'relative_magnification' : relative_magnifications
                            }
        else:
            if jax is True:
                gw_dictionary = {'image_ra': - jnp.ones(n_images) * jnp.inf,
                                'image_dec': - jnp.ones(n_images)* jnp.inf, 
                                'delta_t' : - jnp.ones(n_images - 1) * jnp.inf, 
                                'relative_magnification' : \
                                    - jnp.ones(n_images - 1) * jnp.inf
                                }
            else:
                gw_dictionary = {'image_ra': - np.ones(n_images) * np.inf,
                                'image_dec': - np.ones(n_images)* np.inf, 
                                'delta_t' : - np.ones(n_images - 1) * np.inf, 
                                'relative_magnification' : \
                                    - np.ones(n_images - 1) * np.inf
                                }
    else:
        gw_dictionary = {
                        'image_ra': theta_ra, 
                        'image_dec': theta_dec, 
                        'delta_t' : time_delays,
                        'relative_magnification' : relative_magnifications
                        }

    gw_dictionary['source_ra'] = pointkwargs['ra']
    gw_dictionary['source_dec'] = pointkwargs['dec']
    gw_dictionary['luminosity_distance'] = eff_luminosity_distance.value

    return gw_dictionary