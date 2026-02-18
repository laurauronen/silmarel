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

import logging 

silmarel_logger =  logging.getLogger("silmarel")
silmarel_logger.setLevel(logging.INFO)

jax = False

try: 
    from herculens.MassModel.mass_model import MassModel
    from herculens.PointSourceModel.point_source import PointSource
    jax = True
except ImportError or ModuleNotFoundError:
    silmarel_logger.warning("Herculens not found. JAX functionality will be disabled.")
    jax = False

def tdd (z_l: float,
         z_s: float,
         cosmo: Any) -> Any:
  """
    Obtain time delay distance constant for given parameters.

    Parameters
    ----------
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    H0 : float
        Hubble constant for chosen cosmology

    Returns
    -------
    float
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
            likelihood : str = 'lenstronomy') -> dict:
    """
    Function to simulate a lensed gravitational wave.

    Parameters
    ----------
    pointmodel : Any
        Herculens PointSource model for GW.
    pointkwargs : list
        List-wrapped dictionary with GW ra, dec.
    massmodel : Any
        Herculens MassModel/Lenstronomy LensModel object.
    lenskwargs : list
        Herculens/Lenstronomy lens parameters.
    z_l : float
        Lens redshift.
    z_s : float
        Source redshift.
    H0 : float, optional
        Hubble constant (default is 70)
    n_images : int or float or None, optional
        Number of images to compare against.
        - If None: returns the normal GW dictionary.
        - If float: checks if length of GW parameters matches data.
          If not, returns a dummy dictionary without values.
    likelihood : str, optional
        If 'herculens', uses Herculens; else uses Lenstronomy.

    Returns
    -------
    dict
        Dictionary with time delays, relative magnifications,
        image positions, source position, effective 
        luminosity distance.
    """

    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    luminosity_distance = cosmo.luminosity_distance(z_s)

    # conversion factor
    arcsec = 2 * np.pi / 360 / 3600

    if likelihood == 'herculens':
        # make sure that if herculens is not imported, it raises an error
        if jax is False:
            raise ImportError("Herculens is not installed. Cannot use 'herculens' likelihood.")

        # solve image positions
        theta_ra, theta_dec = \
            pointmodel.image_positions(pointkwargs, kwargs_lens=lenskwargs)

        # remove all duplicates
        size = jnp.size(theta_ra) - 1
        unique_x = jnp.unique(theta_ra, return_index=True, size=size)[1]
        theta_ra = theta_ra[unique_x]
        theta_dec = theta_dec[unique_x]

        # compute fermat potential
        fermat_potential = \
            massmodel.fermat_potential(y_image=theta_dec, x_image=theta_ra,
            kwargs_lens=lenskwargs,
            x_source=pointkwargs['ra'],
            y_source=pointkwargs['dec'])

        # convert fermat potential
        fermat_potential = fermat_potential * (arcsec ** 2)

        # solve time delay distance
        dt_distance = tdd(z_l, z_s, cosmo).value

        # get arrival times and magnifications
        arrival_times = jnp.array(dt_distance * fermat_potential)
        magnifications = \
            jnp.array(massmodel.magnification(theta_ra, theta_dec, lenskwargs))
    elif likelihood == 'lenstronomy':
        # set lenstronomy solver and solve image positions
        solver = LensEquationSolver(lensModel = massmodel)
        theta_ra, theta_dec = \
            solver.image_position_from_source(pointkwargs['ra'],
                                                pointkwargs['dec'],
                                                lenskwargs)

        # get magnifications and arrival times
        magnifications = massmodel.magnification(theta_ra, theta_dec, lenskwargs)
        arrival_times = massmodel.arrival_time(theta_ra, theta_dec, lenskwargs)
    else:
        raise ValueError("likelihood must be either 'herculens' or 'lenstronomy'")

    # compute effective luminosity distance of first image
    eff_luminosity_distance = luminosity_distance / magnifications[0]

    # compute relative parameters for GW lensing
    time_delays = (arrival_times - arrival_times[0])[1:]
    relative_magnifications = (magnifications / magnifications[0])[1:]

    # if provided with number of images
    if n_images:
        # if match, return dict
        if len(time_delays) == n_images - 1:
            gw_dictionary = {
                            'image_ra': theta_ra,
                            'image_dec': theta_dec,
                            'delta_t' : time_delays,
                            'relative_magnification' : relative_magnifications
                            }

        # if no match, set dummy values
        else:
            if likelihood == 'herculens':
                gw_dictionary = {'image_ra': - jnp.ones(n_images) * jnp.inf,
                                'image_dec': - jnp.ones(n_images)* jnp.inf, 
                                'delta_t' : - jnp.ones(n_images - 1) * jnp.inf, 
                                'relative_magnification' : \
                                    - jnp.ones(n_images - 1) * jnp.inf
                                }
            elif likelihood == 'lenstronomy':
                gw_dictionary = {'image_ra': - np.ones(n_images) * np.inf,
                                'image_dec': - np.ones(n_images)* np.inf, 
                                'delta_t' : - np.ones(n_images - 1) * np.inf, 
                                'relative_magnification' : \
                                    - np.ones(n_images - 1) * np.inf
                                }

    # if no image number just return lensed dict
    else:
        gw_dictionary = {
                        'image_ra': theta_ra, 
                        'image_dec': theta_dec, 
                        'delta_t' : time_delays,
                        'relative_magnification' : relative_magnifications
                        }

    # add missing keywords in dict
    gw_dictionary['source_ra'] = pointkwargs['ra']
    gw_dictionary['source_dec'] = pointkwargs['dec']
    gw_dictionary['luminosity_distance'] = eff_luminosity_distance.value
    gw_dictionary['z_source'] = z_s

    return gw_dictionary
