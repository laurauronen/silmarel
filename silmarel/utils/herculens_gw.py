"""
Tools to lens gravitational wave using Herculens setup/models.
"""

import numpy as np 
import jax.numpy as jnp
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source import PointSource

def tdd (z_lens: float, 
         z_source: float, 
         H0 : float = 70) -> float:
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

    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)

    DL = cosmo.angular_diameter_distance(z=z_lens).to(u.m)
    DS = cosmo.angular_diameter_distance(z=z_source).to(u.m)
    DLS = cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source).to(u.m)
    c = constants.c

    return (1 + z_lens) * (DL * DS / DLS / c).to(u.d)

def time_delays(pointsource : Any,
                pointkwargs: list,
                massmodel: Any,
                lenskwargs: list,
                z_lens: float,
                z_source: float,
                likelihood: None | list = None) -> dict:
    """
    Function to obtain time delays, magnifications given GW source and lens.

    ARGS 
    ====
    pointsource     Herculens PointSource light model
    pointkwargs     List of kwargs of PointSource model
    massmodel       Herculens MassModel for lens mass
    lenskwargs      List of kwargs of MassModel
    z_lens          Lens redshift
    z_source        Source redshift
    likelihood      If None:    function will simply provide
                                lensed GW time delays/magnifications
                    If list:    function will act as likelihood
                                and compare against given set of 
                                GW time delays/magnifications

    RETURNS 
    =======
    gw_dictionary   Dictionary of GW likelihood/values
    """
    arcsec = 2 * np.pi / 360 / 3600

    theta_x, theta_y = pointsource.image_positions(pointkwargs[0], kwargs_lens=lenskwargs)

    size = jnp.size(theta_x) - 1
    unique_x = jnp.unique(theta_x, return_index=True, size=size)[1]
    theta_x = theta_x[unique_x]
    theta_y = theta_y[unique_x]

    fermat_potential = massmodel.fermat_potential(y_image=theta_y, x_image=theta_x,
                                                  kwargs_lens=lenskwargs,
                                                  x_source=pointkwargs[0]['ra'],
                                                  y_source=pointkwargs[0]['dec'])

    fermat_potential = fermat_potential * (arcsec ** 2)
    ddt = tdd(z_lens, z_source).value

    arrival_times = jnp.array(ddt * fermat_potential)
    magnifications = jnp.array(massmodel.magnification(theta_x, theta_y, lenskwargs))

    observables = jnp.array([arrival_times, magnifications, theta_x, theta_y]).T
    observables = observables[observables[:,0].argsort()].T

    time_delay = observables[0]
    relative_magnifications = observables[1]
    theta_x = observables[2]
    theta_y = observables[3]

    time_delay = (time_delay - time_delay[0])[1:]
    relative_magnifications = (relative_magnifications/relative_magnifications[0])[1:]

    if likelihood:
        if len(time_delay) == likelihood:
            gw_dictionary = {
                            'image_ra': theta_x,
                            'image_dec': theta_y,
                            'delta_t' : time_delay
                            'relative_magnification' : relative_magnifications
                            }
        else:
            gw_dictionary = {'image_ra': - jnp.ones(likelihood + 1) * jnp.inf,
                          'image_dec': - jnp.ones(likelihood + 1)* jnp.inf, 
                          'delta_t' : - jnp.ones(likelihood) * jnp.inf, 
                          'relative_magnification' : - jnp.ones(likelihood) * jnp.inf }
    else:
        gw_dictionary = {
                        'image_ra': theta_x, 
                        'image_dec': theta_y, 
                        'delta_t' : time_delay,
                        'relative_magnification' : relative_magnifications
                        }
  
    return gw_dictionary
