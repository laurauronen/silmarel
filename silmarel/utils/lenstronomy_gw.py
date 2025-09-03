"""
Tools to lens gravitational wave using Lenstronomy setup/models.
"""
from typing import Any, Optional

from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

def lens_gw (ra: float,
             dec: float,
             lensmodel: Any,
             kwargs_lens: list | dict) -> dict:
    """
    Given lens model and GW location, create lensed parameters for GW.

    ARGS
    ====
    ra              Right ascension of the GW event
    dec             Declination of the GW event
    lensmodel       Lenstronomy LensModel object
    kwargs_lens     Lens model parameters

    RETURNS
    =======
    gw_dictionary   Dictionary of lensed parameters
    """

    solver = LensEquationSolver(lensModel = lensmodel)
    theta_ra, theta_dec = solver.image_position_from_source(ra, dec, kwargs_lens)

    magnifications = lensmodel.magnification(theta_ra, theta_dec, kwargs_lens)
    arrival_times = lensmodel.arrival_time(theta_ra, theta_dec, kwargs_lens)

    image_types = []

    gw_dictionary = {'muX': magnifications,
                  'delta_t': arrival_times, 
                  'delta_N': image_types,
                  'theta_ra': theta_ra, 
                  'theta_dec': theta_dec}

    return gw_dictionary
