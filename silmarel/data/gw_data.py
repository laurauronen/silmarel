"""
Handle GW data reading and processing.
"""

from typing import Optional

import numpy as np
from scipy import stats 

from pandas import DataFrame as DF

from astropy.cosmology import FlatLambdaCDM

import bilby.core.result as bilby_result

class GWData ():
    """
    Object to build KDEs of the GW posteriors provided.
    """

    def __init__(self,
                 n_images,
                 data : Optional[str] = None,
                 sim : Optional[dict] = None,
                 sim_sigma : Optional[dict] = None):

        self.sim = sim
        self.n_images = n_images
        self.sim_sigma = sim_sigma
        
        self.sigma = []
        self.posteriors = []
        self.mins = []
        #self.keys = ['luminosity_distance', 'ra', 'dec']
        self.keys = ['luminosity_distance']

        self.kde = None
        self.density = None

        if data:
            result = bilby_result.read_in_result(data).posterior

            dl = result['luminosity_distance'].values
            self.posteriors.append(dl.values)

            self.mins.append(dl.min())

            self.sigma.append(np.std(dl))

            for key in result.keys():
                if 'relative_magnification' in key or 'delta_t' in key:
                    self.posteriors.append(result[key].values)
                    self.mins.append(result[key].values.min())
                    self.keys.append(key)
                    self.sigma.append(np.std(result[key].values))

            self.kde = stats.gaussian_kde(self.posteriors)
            self.density = self.kde(self.posteriors)

        elif sim:

            self.posteriors.append(sim['luminosity_distance'])
            self.sigma.append(self.sim_sigma['luminosity_distance'])

            for key in sim.keys():
                if key == 'relative_magnification':
                    n = 2
                    for mag in sim[key]:
                        label = f'relative_magnification_{n}1'
                        self.keys.append(label)
                        self.posteriors.append(mag)
                        self.sigma.append(mag * self.sim_sigma['relative_magnification'])
                        n += 1

                if key == 'delta_t':
                    n = 2
                    for t in sim[key]:
                        label = f'delta_t_{n}1'
                        self.keys.append(label)
                        self.posteriors.append(t)
                        self.sigma.append(t * self.sim_sigma['delta_t'])
                        n += 1

        else:
            print("No data or simulation values provided.")

        return

    def get_prob(self,
                 vals : dict) -> dict:

        evals = []

        for key in self.keys:
            evals.append(vals[key])

        px = self.kde.integrate_box(self.mins, vals)

        return px
