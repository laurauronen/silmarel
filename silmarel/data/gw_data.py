"""
Handle GW data reading and processing.
"""

from typing import Optional 

import bilby.core.result as bilby_result
from pandas import DataFrame as DF 

from astropy.cosmology import FlatLambdaCDM
from scipy import stats 

import matplotlib.pyplot as plt
import numpy as np 
class GWData ():
    """
    Object to build KDEs of the GW posteriors provided.
    """

    def __init__(self,
                 data : Optional[str] = None,
                 sim : Optional[dict] = None):

        self.sim = sim
        self.posteriors = []
        self.mins = []
        #self.keys = ['luminosity_distance', 'ra', 'dec']
        self.keys = ['luminosity_distance']

        self.kde = None
        self.density = None

        if data:
            result = bilby_result.read_in_result(data).posterior

            self.posteriors.append(result['luminosity_distance'].values)
            #self.posteriors.append(result['ra'])
            #self.posteriors.append(result['dec'])

            self.mins.append(result['luminosity_distance'].values.min())
            #self.mins.append(result['ra'].min())
            #self.mins.append(result['dec'].min())

            for key in result.keys():
                if 'relative_magnification' in key or 'delta_t' in key:
                    self.posteriors.append(result[key].values)
                    self.mins.append(result[key].values.min())
                    self.keys.append(key)

            self.kde = stats.gaussian_kde(self.posteriors)
            self.density = self.kde(self.posteriors)

        elif sim:

            self.posteriors.append(sim['luminosity_distance'])

            for key in sim.keys():
                if key == 'relative_magnification':
                    n = 2
                    for mag in sim[key]:
                        label = f'relative_magnification_{n}1'
                        self.keys.append(label)
                        self.posteriors.append(mag)
                        n += 1

                if key == 'delta_t':
                    n = 2
                    for t in sim[key]:
                        label = f'delta_t_{n}1'
                        self.keys.append(label)
                        self.posteriors.append(t)
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
