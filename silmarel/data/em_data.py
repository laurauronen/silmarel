from typing import Optional

import numpy as np 

class ImageData():

    def __init__(self, 
                 kwargs_data : dict, 
                 kwargs_psf : dict, 
                 kwargs_numerics : dict,
                 likeli : str = 'lenstronomy'
                 ):
        
        self.data = kwargs_data['image_data']
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.numerics = kwargs_numerics

        return 
        
        
class EMPosteriors():

    def __init__(self):

        return 