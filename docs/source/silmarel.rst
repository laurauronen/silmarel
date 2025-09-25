silmarel
========

Utils
-----

The central tool that allows silmarel to achieve multi-messenger
lensing is the `silmarel.utils.gw_lensing.lens_gw()`. This tool takes in a lens model,
a gravitational wave source position, and returns the time delays and magnifications
needed for MM-lens reconstruction.

.. py:function:: silmarel.utils.gw_lensing.lens_gw(pointmodel, pointkwargs, massmodel, lenskwargs, z_l, z_s, H0, n_images, jax)

   Using a given mass model and GW source position, compute lensed GW data (magnifications, time delays, image positions, effective luminosity distance).

   :param pointmodel: Herculens PointSource model.
   :type pointmodel: Any or None
   :param pointkwargs: Herculens PointSource list-wrapped parameter dict.
   :type pointkwargs: list[dict]
   :param massmodel: Herculens MassModel or Lenstronomy LensModel object.
   :type massmodel: Any
   :param lenskwargs: List-wrapped lens parameters.
   :type lenskwargs: list[dict]
   :param z_l: Lens redshift.
   :type z_l: float
   :param z_s: Source redshift.
   :type z_s: float
   :param H0: Hubble constant, default = 70.
   :type H0: float
   :param n_images: If provided, checks that GW parameters match given number of images.
   :type n_images: int
   :param jax: If True, use Herculens, else use Lenstronomy.
   :type jax: bool


   :return: Lensed GW information.
   :rtype: dict

Simulation
----------

.. automodule:: silmarel.silmarel.simulation.data_sim
   
Data
----

.. autofunction:: silmarel.silmarel.data.em_data

.. autofunction:: silmarel.silmarel.data.gw_data

Likelihood 
----------
