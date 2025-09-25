silmarel
========

Utils
-----

The central tool that allows silmarel to achieve multi-messenger
lensing is the `silmarel.utils.gw_lensing.lens_gw()`. This tool takes in a lens model,
a gravitational wave source position, and returns the time delays and magnifications
needed for MM-lens reconstruction.

.. py:function:: silmarel.utils.gw_lensing.lens_gw(pointmodel, pointkwargs, massmodel, lenskwargs, z_l, z_s, H0, n_images, jax)

   Return a list of random ingredients as strings.

   :param pointmodel: Herculens PointSource model.
   :type pointmodel: Any or None
   :param pointkwargs: Herculens PointSource list-wrapped parameter dict.
   :type pointkwargs: list
   :param massmodel: Herculens MassModel or Lenstronomy LensModel object.
   :type massmodel: Any
   :param lenskwargs: List-wrapped lens parameters.
   :type lenskwargs: list
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


   :return: The ingredients list.
   :rtype: list[str]

Simulation
----------

.. automodule:: silmarel.silmarel.simulation.data_sim
   
Data
----

.. autofunction:: silmarel.silmarel.data.em_data

.. autofunction:: silmarel.silmarel.data.gw_data

Likelihood 
----------
