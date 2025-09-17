Usage
=====

.. installation: 

Installation
------------

To use silmarel, first copy the repository into your home directory.

For a conda environment with python=3.10, you can install the package
from the repository: 

.. code-block:: console

   git clone https://github.com/laurauronen/silmarel.git
   cd silmarel/silmarel/
   pip install .

Utils
-----

The central tool that allows silmarel to achieve multi-messenger
lensing is the `silmarel.utils.gw_lensing`. This tool takes in a lens model,
a gravitational wave source position, and returns the time delays and magnifications
needed for MM-lens reconstruction.

.. autofunction:: silmarel.utils.gw_lensing

Simulation
----------

.. autofunction:: silmarel.simulation.data_sim
   
Data
----

.. autofunction:: silmarel.data.em_data

.. autofunction:: silmarel.data.gw_data

Likelihood 
----------
