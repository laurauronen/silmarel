Welcome to silmarel's documentation!
====================================

**silmarel** is a package intended to make
multi-messenger lensing analyses smooth and easy
to achieve. To this end, we incorporate both 
usual EM lens reconstruction packages with the 
typical data products of lensed GW analyses.

The package also contains likelihoods designed
for multi-messenger lens reconstruction.

Currently-incorporated packages include: 

- EM: lenstronomy, herculens
- GW: golum-JPE

.. note::
   
   This project is under active development.

Contents
--------

.. toctree::

   usage

Contributing 
------------

We welcome community-driven contributions to the 
silmarel package. The aim is to provide the ability
to complete joint-messenger reconstruction with 
as broad a variety of tools are possible.

Contributions to the package can be made by pull 
request. 

Please note, when incorporating a new package
to silmarel's joint-messenger likelihoods, the 
following should be included as a baseline: 

- A class to take in the correct format of EM data;
- A function that can be incorporated into silmarel.utils.lens_gw();
- A likelihood function for the chosen package;
- An example notebook on how to use the reconstruction likelihood.

A simulation module is optional but recommended to allow
for completion of simulation-based studies using silmarel 
and the package of your choice.

The lens_gw() component should include: 

INPUTS:     lens model, GW ra/dec, redshifts,
            cosmology;
METHOD:     takes in lens mass model and GW source 
            position
OUTPUTS:    time delays, magnifications, 
            image positions.

The likelihood component should be of one of two formats: 

1. A “fast” likelihood: 
Likelihood which computes the EM reconstruction independently
first. After the EM lens posteriors are obtained, these 
are then input into a GW-only likelihood. As the EM likelihood
can be computed beforehand, this would allow one to compute 
the GW localisation directly if the lens reconstruction has 
been done beforehand. See silmarel.likelihood.FastLikelihood
for an example.

2. A “full” likelihood: 
A likelihood which directly takes in the ImageData and GWData objects
and reconstructs both the lens and GW source position directly.
This additionally allows for full joint-lens modelling. 
See silmarel.likelihood.MMLikelihood for an example.

If both methods are suitable, both can be added into silmarel.
