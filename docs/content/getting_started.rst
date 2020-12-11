.. Getting Started
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _getting_started:

Getting Started
===============

Dependencies
#############

Punpy has the following dependencies:

* Python (3.7 or above)
* `numpy <https://numpy.org>`_
* `scipy <https://scipy.org>`_
* `emcee <https://emcee.readthedocs.io/en/stable/>`_
* `numdifftools <https://numdifftools.readthedocs.io/en/latest/>`_


Installation
#############

The easiest way to install punpy is using pip::

   $ pip install punpy

Ideally, it is recommended to do this inside a virtual environment (e.g. conda).

Alternatively, for the latest development version, first go to the folder where you want to save punpy and clone the project repository from GitLab::

   $ git clone git@gitlab.npl.co.uk:eco/tools/punpy.git

Then go into the created directory and install the module with pip::

   $ cd punpy
   $ pip install -e .



