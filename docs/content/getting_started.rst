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
* `comet_maths <https://comet-maths.readthedocs.io/en/latest/>`_
* `obsarray <https://obsarray.readthedocs.io/en/latest/>`_


Installation
#############

The easiest way to install punpy is using pip::

   $ pip install punpy

Ideally, it is recommended to do this inside a virtual environment (e.g. conda).

Alternatively, for the latest development version, first go to the folder where you want to save punpy and clone the project repository from GitLab::

   $ git clone git@github.com:comet-toolkit/punpy.git

Then go into the created directory and install the module with pip::

   $ cd punpy
   $ pip install -e .

Example Usage
##############

For examples on how to use punpy either as a standalone package or with digital effects tables, we refer to the `examples section <https://www.comet-toolkit.org/examples/>`_  on the CoMet Website.
There some jupyter notebooks (hosted on google colab) are available with examples.

Below, we show an example for using punpy standalone for illustration purposes.
For more complete examples with more detailed explanations, we refer to the CoMet website `examples <https://www.comet-toolkit.org/examples/>`_.

In this example, we use punpy to propagate uncertainties through a calibration process::

   import punpy
   import numpy as np

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty

   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty
   # (different for each band but fully correlated)
   gains_utemp = gains*0.03

   corr_temp = []

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur=prop.propagate_random(calibrate,[L0,gains,dark],
         [L0_ur,gains_ur,dark_ur])
   L1_us=prop.propagate_systematic(calibrate,[L0,gains,dark],
         [L0_us,gains_us,np.zeros(5)])
   L1_ut=(L1_ur**2+L1_us**2)**0.5
   L1_cov=punpy.convert_corr_to_cov(np.eye(len(L1_ur)),L1_ur)+\
          punpy.convert_corr_to_cov(np.ones((len(L1_us),len(L1_us))),L1_ur)

   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_ut)
   print(L1_cov)


