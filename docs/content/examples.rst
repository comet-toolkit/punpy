.. Examples
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _examples:

Examples on how to use the punpy
=================================

For examples on how to use punpy either as a standalone package or with digital effects tables, we refer to the `examples section <https://www.comet-toolkit.org/examples/>`_  on the CoMet Website.
There some jupyter notebooks (hosted on google colab) are available with examples.

Below, we show two typical examples (one standalone and one with digital effects tables) for illustration purposes.
For more complete examples with more detailed explanations, we refer to the CoMet website `examples <https://www.comet-toolkit.org/examples/>`_.

In our first example, we use punpy as standalone in order to propagate uncertainties through a calibration process.

here::

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
