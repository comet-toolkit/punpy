.. Examples
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _examples:

Examples on how to use the punpy package
==================================================

In progress...

1D input quantities and measurand
###################################
Imagine you are trying to calibrate some L0 data to L1 and you have:

-  A measurement function that uses L0 data, gains, and a dark signal in 5 wavelength bands
-  Random uncertainties and systematic uncertainties on the L0 data;
-  Random and systematic uncertainties on the gains;
-  Random uncertainties on the dark signal.

This could look something like::

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
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty

The resulting uncertainty budget can then be calculated with punpy as::

   import punpy

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur])
   L1_us=prop.propagate_systematic(calibrate,[L0,gains,dark],[L0_us,gains_us,np.zeros(5)])
   L1_ut,L1_cov=prop.propagate_both(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],[L0_us,gains_us,np.zeros(5)])

   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_ut)
   print(L1_cov)

We now have for each band the random uncertainties in L1, systematic uncertainties in L1, total uncertainty in L1 and the covariance matrix between bands.


It is also possible to include covariance between the input variables. E.g. consider the previous example but where 
now the dark signal also has systematic uncertainties, which are entirely correlated with the systamtic uncertainties on the L0 data (quite commonly the same detector is used for dark and L0). We then have::

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
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   dark_us = np.array([0.1,0.2,0.1,0.4,0.3])  # random uncertainty

   # correlation matrix between the input variables:
   corr_input_syst=[[1,0,1],[0,1,0],[1,0,1]]  # Here the correlation is between the first and the third variable, following the order of the arguments in the measurement function

After defining this correlation matrix between the systematic uncertainties on the input variables, the resulting uncertainty budget can then be calculated with punpy as::

   import punpy

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur])
   L1_us=prop.propagate_systematic(calibrate,[L0,gains,dark],[L0_us,gains_us,dark_us],corr_between=corr_input_syst)
   
   print(L1)
   print(L1_ur)
   print(L1_us)
   
This gives us the random and systematic uncertainties, which can be combined to get the total uncertainty. If instead we use the corr_between keyword in prop.propagate_both(), then the specified correlation matrix is assumed to apply to the combined random and systematic uncertainties. 
An appropriate correlation matrix should be calculated in that case from the full covariance matrix (which includes both random and systematic contributions).

If we had a slow measurement function and wanted to do parallel processing using 4 cores::

   import punpy
   import time
   import numpy as np

   # your measurement function
   def calibrate_slow(L0,gains,dark):
      time.sleep(0.1)
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty

   if __name__ == "__main__":
      prop=punpy.MCPropagation(1000,parallel_cores=4)
      L1=calibrate_slow(L0,gains,dark)
      t1=time.time()
      L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],[L0_ur,gains_ur,dark_ur])
      t2=time.time()
      L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],[L0_us,gains_us,np.zeros(5)])
      L1_ut,L1_cov = prop.propagate_both(calibrate_slow,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],[L0_us,gains_us,np.zeros(5)])

      print(L1)
      print(L1_ur)
      print(L1_us)
      print(L1_ut)
      print(L1_cov)
      print("propogate_random took: ",t2-t1," s")

Propagate_random should now have taken a bit more than 25 s rather than the 100 s when processing them in serial (setting parallel_cores=1).

2D input quantities and measurand
###################################

Input arrays and measurand of shape (M,N)

Covariance matrices are flattened so that they are 2D arrays. (M*N,M*N)


Constants in 1D or 2D measurement functions
##############################################
Allowed within punpy

Constants are expanded into the shape of the input arrays.

E.g. if x2 in the measurement function is a constant::

   x2_array=x2_constant*np.ones_like(x1)

The uncertainty on this constant (single number) is treated as a systematic uncertainty common between all elements of the measurand.