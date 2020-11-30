.. Examples
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _examples:

Examples on how to use the punpy package
==================================================

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
   L1_ut=(L1_ur**2+L1_us**2)**0.5
   L1_cov=punpy.convert_corr_to_cov(np.eye(len(L1_ur)),L1_ur)+punpy.convert_corr_to_cov(np.ones((len(L1_us),len(L1_us))),L1_ur)

   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_ut)
   print(L1_cov)

We now have for each band the random uncertainties in L1, systematic uncertainties in L1, total uncertainty in L1 and the covariance matrix between bands.
Here we have manually specified a diagonal correlation matrix (no correlation, np.eye) for the random component and a correlation matrix of ones (fully correlated, np.ones).
It would also have been possible to do::

   import punpy

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],return_corr=True)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],[L0_us,gains_us,np.zeros(5)],return_corr=True)
   L1_ut=(L1_ur**2+L1_us**2)**0.5
   L1_cov=punpy.convert_corr_to_cov(L1_corr_r,L1_ur)+punpy.convert_corr_to_cov(L1_corr_s,L1_ur)

   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_ut)
   print(L1_cov)

This will give nearly the same results other than a small error due to MC noise.

In addition to propagating random (uncorrelated) and systematic (fully correlated) uncertainties 
it is also possible to propagate uncertainties associated with structured errors.
If we know the covariance matrix for each of the input quantities, it is straigtforward to propagate these.
In the below example we assume the L0 data and dark data to be uncorrelated (their covariance matrix is a, 
diagonal matrix) and gains to be a custom covariance::

   import punpy
   import numpy as np


   L0_cov=punpy.convert_corr_to_cov(np.eye(len(L0_ur)),L0_ur)
   dark_cov=punpy.convert_corr_to_cov(np.eye(len(dark_ur)),dark_ur )
   gains_cov= np.array([[0.45,0.35,0.30,0.20,0.05],
                       [0.35,0.57,0.32,0.30,0.07],
                       [0.30,0.32,0.56,0.24,0.06],
                       [0.20,0.30,0.24,0.44,0.04],
                       [0.05,0.07,0.06,0.04,0.21]])


   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ut,L1_corr=prop.propagate_cov(calibrate,[L0,gains,dark],[L0_cov,gains_cov,dark_cov])
   L1_cov=punpy.convert_corr_to_cov(L1_corr,L1_ut)

   print(L1)
   print(L1_ut)
   print(L1_cov)


It is also possible to include covariance between the input variables. E.g. consider an example similar to the first one but where 
now the dark signal also has systematic uncertainties, which are entirely correlated with the systematic uncertainties on the L0 data (quite commonly the same detector is used for dark and L0). We then have::

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
   
This gives us the random and systematic uncertainties, which can be combined to get the total uncertainty. 

Since within python it is possible to do array operation of any given site, it is often possible to process all 10000 MCsteps in our example at the same time.
For the measurand function we defined L0, gains and dark can be processed using (5,10000) arrays rather than the normal (5,1) arrays that were defined above.
The returned measurand will now also be a (5,10000) array in our example.
This makes the processing of the MC steps as efficient as possible. However, not every measurement function will allow to do this. For example, a radiative 
transfer model cannot process 10000 model inputs at the same time. In this case we can force punpy to process the MC steps one-by-one by setting `parallel_cores` to 1.::

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
 
   prop=punpy.MCPropagation(1000,parallel_cores=1)
   L1=calibrate_slow(L0,gains,dark)
   t1=time.time()
   L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],[L0_ur,gains_ur,dark_ur])
   t2=time.time()
   L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],[L0_us,gains_us,np.zeros(5)])

   print(L1)
   print(L1_ur)
   print(L1_us)
   print("propogate_random took: ",t2-t1," s")

To speed up this slow process, it is also possible to use parallel processing. E.g. if we wanted to do parallel processing using 4 cores::

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
      
      print(L1)
      print(L1_ur)
      print(L1_us)
      print("propogate_random took: ",t2-t1," s")

Propagate_random should now have taken a bit more than 25 s rather than the 100 s when processing them in serial (setting parallel_cores=1).

2D input quantities and measurand
###################################
We can expand the previous example to showcase the processing of 2D input quantities.
Often when taking L0 data, it is good practice to take more than a single set of data.
Now we assume we have 10 repeated measurements of the L0 data, darks and gains and still the same measurement function as before,
and random uncertainties on the L0, dark, and gains which all have the same (10,5) shape, and systematic uncertainties on the gains only (same shape).
In this case, other than the input arrays, very little changes in the propagation method and the uncertainties could be propagates as follows::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],\
                  [0.41,0.82,0.73,0.64,0.93],\
                  [0.45,0.79,0.71,0.66,0.98],\
                  [0.42,0.83,0.69,0.64,0.88],\
                  [0.47,0.75,0.70,0.65,0.78],\
                  [0.45,0.86,0.72,0.66,0.86],\
                  [0.40,0.87,0.67,0.66,0.94],\
                  [0.39,0.80,0.70,0.65,0.87],\
                  [0.43,0.76,0.67,0.64,0.98],\
                  [0.42,0.78,0.69,0.65,0.93])
   dark = np.random.rand(10,5)*0.05
   gains = np.tile(np.array([23,26,28,29,31]),(10,1)) # same gains as before, but repeated 10 times so that shapes match

   # your uncertainties
   L0_ur = np.array([[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]])
   gains_ur = 0.02*L0  # 2% random uncertainty
   gains_us = 0.03*L0  # 3% systematic uncertainty 
   dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],return_corr=True)
   L1_us,L1_Corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],[None,gains_us,None],return_corr=True)
   
   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_corr_r)
   print(L1_Corr_s)

This method works well, but if instead of only (10,5) matrices we get larger matrices 
(e.g. 100 repeated measurements with 100 wavelengths), this becomes quite memory intensive 
(especially since punpy would generate samples with 10000 MCsteps in our example).
Instead when doing propagate_random, or propagate_systematic, is possible to split the calculation along the 
repeated measurements dimension, because we know the correlation between repeated measurements (not correlated
for random, fully correlated for systematic). This can be done by setting the `repeat_dim' keyword::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],\
                  [0.41,0.82,0.73,0.64,0.93],\
                  [0.45,0.79,0.71,0.66,0.98],\
                  [0.42,0.83,0.69,0.64,0.88],\
                  [0.47,0.75,0.70,0.65,0.78],\
                  [0.45,0.86,0.72,0.66,0.86],\
                  [0.40,0.87,0.67,0.66,0.94],\
                  [0.39,0.80,0.70,0.65,0.87],\
                  [0.43,0.76,0.67,0.64,0.98],\
                  [0.42,0.78,0.69,0.65,0.93])
   dark = np.random.rand(10,5)*0.05
   gains = np.tile(np.array([23,26,28,29,31]),(10,1)) # same gains as before, but repeated 10 times so that shapes match

   # your uncertainties
   L0_ur = np.array([[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]])
   gains_ur = 0.02*L0  # 2% random uncertainty
   gains_us = 0.03*L0  # 3% systematic uncertainty 
   dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],return_corr=True,repeat_dim=0)
   L1_us,L1_Corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],[None,gains_us,None],return_corr=True,repeat_dim=0)
   
   print(L1)
   print(L1_ur)
   print(L1_us)





Constants in 1D or 2D measurement functions
##############################################
Allowed within punpy

Constants are expanded into the shape of the input arrays.

E.g. if x2 in the measurement function is a constant::

   x2_array=x2_constant*np.ones_like(x1)

The uncertainty on this constant (single number) is treated as a systematic uncertainty common between all elements of the measurand.