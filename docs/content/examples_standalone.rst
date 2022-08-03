.. Examples
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _examples_standalone:

Examples on how to use the punpy package standalone
==================================================

1D input quantities and measurand
###################################
Imagine you are trying to calibrate some L0 data to L1 and you have:

-  A measurement function that uses L0 data, gains, and a dark signal in 5 wavelength bands
-  Random uncertainties and systematic uncertainties on the L0 data;
-  Random and systematic uncertainties on the gains;
-  Random uncertainties on the dark signal.s

After defining the data, the resulting uncertainty budget can then be calculated with punpy using the MC methods as::

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
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty

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

We now have for each band the random uncertainties in L1, systematic uncertainties in L1, total uncertainty in L1 and the covariance matrix between bands.
Here we have manually specified a diagonal correlation matrix (no correlation, np.eye) for the random component and a correlation matrix of ones (fully correlated, np.ones).
It would also have been possible to use the keyword `return_corr` to get the measured correlation matrix. In the next example we use the LPU methods set the `return_corr` keyword::

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
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],return_corr=True)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [L0_us,gains_us,np.zeros(5)],return_corr=True)
   L1_ut=(L1_ur**2+L1_us**2)**0.5
   L1_cov=punpy.convert_corr_to_cov(L1_corr_r,L1_ur)+\
          punpy.convert_corr_to_cov(L1_corr_s,L1_ur)

   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_ut)
   print(L1_cov)

This will give nearly the same results other than a small error due to MC noise.

Next we give an example where we try out a measurement function with multiple outputs.
In order to process a measurement function with multiple outputs, it is necessary to set the keyword `output_vars` to the number of outputs::

   import punpy
   import numpy as np

   # your measurement function
   def calibrate_2output(L0,gains,dark):
      return (L0-dark)*gains,(L0*gains-dark)

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   
   prop=punpy.MCPropagation(10000)
   L1=calibrate_2output(L0,gains,dark)
   L1_ur,L1_corr_r,L1_corr_r_between=prop.propagate_random(
                                     calibrate_2output,[L0,gains,dark],
                                     [L0_ur,gains_ur,dark_ur],
                                     return_corr=True,output_vars=2)
   L1_us,L1_corr_s,L1_corr_s_between=prop.propagate_systematic(
                                     calibrate_2output,[L0,gains,dark],
                                     [L0_us,gains_us,np.zeros(5)],
                                     return_corr=True,output_vars=2)
   
   print(L1)
   print(L1_ur)
   print(L1_us)

Due to the multiple vars, L1_ur now has the shape (2,5) so L1_ur[0] now has the same uncertainties as 
the previous example, L1_corr_r[0] is the same as L1_corr_r before. Analogously, L1_ur[1] and L1_corr_r[0]
give the random uncertainty and correlation matrix for the second output of the measurand.
There is now also a L1_corr_r_between which gives the correlation matrix between the two output variables 
of the measurment function (averaged over all wavelengths).

In addition to propagating random (uncorrelated) and systematic (fully correlated) uncertainties 
it is also possible to propagate uncertainties associated with structured errors.
If we know the covariance matrix for each of the input quantities, it is straigtforward to propagate these.
In the below example we assume the L0 data and dark data to be uncorrelated (their covariance matrix is a, 
diagonal matrix) and gains to be a custom covariance::

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

   L0_cov=punpy.convert_corr_to_cov(np.eye(len(L0_ur)),L0_ur)
   dark_cov=punpy.convert_corr_to_cov(np.eye(len(dark_ur)),dark_ur )
   gains_cov= np.array([[0.45,0.35,0.30,0.20,0.05],
                       [0.35,0.57,0.32,0.30,0.07],
                       [0.30,0.32,0.56,0.24,0.06],
                       [0.20,0.30,0.24,0.44,0.04],
                       [0.05,0.07,0.06,0.04,0.21]])


   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ut,L1_corr=prop.propagate_cov(calibrate,[L0,gains,dark],
                                    [L0_cov,gains_cov,dark_cov])
   L1_cov=punpy.convert_corr_to_cov(L1_corr,L1_ut)

   print(L1)
   print(L1_ut)
   print(L1_cov)


It is also possible to include covariance between the input variables. E.g. consider an example similar to the first one but where 
now the dark signal also has systematic uncertainties, which are entirely correlated with the systematic uncertainties on the L0 data 
(quite commonly the same detector is used for dark and L0). After defining this correlation matrix between the systematic uncertainties 
on the input variables, the resulting uncertainty budget can then be calculated with punpy as::

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
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   dark_us = np.array([0.1,0.2,0.1,0.4,0.3])  # random uncertainty

   # correlation matrix between the input variables:
   corr_input_syst=[[1,0,1],[0,1,0],[1,0,1]]  # Here the correlation is
   # between the first and the third variable, following the order of 
   # the arguments in the measurement function

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur=prop.propagate_random(calibrate,[L0,gains,dark],
                               [L0_ur,gains_ur,dark_ur])
   L1_us=prop.propagate_systematic(calibrate,[L0,gains,dark],
         [L0_us,gains_us,dark_us],corr_between=corr_input_syst)
   
   print(L1)
   print(L1_ur)
   print(L1_us)
   
This gives us the random and systematic uncertainties, which can be combined to get the total uncertainty. 

Since within python it is possible to do array operation using arrays of any size (as long as shapes of different arrays match up), 
it is often possible to process all 10000 MCsteps in our example at the same time.
For the measurand function we defined L0, gains and dark can be processed using (5,10000) arrays rather than the normal (5,1) arrays that were defined above.
The returned measurand will now also be a (5,10000) array in our example.
This makes the processing of the MC steps as efficient as possible. However, not every measurement function will allow to do this. For example, a radiative 
transfer model cannot process 10000 model inputs at the same time. In this case we can force punpy to process the MC steps one-by-one by setting `parallel_cores` to 1.::

   import punpy
   import time
   import numpy as np

   # your measurement function
   def calibrate_slow(L0,gains,dark):
      y2=np.repeat((L0-dark)*gains,30000)
      y2=y2+np.random.random(len(y2))
      y2=y2.sort()
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   
   prop=punpy.MCPropagation(1000,parallel_cores=1)
   L1=calibrate_slow(L0,gains,dark)
   t1=time.time()
   L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],
                                 [L0_ur,gains_ur,dark_ur])
   t2=time.time()
   L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],
                                     [L0_us,gains_us,np.zeros(5)])

   print(L1)
   print(L1_ur)
   print(L1_us)
   print("propogate_random took: ",t2-t1," s")

We compare this to the runtime for the LPU methods::

   import punpy
   import time
   import numpy as np

   # your measurement function
   def calibrate_slow(L0,gains,dark):
      y2=np.repeat((L0-dark)*gains,30000)
      y2=y2+np.random.random(len(y2))
      y2=y2.sort()
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   
   prop=punpy.LPUPropagation(parallel_cores=1)
   L1=calibrate_slow(L0,gains,dark)
   t1=time.time()
   L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],
                                 [L0_ur,gains_ur,dark_ur])
   t2=time.time()
   L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],
                                     [L0_us,gains_us,np.zeros(5)])

   print(L1)
   print(L1_ur)
   print(L1_us)
   print("propogate_random took: ",t2-t1," s")

We find that the LPU method is faster in this case. Though this depends on the number of MCsteps that is used in the MC method and the number of elements in the Jacobian (here 5*5).
To speed up this slow process, it is also possible to use parallel processing. E.g. if we wanted to do parallel processing using 4 cores::

   import punpy
   import time
   import numpy as np

   # your measurement function
   def calibrate_slow(L0,gains,dark):
      y2=np.repeat((L0-dark)*gains,30000)
      y2=y2+np.random.random(len(y2))
      y2=y2.sort()
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   
   if __name__ == "__main__":
      prop=punpy.MCPropagation(1000,parallel_cores=6)
      L1=calibrate_slow(L0,gains,dark)
      t1=time.time()
      L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],
                                    [L0_ur,gains_ur,dark_ur])
      t2=time.time()
      L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],
                                        [L0_us,gains_us,np.zeros(5)])
      
      print(L1)
      print(L1_ur)
      print(L1_us)
      print("propogate_random took: ",t2-t1," s")

By using 6 cores, Propagate_random should now be faster than the LPU method and significantly faster than when processing them in serial (setting parallel_cores=1).
Here, there is no point to do parallel processing for the LPU methods because these methods can only be run in parallel when the `repeat_dims` keyword is set (see next section).
However it is only possible to speed up the LPU methods in this case. Since all of the input quantities are of the same shape as the measurand, 
and the measurement function works on each measurement independently (The calibrations of different wavelengths don't affect eachother), we know that the Jacobian
will only have diagonal elements. This means we can set the `Jx_diag` keyword to True (either when creating the object, or for an individual propagation method). 
This significantly speeds up the calculation as the off-diagonal elements of the Jacobian don't need to be calculated::

   import punpy
   import time
   import numpy as np

   # your measurement function
   def calibrate_slow(L0,gains,dark):
      y2=np.repeat((L0-dark)*gains,30000)
      y2=y2+np.random.random(len(y2))
      y2=y2.sort()
      return (L0-dark)*gains

   # your data
   L0 = np.array([0.43,0.8,0.7,0.65,0.9])
   dark = np.array([0.05,0.03,0.04,0.05,0.06])
   gains = np.array([23,26,28,29,31])

   # your uncertainties
   L0_ur = L0*0.05  # 5% random uncertainty
   L0_us = np.ones(5)*0.03  # systematic uncertainty of 0.03 
                            # (common between bands)
   gains_ur = np.array([0.5,0.7,0.6,0.4,0.1])  # random uncertainty
   gains_us = np.array([0.1,0.2,0.1,0.4,0.3])  # systematic uncertainty 
   # (different for each band but fully correlated)
   dark_ur = np.array([0.01,0.002,0.006,0.002,0.015])  # random uncertainty
   
   prop=punpy.LPUPropagation(parallel_cores=1,Jx_diag=True)
   L1=calibrate_slow(L0,gains,dark)
   t1=time.time()
   L1_ur = prop.propagate_random(calibrate_slow,[L0,gains,dark],
                                 [L0_ur,gains_ur,dark_ur])
   t2=time.time()
   L1_us = prop.propagate_systematic(calibrate_slow,[L0,gains,dark],
                                     [L0_us,gains_us,np.zeros(5)])

   print(L1)
   print(L1_ur)
   print(L1_us)
   print("propogate_random took: ",t2-t1," s")



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
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98],
                  [0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86],
                  [0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.43,0.76,0.67,0.64,0.98],
                  [0.42,0.78,0.69,0.65,0.93]])
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
   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],return_corr=True)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [None,gains_us,None],return_corr=True)
   
   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_corr_r)
   print(L1_corr_s)

Note that the correlation matrices have a shape of (50,50), thus giving the correlation coefficient between all 50 elements of the L0 data. 
Often we know the correlation between repeated measurements and are only interested in the corrlation matrix along a specific axis (in our 
example the wavelength axis). If this is the case, this axis can be indicated by giving the `corr_axis` keyword the relevant dimension 
(1 here because wavelength dimension has index 1)::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98],
                  [0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86],
                  [0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.43,0.76,0.67,0.64,0.98],
                  [0.42,0.78,0.69,0.65,0.93]])
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
   gains_ur = 0.02*gains # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],return_corr=True,corr_axis=1)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [None,gains_us,None],return_corr=True,corr_axis=1)
   
   print(L1)
   print(L1_ur)
   print(L1_us)
   print(L1_corr_r)
   print(L1_corr_s)

This method works well, but if instead of only (10,5) matrices we get larger matrices 
(e.g. 100 repeated measurements with 100 wavelengths), this becomes quite memory intensive when using the MC methods
(especially since punpy would generate samples with 10000 MCsteps in our example).
Instead when doing propagate_random, or propagate_systematic, is possible to split the calculation along the 
repeated measurements dimension, because we know the correlation between repeated measurements (not correlated
for random, fully correlated for systematic). This can be done by setting the `repeat_dims` keyword::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98],
                  [0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86],
                  [0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.43,0.76,0.67,0.64,0.98],
                  [0.42,0.78,0.69,0.65,0.93]])
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
   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],return_corr=True,
                   repeat_dims=0,corr_axis=1)
   L1_us,L1_Corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [None,gains_us,None],return_corr=True,
                   repeat_dims=0,corr_axis=1)
   
   print(L1)
   print(L1_ur)
   print(L1_us)

This way the code uses less memory and as a result is typically faster.
There is also an important benefit setting `repeat_dims` when using LPU methods.
Without setting the `repeat_dims` keyword, the Jacobian that needs to be calculated has 50*50 elements.
When setting the `repeat_dims` keyword, the Jacobian is calculated for each repeated measurement individually,
which means that will be 10*5*5 (10 repeats of Jacobain over 5 wavelengths). This means that there are 10 times less
elements calculated than the case without `repeat_dims`. This significantly speeds up the calculation.
This means there is not possible to account for how the different repeat measurements affect eachother.
However, the assumption with repeated measurments is that they can be separated, and that the correlation between them is known
anyway, so this is not a problem. We find that the following example is much faster then running the same without the `repeat_dims` keyword set::

    import numpy as np
    import punpy
    import time

    # your measurement function
    def calibrate_slow(L0,gains,dark):
        y2=np.repeat((L0-dark)*gains,3000)
        y2=y2+np.random.random(len(y2))
        y2=y2.sort()
        return (L0-dark)*gains

    # your data
    L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                [0.41,0.82,0.73,0.64,0.93],
                [0.45,0.79,0.71,0.66,0.98],
                [0.42,0.83,0.69,0.64,0.88],
                [0.47,0.75,0.70,0.65,0.78],
                [0.45,0.86,0.72,0.66,0.86],
                [0.40,0.87,0.67,0.66,0.94],
                [0.39,0.80,0.70,0.65,0.87],
                [0.43,0.76,0.67,0.64,0.98],
                [0.42,0.78,0.69,0.65,0.93]])
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
    gains_ur = 0.02*gains  # 2% random uncertainty
    gains_us = 0.03*gains  # 3% systematic uncertainty
    dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

    if __name__ == "__main__":

        prop=punpy.LPUPropagation()
        L1=calibrate_slow(L0,gains,dark)
        t1=time.time()

        L1_ur,L1_corr_r=prop.propagate_random(calibrate_slow,[L0,gains,dark],
                        [L0_ur,gains_ur,dark_ur],
                        return_corr=True,corr_axis=1,repeat_dims=0)

        t2=time.time()

        print(L1)
        print(L1_ur)
        print("propogate_random took: ",t2-t1," s")


There is another important benefit to setting the `repeat_dims` keyword when using the LPU methods.
In this case it is possible to use parallel processing, in which case each repeated measurements is processed in parallel.
This again speeds up the process::

    import numpy as np
    import punpy
    import time

    # your measurement function
    def calibrate_slow(L0,gains,dark):
        y2=np.repeat((L0-dark)*gains,3000)
        y2=y2+np.random.random(len(y2))
        y2=y2.sort()
        return (L0-dark)*gains

    # your data
    L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                [0.41,0.82,0.73,0.64,0.93],
                [0.45,0.79,0.71,0.66,0.98],
                [0.42,0.83,0.69,0.64,0.88],
                [0.47,0.75,0.70,0.65,0.78],
                [0.45,0.86,0.72,0.66,0.86],
                [0.40,0.87,0.67,0.66,0.94],
                [0.39,0.80,0.70,0.65,0.87],
                [0.43,0.76,0.67,0.64,0.98],
                [0.42,0.78,0.69,0.65,0.93]])
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
    gains_ur = 0.02*gains  # 2% random uncertainty
    gains_us = 0.03*gains  # 3% systematic uncertainty
    dark_ur = np.ones((10,5))*0.02  # random uncertainty of 0.02

    if __name__ == "__main__":

        prop=punpy.LPUPropagation(parallel_cores=4)
        L1=calibrate_slow(L0,gains,dark)
        t1=time.time()

        L1_ur,L1_corr_r=prop.propagate_random(calibrate_slow,[L0,gains,dark],
                        [L0_ur,gains_ur,dark_ur],
                        return_corr=True,corr_axis=1,repeat_dims=0)

        t2=time.time()

        print(L1)
        print(L1_ur)
        print("propogate_random took: ",t2-t1," s")

There is another useful option that allows some input quantities to have repeated axis, whereas other ones do not.
This also results in not all input quantities needing to have the same shape. For example, if we had 10 repeated measurements for L0,
but only one set of gains, and one dark measurement. In that case the keyword `param_fixed` would be set to False for L0 and True for 
gains and dark, as in the examples below::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98],
                  [0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86],
                  [0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.43,0.76,0.67,0.64,0.98],
                  [0.42,0.78,0.69,0.65,0.93]])
   dark = np.random.rand(5)*0.05
   gains = np.array([23,26,28,29,31]) # same gains as before, but repeated 10 times so that shapes match

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
   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones(5)*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],param_fixed=[False,True,True],
                   return_corr=True,repeat_dims=0,corr_axis=1)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [None,gains_us,None],param_fixed=[False,True,True],
                   return_corr=True,repeat_dims=0,corr_axis=1)
   
   print(L1)
   print(L1_ur)
   print(L1_us)


Finally, there is one more important functionality that is showcased in the next example.
As mentioned above, random uncertainties are always uncorrelated with respect to repeated measurements.
And systematic uncertainties are always fully correlated along the repeated dimension (specified in repeat_dims).
However, when there is more than one dimension as is the case here, it is possible that for example the 
systematic uncertainties are not correlated along the wavelength dimension (while still being correlated along repeat_dims).
Therefor, there is a keyword `corr_x` that allows to give the correlation along the non-repeated axis for each input quantity.
For corr_x, it is possible to specify a custom correlation matrix. This correlation matrix applies to each of the repeated measurements.
From this the covariance is than calculated together with the specified uncertainties. This means that even though the correlation 
matrix is the same for each repeated measurement, the covariances for each measurement will be different since the 
uncertainties for each repeated measurement are different. Note also that if a correlation matrix is specified, but the 
uncertainties are set to zero or None, no uncertainty will be added (see L0 in propagate_systematic in example below).

Alternatively, it is possible to set the `corr_x` keyword to one of two strings or None. It can be set to "rand", which is 
equivalent to setting the corr_x for that input quantitiy to np.eye (though using "rand" is faster).
Setting it to "syst" is equivalent to using a corr_x for that input quantity equal to np.ones.
When it is set to None, it defaults to "rand" for propagate_random and "syst" for propagate_systematic.
In the below example we could thus have set "rand" in propagate_random to None without difference::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98],
                  [0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86],
                  [0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.43,0.76,0.67,0.64,0.98],
                  [0.42,0.78,0.69,0.65,0.93]])
   dark = np.random.rand(5)*0.05
   gains = np.array([23,26,28,29,31]) # same gains as before, but repeated 10 times so that shapes match

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

   L0_corr=np.array(
   [[1.        , 0.69107369, 0.5976143 , 0.44946657, 0.16265001],
   [0.69107369, 1.        , 0.56639386, 0.5990423 , 0.20232566],
   [0.5976143 , 0.56639386, 1.        , 0.48349378, 0.17496355],
   [0.44946657, 0.5990423 , 0.48349378, 1.        , 0.13159034],
   [0.16265001, 0.20232566, 0.17496355, 0.13159034, 1.        ]])

   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones(5)*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],
                   [L0_ur,gains_ur,dark_ur],corr_x=[L0_corr,None,"rand"],
                   param_fixed=[False,True,True],return_corr=True,
                   repeat_dims=0,corr_axis=1)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],
                   [None,gains_us,None],corr_x=[L0_corr,None,"rand"],
                   param_fixed=[False,True,True],return_corr=True,
                   repeat_dims=0,corr_axis=1)
   
   print(L1)
   print(L1_ur)
   print(L1_us)


The combination of these different options allow us to propagate uncertainties with nearly any shape or correlation.

3D input quantities and measurand
###################################
Punpy can also deal with input data in 3D (though not with any dimensions higher than that).
This kind of data we get when for example analysing images with spectra or multiband data in every pixel.
The processing is very similar to above. The different pixels can often all be considered repeated measurements (systematic uncertainties are common to all pixels).
In this case, the `repeat_dims` keyword can be set to a list of multiple dimensions as in the example below for a 3-by-3 pixel image with 5 wavebands::


   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98]],
                  [[0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86]],
                  [[0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.42,0.78,0.69,0.65,0.93]]])
   dark = np.random.rand(5)*0.05
   gains = np.array([23,26,28,29,31]) # same gains as before, but repeated 10 times so that shapes match

   # your uncertainties
   L0_ur = np.array([[[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]]])

   L0_corr=np.array(
   [[1.        , 0.69107369, 0.5976143 , 0.44946657, 0.16265001],
   [0.69107369, 1.        , 0.56639386, 0.5990423 , 0.20232566],
   [0.5976143 , 0.56639386, 1.        , 0.48349378, 0.17496355],
   [0.44946657, 0.5990423 , 0.48349378, 1.        , 0.13159034],
   [0.16265001, 0.20232566, 0.17496355, 0.13159034, 1.        ]])

   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones(5)*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],corr_x=[L0_corr,None,"rand"],param_fixed=[False,True,True],return_corr=True,repeat_dims=[0,1],corr_axis=2)
   L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],[None,gains_us,None],corr_x=[L0_corr,None,"rand"],param_fixed=[False,True,True],return_corr=True,repeat_dims=[0,1],corr_axis=2)
   
   print(L1)
   print(L1_ur)
   print(L1_us)


It is also still possible to do the processing without the additional keywords if all input quantities have the same shape.
This will give similar uncertainties to the above, but will use more memory and result in different correlation between wavelengths 
(in the example below there is no correlation for random and full correlation for systematic)::

   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98]],
                  [[0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86]],
                  [[0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.42,0.78,0.69,0.65,0.93]]])
   dark = np.random.rand(3,3,5)*0.05
   gains = np.tile(np.array([23,26,28,29,31]),(3,3,1)) # same gains as before, but repeated 10 times so that shapes match

   # your uncertainties
   L0_ur = np.array([[[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]]])

   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones((3,3,5))*0.02  # random uncertainty of 0.02

   prop=punpy.MCPropagation(10000)
   L1=calibrate(L0,gains,dark)
   L1_ur=prop.propagate_random(calibrate,[L0,gains,dark],
         [L0_ur,gains_ur,dark_ur])
   L1_us=prop.propagate_systematic(calibrate,[L0,gains,dark],
         [None,gains_us,None])

   print(L1)
   print(L1_ur)
   print(L1_us)

And it is still possible to use the LPU methods (with or without repeat_dims)::


   import numpy as np
   import punpy

   # your measurement function
   def calibrate(L0,gains,dark):
      return (L0-dark)*gains

   # your data
   L0 = np.array([[[0.43,0.80,0.70,0.65,0.90],
                  [0.41,0.82,0.73,0.64,0.93],
                  [0.45,0.79,0.71,0.66,0.98]],
                  [[0.42,0.83,0.69,0.64,0.88],
                  [0.47,0.75,0.70,0.65,0.78],
                  [0.45,0.86,0.72,0.66,0.86]],
                  [[0.40,0.87,0.67,0.66,0.94],
                  [0.39,0.80,0.70,0.65,0.87],
                  [0.42,0.78,0.69,0.65,0.93]]])
   dark = np.random.rand(5)*0.05
   gains = np.array([23,26,28,29,31]) # same gains as before, but repeated 10 times so that shapes match

   # your uncertainties
   L0_ur = np.array([[[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]],
                     [[0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06],
                     [0.02, 0.04, 0.02, 0.01, 0.06]]])

   L0_corr=np.array(
   [[1.        , 0.69107369, 0.5976143 , 0.44946657, 0.16265001],
   [0.69107369, 1.        , 0.56639386, 0.5990423 , 0.20232566],
   [0.5976143 , 0.56639386, 1.        , 0.48349378, 0.17496355],
   [0.44946657, 0.5990423 , 0.48349378, 1.        , 0.13159034],
   [0.16265001, 0.20232566, 0.17496355, 0.13159034, 1.        ]])

   gains_ur = 0.02*gains  # 2% random uncertainty
   gains_us = 0.03*gains  # 3% systematic uncertainty 
   dark_ur = np.ones(5)*0.02  # random uncertainty of 0.02

   if __name__ == "__main__":
        prop=punpy.LPUPropagation(parallel_cores=4)
        L1=calibrate(L0,gains,dark)
        L1_ur,L1_corr_r=prop.propagate_random(calibrate,[L0,gains,dark],[L0_ur,gains_ur,dark_ur],corr_x=[L0_corr,None,"rand"],param_fixed=[False,True,True],return_corr=True,repeat_dims=[0,1],corr_axis=2)
        L1_us,L1_corr_s=prop.propagate_systematic(calibrate,[L0,gains,dark],[None,gains_us,None],corr_x=[L0_corr,None,"rand"],param_fixed=[False,True,True],return_corr=True,repeat_dims=[0,1],corr_axis=2)
   
        print(L1)
        print(L1_ur)
        print(L1_us)

