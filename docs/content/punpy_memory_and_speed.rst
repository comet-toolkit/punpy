.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _punpy_memory_and_speed:

Managing memory and processing speed in punpy
==============================================

Input quantities with repeated measurements along one axis
###############################################################

In general, random uncertainties are uncorrelated between repeated measurements, and systematic 
uncertainties are fully correlated between repeated measurements. 
If the input quantities are arrays and no further information is specified, punpy assumes that all the different
values in the array are repeated measurements, and the correlation between the values is treated accordingly.

However, it is also possible that the arrays provided in the input quantities have multiple dimensions, 
one of which is for repeated measurements, and one is another dimension. E.g. when propagating uncertainties 
in spectra, often one of the input quantities is a 2D array where along one dimension there are repeated 
measurements and along another there are different wavelengths. In this case the `repeat_dims` keyword can 
be set to an integer indicating which dimension has repeated measurements and the `corr_x` keyword can be 
set to indicate for each input quantity the correlation matrix along the other dimension (wavelength in the above example). 
When the `repeat_dims` keyword is set, punpy also splits the calculations and does them separately per repeated measurement.
This significantly reduces the memory requirements and as a result speeds up the calculations. It is however possible that 
not all of the input quantities have repeated measurements. E.g. one of the input quantities could be an array of three 
calibration coefficients, whereas another input quantity is an array with repeated spectral measurements which are being calibrated.
If the `repeat_dims` keyword does not apply to one of the input quantities, this can be specified by the `param_fixed` keyword. 
This keyword then needs to be set to a list of bools where each bool indicates whether the corresponding input quantity 
should remain fixed (True) or should be split along repeat_dims (False).

If `return_corr` is set to True, the keyword `corr_axis` can be used to indicate along which axis the correlation should be 
calculated (this is typically the other dimension to the repeat_dims one). If x1, x2, us_x1, us_x2 are all 
arrays with shape (n_wav,n_repeats) where n_wav is the number of wavelengths and n_repeats is the number of repeated 
measurements, and x3 is an array with some calibration coefficients (with uncertainties u_x3)::
	
   import numpy as np

   corr_wav_x1= np.eye(len(wavelengths))  # This is a diagonal (i.e. 
   # uncorrelated) correlation matrix with shape (n_wav,n_wav) where 
   # n_wav is the number of wavelengths.
   
   corr_wav_x2= np.ones((len(wavelengths),len(wavelengths))  # This is
   # a correlation matrix of ones (i.e. fully correlated) with shape 
   #(n_wav,n_wav) where n_wav is the number of wavelengths.
   
   corr_wav_x3= None  # When set to None, the correlation between
   # wavelength defaults to the same as the correlation between repeated 
   # wavelengths (i.e. fully correlated for propagate_systematic()).

   param_fixed_x1x2x3 = [False,False,True] # indicates that x1 and x2 
   # have repeated measurements along repeat_dims and calculations will  
   # be split up accordingly, and x3 will remain fixed and not split up  
   # (x3 does not have the right shape to be split up)

   us_y, corr_y = prop.propagate_systematic(measurement_function, 
                  [x1, x2, x3], [us_x1, us_x2, us_x3], 
                  corr_x=[corr_wav_x1,corr_wav_x2,corr_wav_x3], 
                  param_fixed=param_fixed_x1x2x3, fixed return_corr=True, 
                  repeat_dims=1, corr_axis=0)

Here only one matrix is returned for corr_y, rather than a correlation matrix per repeated measurement. The matrices for each repeated measurement have been averaged.
It is also possible to set `corr_axis` without the need for `repeat_dims` to be set. In this case the correlation coefficients will be averaged over all dimensions other than `corr_axis`.
Another important option is that the `corr_x` for each input quantitty can not only be set to None or a custom correlation matrix, but also to the strings "rand" or "syst". For
"rand" these is no error correlation along the non-repeated dimension and for "syst" the errors along the non-repeated dimension are fully correlated. 
In the above code, we could have thus used "rand" and "syst" instead of corr_wav_x1 and corr_wav_x2 respectively, which would in fact have made the calculation slightly faster.

Processing the MC samples in parallel
######################################

At the start of this section we already saw that the optional `parallel_cores` keyword can be used to running the MC
samples one-by-one through the measurement function rather than all at once as in the standard case. It is also possible
to use the same keyword to use parallel processing. Here, only the processing of the input quantities through the measurement
function is done in parallel. Generating the samples and calculating the covariance matrix etc is still done as normal.
Punpy uses the multiprocessing module which comes standard with your python distribution.
The gain by using parallel processing only really outweighs the overhead if the measurement function is relatively slow
(of the order of 0.1 s or slower for one set of input quantities).

Parallel processing for MC can be done as follows::

   if __name__ == "__main__":
      prop = punpy.MCPropagation(10000,parallel_cores=4)
      ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], 
             [ur_x1, ur_x2, ur_x3])
      us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], 
             [us_x1, us_x2, us_x3])

Note that the use of 'if __name__ == "__main__":' is required when using a Windows machine for multiprocessing and is generally good practise.
When processing in parallel, child processes are generated from the parent code, and the above statement is necessary in Windows to avoid the child processes to generate children themselves.
Everything using the results of the multiprocessing needs to be inside the 'if __name__ == "__main__"'.
However the measurement function itself needs to be outside this since the child processes need to find this.

For the LPU method, it is also possible to use parallel processing, though only if the `repeat_dims` keyword is set.
In this case each of the repeated measurements is processed in parallel::

   if __name__ == "__main__":
      prop = punpy.LPUPropagation(parallel_cores=4)
      ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], 
             [ur_x1, ur_x2, ur_x3],repeat_dims=0)
      us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], 
             [us_x1, us_x2, us_x3],repeat_dims=0)

Separating MC propagation in different stages
###############################################

Parallel processing for MC can be done as follows::

   if __name__ == "__main__":
      prop = punpy.MCPropagation(10000,parallel_cores=4)
      ur_y = prop.propagate_random(measurement_function, [x1, x2, x3],
             [ur_x1, ur_x2, ur_x3])
      us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3],
             [us_x1, us_x2, us_x3])

Note that the use of 'if __name__ == "__main__":' is required when using a Windows machine for multiprocessing and is generally good practise.
When process

Additional options
#####################

For both methods there are some cases, when there is only one correlation matrix contributing to the measurand (e.g. a complicated
measurement function where all but one of the input quantities are known with perfect precision, i.e. without uncertainty),
it can be beneficial to just copy this correlation matrix to the measurand rather than calculating it (since copying is faster
and does not introduce MC noise). When the `fixed_corr_var` is set to True, punpy automatically detects if there is only one
term of uncertainty, and if so copies the relevant correlation matrix to the output instead of calculating it. If `fixed_corr_var`
is set to an integer, the correlation matrix corresponding to that dimension is copied instead::

   prop = punpy.MCPropagation(10000)
   ur_y = prop.propagate_random(
   measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3],
   corr_between=corr_x1x2x3, fixed_corr_var=True)
