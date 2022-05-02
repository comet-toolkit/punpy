.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _using_punpy_standalone:

Using punpy as a standalone package
======================================
In this section we give a short overview of some of the key capabilities of punpy for propagating uncertainties through a measurement function.
When using punpy as a standalone tool, the input quantities and their uncertainties are manually specified.
The code in this section is just as illustration and we refer to the Examples Section for example with all requied information for running punpy.
The punpy package can propagate various types of uncertainty through a given measurement function. In the next subsection we discuss how these measurement functions are defined in python.

Measurement Function
####################

The measurement function can be written mathematically as:

.. math:: y = f\left( x_{i},\ldots,\ x_{N} \right)

where:

-  :math:`f` is the measurment function;
-  :math:`y` is the measurand;
-  :math:`x_{i}` are the input quantities.

The measurand and input quantities are often vectors consisting of multiple numbers. E.g. in spectroscopy, the input quantities and measurand each have measurements for multiple wavelengths. These wavelengths are the same for the input quantities and the measurand. We refer to the 'Principles of Uncertainty Analysis' section below for more details on the vocabulary used and the various types of uncertainty.

When using punpy as a standalone package, the measurement function can be any python function that takes a number of input quantities as arguments (usually arrays) and returns a measurand (usually array).
For example::

   def measurement_function(x1,x2,x3):
      y=x1+x2-x3 # here any real measurement function can be implemented
      return y


Propagating random and systematic uncertainties
################################################
    
Once this kind of measurement function is defined, we can use the various punpy methods to propagate uncertainties though this measurement function. In order to do this, we first create a prop object::

   import punpy

   prop=punpy.MCPropagation(10000) # Here the number is how 
   # many MC samples will be generated

   # Or if you have a measurement function that does not accept 
   # higher dimensional arrays as argument:
   prop=punpy.MCPropagation(10000,parallel_cores=1)

   # Alternatively it is possible to use LPU methods to 
   # propagate uncertainties
   prop=punpy.LPUPropagation()

In order to do propagate uncertainties, punpy uses Monte Carlo (MC) methods (see Section :ref:`Monte Carlo Method`) 
or Law of Propagation of Uncertainties (LPU) methods (see Section :ref:`LPU Method`). MC methods generate MC samples from the input 
quantities (which can be individually correlated or not), and then propagate these samples through the
measurement function. This is typically done by passing an array consisting of all MC steps of an
input quantity instead of the input quantity themselve for each of the input quantities. Here it is assumed
the measurement function can deal with these higher dimensional arrays by just performing numpy operations.
However, this is not always the case. If the inputs to the measurement function are less flexible,
We can instead pass each MC sample individually tothe measurement function by setting the optional
`parallel_cores` keyword to 1. At the end of this section we'll also see how to use this keyword for parallel processing.
The LPU methods implement the law of propagation of uncertainties from the 
GUM (Guide to the Expression of Uncertainty in Measurement) by calculating the Jacobian and using this to propagate the uncertainties.

Once a prop object has been made, a number of methods can then be used to propagate uncertainties, depending on the kind of uncertainties that need to be propagated.
We start by showing how to propagating random and systematic uncertainties.
When given values (arrays or numbers) for the input quantities xn, and their random (ur_xn) 
or systematic (us_xn) uncertainties, punpy can be used to propage these uncertainties as follows::

   y = measurement_function(x1, x2, x3)
   ur_y = prop.propagate_random(measurement_function, 
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3])
   us_y = prop.propagate_systematic(measurement_function, 
          [x1, x2, x3], [us_x1, us_x2, us_x3])

Propagating uncertainties when measurements are correlated (within input quantity)
#################################################################################

Sometimes the elements of an input quantity xn are not simply independent (random uncertainties) or fully correlated (systematic uncertainty), but rather a combination of the two.
In this case, it is possible to specify a covariance matrix cov_xn between all the elements of xn. If such a covariance matrix is known for every xn, punpy can use them to propage the combined uncertainty::

   uc_y, corrc_y = prop.propagate_cov(measurement_function, 
                   [x1, x2, x3], [cov_x1, cov_x2, cov_x3])

Here, in addition to the uncertainties on the measurand, we also provide a correlation matrix between the elements in the measurand.
If required, this correlation matrix can easily be converted to a covariance matrix as::

   cov_y = prop.convert_corr_to_cov(corr_y, u_y)

Note that propagate_cov() by default returns the correlation matrix, yet propagate_random() and propagate_systematic() 
return only the uncertainties on the measurand (because the correlation matrices are trivial in this case).
However these functions have an optional `return_corr` argument that can be used to define whether the correlation matrix should be returned.

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


Propagating uncertainties when input quantities are correlated (between different input quantities)
###################################################################################################

In addition to the elements within an input quantity being correlated, it is also possible the input quantities are correlated to eachother.
If this is the case, this functionality can be included in each of the functions specified above by giving an argument to the optional keyword `corr_between`.
This keyword needs to be set to the correlation matrix between the input quantities, and thus needs to have the appropriate shape (e.g. 3 x 3 array for 3 input quantities)::

   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], 
          [ur_x1, ur_x2, ur_x3], corr_between = corr_x1x2x3)
   uc_y, corr_y = prop.propagate_cov(measurement_function, [x1, x2, x3], 
                  [cov_x1, cov_x2, cov_x3], corr_between = corr_x1x2x3)


Multiple outputs
################

In some cases, the measurement function has multiple outputs::

   def measurement_function(x1,x2,x3):
      y1=x1+x2-x3 # here any real measurement function can be implemented
      y2=x1-x2+x3 # here any real measurement function can be implemented
      return y1,y2

These functions can still be handled by punpy, but require the `output_vars` keyword to be set to the number of outputs::

   us_y, corr_y, corr_out = prop.propagate_systematic(measurement_function,
                            [x1, x2, x3], [us_x1, us_x2, us_x3], 
                            return_corr=True, corr_axis=0, output_vars=2)

Note that now there is an additional output corr_out, which gives the correlation between the different output variables (in the above case a 2 by 2 matrix).
Here the correlation coefficients between the 2 variables are averaged over all measurements. 


Additional options
##################

For the MC method, it is also possible to return the generated samples by setting the optional `return_samples` keyword to True::
	
   prop = punpy.MCPropagation(10000)
   ur_y, samplesr_y, samplesr_x = prop.propagate_random(
   measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3],
   corr_between=corr_x1x2x3, return_samples=True)

   ub_y, corr_y, samplesr_y, samplesr_x = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], 
   return_corr=True, return_samples=True)

For the LPU method, it is possible to additionally return the calculated Jacobian matrix by setting the `return_Jacobian` keyword to True.
In addition, instead of calculating the Jacobian as part of the propagation, it is also possible to give a precomputed Jacobian matrix, by setting the `Jx` keyword.
This allows to use the Jacobian matrix from a previous step or an analytical prescription, which results in much faster processing::

   prop = punpy.LPUPropagation()
   ur_y, Jac_x = prop.propagate_random(
   measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3],
   corr_between=corr_x1x2x3, return_Jacobian=True)

   ub_y, corr_y = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], 
   return_corr=True, Jx=Jac_x)

It is not uncommon to have measurment functions that take a number of input quantities, where each input quantity is a vector or array.
If the measurand and each of the input quantities all have the same shape, and the measurement function is applied independently to each 
element in these arrays, then most of the elements in the Jacobian will be zero (all except the diagonal elements for each square Jacobian
matrix corresponding to each input quantity individually). Rather than calculating all these zeros, it is possible to set the `Jx_diag` keyword 
to True which will automatically ignore all the off-diagonal elements and result in faster processing::

   prop = punpy.LPUPropagation()
   ub_y, corr_y = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], 
   return_corr=True, Jx_diag=True)

For the LPU methods, the numdifftools package is used to calculate the Jacobian. This package automatically determines the stepsize in the numerical
differentiation, unless a manual stepsize is set. For some measurement functions, it can be necessary to set a manual stepsize (because of the limited 
range of the input quantities, or because one of the input quantities has to remain sorted, or ...). It is possible to set the stepsize to be passed to 
the numdifftools jacobian method by setting the `step` keyword when creating the propagation object:

   prop = punpy.LPUPropagation(step=0.01)
   ub_y, corr_y = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], 
   return_corr=True)

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

