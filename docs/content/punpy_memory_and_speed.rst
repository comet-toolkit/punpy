.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _punpy_memory_and_speed:

Managing memory and processing speed in punpy
==============================================

Storing error correlation matrices separately per dimension
#############################################################
Random and systematic uncertainty components take up very little space, as each of their error
correlation dimensions are by defnition parameterised as random or systematic.
For structured components with error correlation matrices stored as separate variables, it is not
uncommon for these matrices to take up a lot of memory. This is especially the case when
each of the dimensions is not parametrised separately, and instead an error correlation
matrix is provided along the combination of angles. E.g. for a variable with dimensions (x,y,z),
which correspond to a shape of e.g. (20,30,40), the resulting total error correlation matrix will have shape
(20*30*40,20*30*40) which would contain 575 million elements. The shape chosen here as an example is
quite moderate, so it is clear this could be an issue when using larger datasets.

The solution to this is to avoid storing the full (x*y*z,x*y*z) error correlation matrix when possible.
In many cases, even though the errors for pixels along a certain dimension (e.g. x) might
be correlated, this error correlation w.r.t x does not change for different values of y or z.
In that case, the error correlation for x can be separated and stored as a matrix of shape (x,x).

One way to reduce the memory load is by separately storing the error-correlation matrices in the output dataset.
In the :ref:`_punpy_standalone` section, we showed that the `corr_dims` keyword can be used to output the error correlation matrix for a single dimension rather than the full error correlation matrix.
This can also be used to separately store the error correlation for each dimension by passing a list of all dimension indices for `corr_dims`::

   ur_y, corr_y = prop.propagate_random(measurement_function,
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3], return_corr=True, corr_dims=[0,1])

where now corr_y will have two elements, the first of which will be a (k,k) shaped matrix and the second one a (l,l) matrix.
Saving the error correlations in this manner often required less memory than storing the full error correlation matrix.
However, because of the averaging over the other dimension(s) when calculating these one-dimensional errror correlation matrices, some information in lost.
In fact, this approach should only be done when the error correlation matrices do not vary along the other dimension(s).
Whether this method can be applied thus depends on the measurement function itself and it should be used with caution (a good rule of thumb to decide whether this approach can be used is whether the measurement function could be applied without passing data along the averaged dimension(s) at once).
There are cases where the measurement function can only partially be decomposed (e.g. a measurement function where the first nd second dimensions of the measurand have some complex correlation, but the third dimension can easily be separated out as effectively the calculation could be done entirely separate for each index along this third dimension).
In such cases, the corr_dims keyword for the dimensions that cannot be separted can be given a string with the dimension indices separated by a dot::

  ur_y, corr_y = prop.propagate_random(measurement_function,
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3], return_corr=True, corr_dims=["0.1",2])

Here, if the measurand (and ur_y) are of shape (k,l,m), corr_y will have two elements. The first element will be of shape (k*l,k*l) and the second element of shape (m,m).

We note that for the digital effects table use case, when creating the MeasurementFunction object, it is possible to provide the `corr_dims` keyword as strings with the dimension names rather than dimension indices (both options work).
When using dimension names as strings, they can still be separated by a dot to indicate combined error correlation matrices in the outputs::

   gl = GasLaw(prop, ["n_moles", "temperature", "pressure"], corr_dims=["x.y","time"])

Using error correlation dictionaries
######################################
In addition to the large error correlation matrices in the output, another memory issue comes from the calculation of the error-correlation for each of the input quantities (which often have the same dimensions).
When using punpy as standalone, one can pass the error correlation for separate dimensions using a dictionary::

        ufb, ucorrb = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=[{"0":np.eye(len(xerrb)),"1":"syst"}, "syst"],
            return_corr=True,
        )

This avoids having to pass the correlation matrix as one large array.
When using digital effects tables with punpy, the use of these correlation dictionaries can be handled internally.
This can be achieved by setting the use_err_corr_dict keyword::

   gl = IdealGasLaw(
      prop=prop,
      xvariables=["pressure", "temperature", "n_moles"],
      yvariable="volume",
      yunit="m^3",
      use_err_corr_dict=True,
   )

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
be set to an integer indicating which dimension has repeated measurements.
When the `repeat_dims` keyword is set, punpy also splits the calculations and does them separately per repeated measurement.
This reduces the memory requirements and as a result speeds up the calculations. It is however possible that
not all of the input quantities have repeated measurements. E.g. one of the input quantities could be an array of three 
calibration coefficients, whereas another input quantity is an array with repeated spectral measurements which are being calibrated.
If the `repeat_dims` keyword does not apply to one of the input quantities, this can be specified by the `param_fixed` keyword. 
This keyword then needs to be set to a list of bools where each bool indicates whether the corresponding input quantity 
should remain fixed (True) or should be split along repeat_dims (False).

If x1, x2, us_x1, us_x2 are all arrays with shape (n_wav,n_repeats) where n_wav is the
number of wavelengths and n_repeats is the number of repeated
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
                  repeat_dims=1, corr_dims=0)

Here only one matrix is returned for corr_y with a shape matching the provided corr_dims, rather than a correlation matrix per repeated measurement. The matrices for each repeated measurement have been averaged.
We note that if no corr_dims are set, the default option is to return a combined error correlation matrix for all dimensions that are not in repeat_dims.

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

One other important aspect is that in order for the parallel processing to work, the measurement function cannot be a normal function of a class.
It can however be a static function of a class.
This means that if we want to do parallel processing for a measurement function in a punpy MeasurementFunction class in order to use digital effects tables, we need to define it as a static function::

   # Define your measurement function inside a subclass of MeasurementFunction
   class IdealGasLaw(MeasurementFunction):
       @staticmethod
       def meas_function(pres, temp, n):
           return (n * temp * 8.134) / pres

Measurement function for which multiprocessing can be used can thus not have self as their first argument.

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
In some cases, it is necessary to run a large MC sample but the measurement function requires too much memory to run all the MC samples in one single run.
In such cases it is possible to break up the punpy processing in differnet stages. Generally, there are 4 stages:
-  Generating the MC sample of the input quantities.
-  Running these samples through the measurement function.
-  Combining the MC samples of measurands.
-  Processing the MC measurand sample to produce the required outputs (sush as uncertainties and error correlation matrices).

In code, this looks like::

   MC_x = prop.generate_MC_sample(xsd, xerrsd, corrd)
   MC_y1 = prop.run_samples(functiond, MC_x, output_vars=2, start=0, end=10000)
   MC_y2 = prop.run_samples(functiond, MC_x, output_vars=2, start=10000, end=20000)
   MC_y = prop.combine_samples([MC_y1, MC_y2])

   ufd, ucorrd, corr_out = prop.process_samples(
      MC_x, MC_y, return_corr=True, corr_dims=0, output_vars=2
   )

Here the run has been broken up into two seperate calls to run the samples, which can be controlled by specifying the start and end indices of the MC sample of input quantities (i.e. which MC iterations should be processed by this call).
This can be broken up into any number of samples. The runnning of these samples through the measurand can even be distributed on different computers. The different measurand samples could then simply be stored in files, before bringing them all together and analysing the combined measurand MC sample.
This also allows detailed controll (e.g. quality checks) on the measurand MC samples, prior to processing the samples.

Additional options
#####################
For both MC and LPU methods there are some cases, when there is only one correlation matrix contributing to the measurand (e.g. a complicated
measurement function where all but one of the input quantities are known with perfect precision, i.e. without uncertainty),
it can be beneficial to just copy this correlation matrix to the measurand rather than calculating it (since copying is faster
and does not introduce MC noise). When the `fixed_corr_var` is set to True, punpy automatically detects if there is only one
term of uncertainty, and if so copies the relevant correlation matrix to the output instead of calculating it. If `fixed_corr_var`
is set to an integer, the correlation matrix corresponding to that dimension is copied without any checks::

   prop = punpy.MCPropagation(10000)
   ur_y = prop.propagate_random(
   measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3],
   corr_between=corr_x1x2x3, fixed_corr_var=True)
