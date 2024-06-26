.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _punpy_standalone:

Punpy as a standalone package
======================================
In this section we give a short overview of some of the key capabilities of punpy for propagating uncertainties through a measurement function.
When using punpy as a standalone tool, the input quantities and their uncertainties are manually specified.
The code in this section is just as illustration and we refer to the the CoMet website `examples <https://www.comet-toolkit.org/examples/>`_ for examples with all required information for running punpy.
The punpy package can propagate various types of uncertainty through any python measurement function. In the next subsection we discuss how these python measurement functions are defined.

Measurement functions
#######################

The measurement function can be written mathematically as:

.. math:: y = f\left( x_{i},\ldots,\ x_{N} \right)

where:

*  :math:`f` is the measurment function;
*  :math:`y` is the measurand;
*  :math:`x_{i}` are the input quantities.

The measurand and input quantities are often vectors consisting of multiple numbers. E.g. in spectroscopy, the input quantities and measurand each have measurements for multiple wavelengths. These wavelengths are the same for the input quantities and the measurand.
We refer to the :ref:`atbd` section for more details on the vocabulary used and the various types of uncertainty.

When using punpy, the measurement function can be any python function that takes a number of input quantities as arguments (as floats or numpy arrays of any shape) and returns a measurand (as a float or numpy array of any shape).
For example::

   def measurement_function(x1,x2,x3):
      y=x1+x2-x3 # here any real measurement function can be implemented
      return y

Any function that can be formatted to take only numpy arrays or floats/ints as its inputs and only returns one or more numpy arrays and or floats/ints can be used in punpy.
Note that this measurement function can optionally be part of a class, which can allow to do an additional setup step to set further, non-numerical parameters::

   class Func:
      def __init__(self,option1,option2):
         self.option1 = option1
         self.option2 = option2

      def measurement_function(self,x1,x2,x3):
         if self.option1=="green":
            y=x1+x2-x3 # here any real measurement function can be implemented
         else:
            y=self.option2
         return y


Monte Carlo and Law of Propagation of Uncertainty methods
##########################################################
Once this kind of measurement function is defined, we can use the various punpy methods
to propagate uncertainties though this measurement function. In order to do this, we
first create a prop object (object of punpy MCPropagation of LPUPropagation class)::

   import punpy

   prop=punpy.MCPropagation(10000) # Here the number is how 
   # many MC samples will be generated

   # Or if you have a measurement function that does not accept 
   # higher dimensional arrays as argument:
   prop=punpy.MCPropagation(10000,parallel_cores=1)

   # Alternatively it is possible to use LPU methods to 
   # propagate uncertainties
   prop=punpy.LPUPropagation()

In order to do propagate uncertainties, punpy uses Monte Carlo (MC) methods (see
Section :ref:`Monte Carlo Method`)
or Law of Propagation of Uncertainties (LPU) methods (see Section :ref:`LPUMethod`).
MC methods generate MC samples from the input quantities (which can be correlated to eachother or not),
and then propagate these samples through the measurement function.
The LPU methods implement the law of propagation of uncertainties from the 
GUM (Guide to the Expression of Uncertainty in Measurement) by calculating the Jacobian and using this to propagate the uncertainties.


For the MC method, the number of MC samples that is used is set as the first argument when creating the MCPropagation object (see example above).
Two approaches can be followed to propagate the MC samples of the input quantities to the measurand, depending on whether the measurement function can be applied to numpy arrays of arbitrary size.

The most computationally efficient way is to pass an array consisting of all MC steps of an
input quantity instead of the input quantity themselves. Each of the input quantities will thus get an additional dimension in the MC sample.
If the measurement function can deal with these higher dimensional arrays by just performing numpy operations, this gives the most computationally efficient MC propagation.
This method is the default in punpy (which corresponds to setting the optional `parallel_cores` keyword to 0).

However, this is not the case for every measurement function. If the inputs to the measurement
function are less flexible, and don't support additional dimensions, it is possible to instead run the MC samples one by one.
In order to pass each MC sample individually to the measurement function, it is possible to set the optional
`parallel_cores` keyword to 1. In :ref:`punpy_memory_and_speed`, we will show how the same keyword can be used to do parallel processing for such measurement functions.


For the LPU methods, the numdifftools package (used within comet_maths) is used to calculate the Jacobian. This package automatically determines the stepsize in the numerical
differentiation, unless a manual stepsize is set. For some measurement functions, it can be necessary to set a manual stepsize (e.g. because of the limited
range of the input quantities). It is possible to set the stepsize to be passed to
the numdifftools jacobian method by setting the `step` keyword when creating the propagation object::

   prop = punpy.LPUPropagation(step=0.01)

Both the MC and LPU options also have the `verbose` keyword to set the verbosity, which allows the user to get additional information such as the time the runs took and additional warnings::

   prop=punpy.MCPropagation(10000,verbose=True)


Propagating random and systematic uncertainties
################################################
Once a prop object has been made (see previous sections), a number of methods can be used to propagate uncertainties, depending on the kind of uncertainties that need to be propagated.
We start by showing how to propagating random and systematic uncertainties.
For random uncertainties, the errors associated with these uncertainties are entirely independent of each-other (errors for different elements of the input quantity are uncorrelated).
For systematic uncertainties, the errors of the different elements (along the different dimensions of the input quantity) are entirely correlated. This typically means they are all affected by the same effect (e.g. if the calibration gain of a sensor is wrong, all its measurements will be wrong by the same factor).

For given values (arrays or numbers) of the input quantities xn, and their random (ur_xn)
or systematic (us_xn) uncertainties, punpy can be used to propagate these uncertainties as follows::

   y = measurement_function(x1, x2, x3)
   ur_y = prop.propagate_random(measurement_function, 
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3])
   us_y = prop.propagate_systematic(measurement_function, 
          [x1, x2, x3], [us_x1, us_x2, us_x3])

In addition to returning the uncertainties, punpy can also be used to return the correlation matrix.
This is not particularly useful when the input quantities all have a random or all have a systematic error correlation as in this section, but is very relevant when dealing with input quantities that have other error correlations (see next section).
This is done by setting the `return_corr` keyword to True::

   ur_y, corr_y = prop.propagate_random(measurement_function,
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3], return_corr=True)

Here, the returned error correlation matrix will cover all dimensions of the associated uncertainty. If ur_y has shape (k,l), corr_y has shape (k*l,k*l).
The order of the correlation coefficient elements corresponds to the order for a flattened version of ur_y (ur_y.flatten()).
By default, the returned correlation matrices are forced to be positive-definite. This means they are sometimes slightly changed in order to accomodate this. If this is the case, a warning is shown.
To just return the unmodified correlation matrices, it is possible to set the `PD_corr` keyword to False.


Propagating uncertainties when measurements are correlated (within input quantity)
###################################################################################
Sometimes the elements of an input quantity xn are not simply independent (random uncertainties) or fully correlated (systematic uncertainty), but rather something in between.
In this case, it is possible to specify an error-correlation matrix between the different elements (at different coordinates/indices) of the input quantity, which gives the correlation coefficient between the errors for different elements within the input quantities.
If the input quantity is one-dimensional of size (k), the error correlation matrix will be a matrix of shape (k,k). If the input quantity is two dimensional of size (k,l), the error correlation matrix will be of size (k*l,k*l).

This error correlation matrix cannot be calculated from the uncertainties themselves (it is not the correlation between the values of the uncertainties or something like that) but instead required knowledge of how the measurements were done and the sensors used.
It is a matrix that needs to be provided. Fur more detailed information on error correlation matrices, we refer to the `Guide to the expression
of uncertainty in measurement <https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf>`_ and the `FIDUCEO website <https://www.fiduceo.eu>`_.

If such an error-correlation matrix corr_xn is known for every xn, punpy can use them to propage the combined uncertainty::

   uc_y, corrc_y = prop.propagate_standard(measurement_function,
                   [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=[corr_x1, corr_x2, corr_x3])

Here the corr_xn can either be a square array with the appropriate error-correlation coefficients, or alternatively the string "rand" or "syst" for random and systematic error correlations respectively.
In the case of random or systematic error-correlations, the error correlation matrices are always the same (diagonal matrix of ones and full matrix of ones for random and systematic respectively), and the string is thus sufficient to define the complete error correlation matrix.
Note that these error-correlation matrices can also optionally be passed to the propagate_random() and propagate_systematic() functions.
In this case, the only difference with propagate_standard, is that in case no error_correlation matrix is provided (i.e. None), the error correlation matrix defaults to the random or systematic case.
The following four expressions are entirely equivalent::

  uc_y, corrc_y = prop.propagate_standard(measurement_function,
                   [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=[np.eye(us_x1.flatten()), corr_x2, np.eye(us_x1.flatten())])
  uc_y, corrc_y = prop.propagate_standard(measurement_function,
                   [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=["rand", corr_x2, "rand"])
  uc_y, corrc_y = prop.propagate_random(measurement_function,
                   [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=[np.eye(us_x1.flatten()), corr_x2, np.eye(us_x1.flatten())], return_corr=True)
  uc_y, corrc_y = prop.propagate_random(measurement_function,
                   [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=[None, corr_x2, None], return_corr=True)


Instead of working with error-correlation matrices, it is also possible to use error covariance matrices.
It is straightforward to convert between error correlation and covariance matrices using comet_maths::

  import comet_maths as cm
  cov_x1 = cm.convert_corr_to_cov(corr_x1, us_x1)

  #and back to corr and uncertainty:
  corr_x1 = cm.convert_cov_to_corr(cov_x1, us_x1)
  corr_x1 = cm.correlation_from_covariance(cov_x1)
  us_x1 = cm.uncertainty_from_covariance(cov_x1)

Using covariance matrices, the uncertainties can be propagated using::

  uc_y, corr_y = prop.propagate_cov(measurement_function, [x1, x2, x3],
                  [cov_x1, cov_x2, cov_x3])

If required, the resulting measurand correlation matrix can easily be converted to a covariance matrix as::

   cov_y = cm.convert_corr_to_cov(corr_y, u_y)

Note that propagate_standard() and propagate_cov() by default return the correlation matrix, yet propagate_random() and propagate_systematic()
return only the uncertainties on the measurand (because the correlation matrices are trivial in their standard use case).
However all these functions have an optional `return_corr` argument that can be used to define whether the correlation matrix should be returned.


Propagating uncertainties when input quantities are correlated (between different input quantities)
###################################################################################################
In addition to the elements within an input quantity being correlated, it is also possible the input quantities are correlated to eachother.
If this is the case, this functionality can be included in each of the functions specified above by giving an argument to the optional keyword `corr_between`.
This keyword needs to be set to the correlation matrix between the input quantities, and thus needs to have the appropriate shape (e.g. 3 x 3 array for 3 input quantities)::

   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], 
          [ur_x1, ur_x2, ur_x3], corr_between = corr_x1x2x3)
   uc_y, corr_y = prop.propagate_cov(measurement_function, [x1, x2, x3], 
                  [cov_x1, cov_x2, cov_x3], corr_between = corr_x1x2x3)

More advanced punpy input and output
######################################
When returning the error correlation functions, rather than providing this full correlation matrix, it is also possible to get punpy to only calculate the error correlation matrix along one (or a list of) dimensions.
If `return_corr` is set to True, the keyword `corr_dims` can be used to indicate the dimension(s) for which the correlation should be calculated.
In this case the correlation coefficients are calculated using only the first index along all dimensions other than `corr_dims`.
We note that `corr_dims` should only be used when the error correlation matrices do not vary along the other dimension(s).
A warning is raised if any element of the correlation matrix varies by more than 0.05 between using only the first index along
all dimensions other than `corr_dims` or using only the last index along all dimensions other than `corr_dims`.
The `corr_dims` keyword can be passed to any of the propagate_* functions::

   ur_y, corr_y = prop.propagate_random(measurement_function,
          [x1, x2, x3], [ur_x1, ur_x2, ur_x3], return_corr=True, corr_dims=0)

If ur_y again had shape (k,l), corr_y would now have shape (k,k).

For the MC method, it is also possible to return the generated samples by setting the optional `return_samples` keyword to True::

   prop = punpy.MCPropagation(10000)
   ur_y, samplesr_y, samplesr_x = prop.propagate_random(
   measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3],
   corr_between=corr_x1x2x3, return_samples=True)

   ub_y, corr_y, samplesr_y, samplesr_x = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3],
   return_corr=True, return_samples=True)

It is also possible to pass a sample of input quantities rather than generating a new MC sample.
This way, the exact same sample can be used as n a previous run, or one can generate a sample manually::

   ub_y, corr_y = prop.propagate_systematic(
   measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3],
   return_corr=True, samples=samplesr_x)

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



Multiple measurands
####################
In some cases, the measurement function has multiple outputs (measurands)::

   def measurement_function(x1,x2,x3):
      y1=x1+x2-x3 # here any real measurement function can be implemented
      y2=x1-x2+x3 # here any real measurement function can be implemented
      return y1,y2

These functions can still be handled by punpy, but require the `output_vars` keyword to be set to the number of outputs::

   us_y, corr_y, corr_out = prop.propagate_systematic(measurement_function,
                            [x1, x2, x3], [us_x1, us_x2, us_x3], 
                            return_corr=True, corr_dims=0, output_vars=2)

Note that now the `corr_dims` keyword is set to a list with the corr_dims for each output variable, and there is an additional output corr_out, which gives the correlation between the different output variables (in the above case a 2 by 2 matrix).
Here the correlation coefficients between the 2 variables are averaged over all measurements. 

When there are multiple output variables, it is also possible to specify separate corr_dims for each measurand.
This is done by setting the `separate_corr_dims` keyword to True, and passing a list of corr_dims::

   us_y, corr_y, corr_out = prop.propagate_systematic(measurement_function,
                            [x1, x2, x3], [us_x1, us_x2, us_x3],
                            return_corr=True, corr_dims=[0,1],separate_corr_dims=True, output_vars=2)

It is also possible to set one of the separate `corr_dims` to None if you do not want the error correlation to be calculated for that measurand. In that case None will be returned (as corr_y[1] in below example)::

   us_y, corr_y, corr_out = prop.propagate_systematic(measurement_function,
                            [x1, x2, x3], [us_x1, us_x2, us_x3],
                            return_corr=True, corr_dims=[0,None],separate_corr_dims=True, output_vars=2)


Different Probability Density Functions
#########################################
The standard probability density function in punpy is a Gaussian distribution.
This means the generated MC samples will follow a gaussian distribution with the input quantity values as mean and uncertainties as standard deviation.
However other probabilty density functions are also possible.
Currently there are two additional options (with more to follow in the future).

The first alternative is a truncated Gaussian distribution. This distribution is just like the Gaussian one, except that there are no values outside a given minimum or maximum value.
A typical use case of this distribution is when a certain input quantity can never be negative.
In such a case the uncertainty propagation could be done like this::

   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3],
          [ur_x1, ur_x2, ur_x3], corr_between = corr_x1x2x3, pdf_shape="truncated_gaussian", pdf_params={"min":0.})

When the alternative probability density functions require additional parameters, these can be passed in the optional pdf_params dictionary.
For the truncated Gaussian example, this dictionary can contain a value for "min" and "max" for the minimum and maximum allowed values respectively.

The second alternative is a tophat distribution. In this case the MC sample will have a uniform probabilty distribution from the value of the input quantity minus its uncertainty to the value of the input quantity plus its uncertainty.
We note that for these modified probability density functions, the standard deviation of the MC sample is not the same as the uncertainty anymore.

Handling errors in the measurement function
############################################
There are cases where a measurement function occasionally runs into an error (e.g. if certain specific combinations of input quantities generated by the MC approach are invalid).
This can completely mess up a long run even if it happens only occasionally.
In cases where the measurement function does not raise an error but returns a measurand which has only non-finite values (np.nan or np.inf as one of the values), that MC sample of the measurand will automatically be ignored by punpy and not used when calculating the uncertainties or any of the other outputs.

In cases where an error is raised, one can catch this error using a try statement and instead return np.nan.
punpy will ignore all these nan's in the measurand MC sample, and will just calculate uncertainties and its other output without these nan samples::

   import numpy as np
   def function_fail(x1, x2):
      zero_or_one=np.random.choice([0,1],p=[0.1,0.9])
      with np.errstate(divide='raise'):
         try:
           return 2 * x1 - x2/zero_or_one
         except:
           return np.nan

   prop = punpy.MCPropagation(1000)
   ur_y = prop.propagate_random(function_fail, [x1, x2], [ur_x1, ur_x2])

Here the measurement will fail about 10% of the time (by raising a FloatingPointError due to division by zero).
The resulting sample of valid measurands will thus have about 900 samples, which should still be enough to calculate the uncertainties.

By default, numpy will only ignore MC samples where all the values are non-finite.
However, it is also possible to ignore all MC samples where any of the values are non-finite.
This can be done by setting the `allow_some_nans` keyword to False.


Shape of input quanties within the measurement function
##########################################################
When setting parallel_cores to 1 or more, the shape of the input quantities used for each iteration in the measurement function matches the shape of the input quantities themselves.
However, when parallel_cores is set to 0, all iterations will be processed simultaneously and there will be an additional dimension for the MC iterations.
Generally within punpy, the MC dimension in the samples is the first one (i.e. internally as well as when MC samples are returned).
However, when processing all iterations simultaniously, in most cases it is more practical to have the MC dimension as the last dimension.
This is because we use numpy arrays and these are compatible when the last dimensions match following the `numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>`_.
So as default, the shape of the input quantities when using parallel_cores will have the MC iterations as its last dimension.
However, in some cases it is more helpful to have the MC iterations as the first dimension.
If this is the case, the MC iteration dimension can be made the first dimension by setting the `MClastdim` keyword to False::

      prop = punpy.MCPropagation(1000,MCdimlast=False)
