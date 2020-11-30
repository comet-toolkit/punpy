.. Overview of method
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _overview_of_method:

Overview of method
======================================

Summary
#########
**Measurement Function**

In this section we give a short overview of some of the key capabilities of punpy for propagating uncertainties through a measurement function.
The code in this section is just as illustration and we refer to the Examples Section for example with all requied information for running punpy.
The punpy package can propagate various types of uncertainty through a given measurement function. 
The measurement function can be written mathematically as:

.. math:: y = f\left( x_{i},\ldots,\ x_{N} \right)

where:

-  :math:`f` is the measurment function;
-  :math:`y` is the measurand;
-  :math:`x_{i}` are the input quantities.

The measurand and input quantities are often vectors consisting of multiple numbers. E.g. in spectroscopy, the input quantities and measurand each have measurements for multiple wavelengths. These wavelengths are the same for the input quantities and the measurand. We refer to the 'Principles of Uncertainty Analysis' section below for more details on the vocabulary used and the various types of uncertainty.

Within punpy the measurement function can be any python function that takes a number of input quantities as arguments (usually arrays) and returns a measurand (usually array).
For example::

    def measurement_function(x1,x2,x3):
	y=x1+x2-x3 # here any real measurement function can be implemented
        return y

**Propagating random and systematic uncertainties** 
    
Once this kind of measurement function is defined, we can use the various punpy methods to propagate uncertainties though this measurement function. In order to do this, we first create a MCPropagation object::

   import punpy

   prop=punpy.MCPropagation(10000) # Here the number is how many MC samples will be generated

   # Or if you have a measurement function that does not accept higher dimensional arrays as argument:
   prop=punpy.MCPropagation(10000,parallel_cores=1)

   #Alternatively it is possible to use Jacobian methods to propagate uncertainties
   prop=punpy.JacobianPropagation()

In order to do propagate uncertainties, punpy uses Monte Carlo (MC) methods (see Section :ref:`Monte Carlo Method`) 
or Jacobian methods (see Section :ref:`Jacobian Method`). MC methods generate MC samples from the input 
quantities (which can be individually correlated or not), and then propagate these samples through the
measurement function. This is typically done by passing an array consisting of all MC steps of an
input quantity instead of the input quantity themselve for each of the input quantities. Here it is assumed
the measurement function can deal with these higher dimensional arrays by just performing numpy operations.
However, this is not always the case. If the inputs to the measurement function are less flexible,
We can instead pass each MC sample individually tothe measurement function by setting the optional
`parallel_cores` keyword to 1. At the end of this section we'll also see how to use this keyword for parallel processing.
The Jacobian methods implement the law of propagation of uncertainties from the 
GUM (Guide to the Expression of Uncertainty in Measurement).

Once a prop object has been made, a number of methods can then be used to propagate uncertainties, depending on the kind of uncertainties that need to be propagated.
We start by showing how to propagating random and systematic uncertainties.
When given values (arrays or numbers) for the input quantities xn, and their random (ur_xn) 
or systematic (us_xn) uncertainties, punpy can be used to propage these uncertainties as follows::

   y = measurement_function(x1, x2, x3)
   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3])
   us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3])

**Propagating uncertainties when measurements are correlated (within input quantity)**

Sometimes the elements of an input quantity xn are not simply independent (random uncertainties) or fully correlated (systematic uncertainty), but rather a combination of the two.
In this case, it is possible to specify a covariance matrix cov_xn between all the elements of xn. If such a covariance matrix is known for every xn, punpy can use them to propage the combined uncertainty::

   uc_y, corrc_y = prop.propagate_cov(measurement_function, [x1, x2, x3], [cov_x1, cov_x2, cov_x3])

Here, in addition to the uncertainties on the measurand, we also provide a correlation matrix between the elements in the measurand.
If required, this correlation matrix can easily be converted to a covariance matrix as::

   cov_y = prop.convert_corr_to_cov(corr_y, u_y)

Note that propagate_cov() by default returns the correlation matrix, yet propagate_random() and propagate_systematic() 
return only the uncertainties on the measurand (because the correlation matrices are trivial in this case).
However these functions have an optional `return_corr` argument that can be used to define whether the correlation matrix should be returned.

**Input quantities with repeated measurements along one axis**

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

   corr_wav_x1= np.eye(len(wavelengths))  		     # This is a diagonal (i.e. uncorrelated) correlation matrix with shape (n_wav,n_wav) where n_wav is the number of wavelengths.
   corr_wav_x2= np.ones((len(wavelengths),len(wavelengths))  # This is a correlation matrix of ones (i.e. fully correlated) with shape (n_wav,n_wav) where n_wav is the number of wavelengths.
   corr_wav_x3= None 					     # When set to None, the correlation between wavelength defaults to the same as the correlation between repeated wavelengths (i.e. fully correlated for propagate_systematic()).
   param_fixed_x1x2x3 = [False,False,True]		     # indicates that x1 and x2 have repeated measurements along repeat_dims and calculations will be split up accordingly, and x3 will remain fixed and not split up (x3 does not have the right shape to be split up)
   us_y, corr_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], corr_x=[corr_wav_x1,corr_wav_x2,corr_wav_x3], param_fixed=, fixed return_corr=True, repeat_dims=1, corr_axis=0)

Here only one matrix is returned for corr_y, rather than a correlation matrix per repeated measurement. The matrices for each repeated measurement have been averaged.
It is also possible to set `corr_axis` without the need for `repeat_dims` to be set. In this case the correlation coefficients will be averaged over all dimensions other than `corr_axis`.
Another important option is that the `corr_x` for each input quantitty can not only be set to None or a custom correlation matrix, but also to the strings "rand" or "syst". For
"rand" these is no error correlation along the non-repeated dimension and for "syst" the errors along the non-repeated dimension are fully correlated. 
In the above code, we could have thus used "rand" and "syst" instead of corr_wav_x1 and corr_wav_x2 respectively, which would in fact have made the calculation slightly faster.


**Propagating uncertainties when input quantities are correlated (between different input quantities)**

In addition to the elements within an input quantity being correlated, it is also possible the input quantities are correlated to eachother.
If this is the case, this functionality can be included in each of the functions specified above by giving an argument to the optional keyword `corr_between`.
This keyword needs to be set to the correlation matrix between the input quantities, and thus needs to have the appropriate shape (e.g. 3 x 3 array for 3 input quantities)::

   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], corr_between = corr_x1x2x3)
   uc_y, corr_y = prop.propagate_cov(measurement_function, [x1, x2, x3], [cov_x1, cov_x2, cov_x3], corr_between = corr_x1x2x3)

**Additional options**

It is also possible to return the generated samples by setting the optional `return_samples` keyword to True::

   ur_y, samplesr_y, samplesr_x = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], corr_between=corr_x1x2x3, return_samples=True)
   ub_y, corr_y, samplesr_y, samplesr_x = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], return_corr=True, return_samples=True)

In some cases, the measurement function has multiple outputs::

    def measurement_function(x1,x2,x3):
	y1=x1+x2-x3 # here any real measurement function can be implemented
        y2=x1-x2+x3 # here any real measurement function can be implemented
        return y1,y2

These functions can still be handled by punpy, but require the `output_vars` keyword to be set to the number of outputs::

   us_y, corr_y, corr_out = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3], return_corr=True, corr_axis=0,output_vars=2)

Note that now there is an additional output corr_out, which gives the correlation between the different output variables (in the above case a 2 by 2 matrix).
Here the correlation coefficients between the 2 variables are averaged over all measurements. 

In some cases, when there is only one correlation matrix contributing to the measurand (e.g. a complicated 
measurement function where all but one of the input quantities are known with perfect precision, i.e. without uncertainty),
it can be beneficial to just copy this correlation matrix to the measurand rather than calculating it (since copying is faster
and does not introduce MC noise). When the `fixed_corr_var` is set to True, punpy automatically detects if there is only one 
term of uncertainty, and if so copies the relevant correlation matrix to the output instead of calculating it. If `fixed_corr_var`
is set to an integer, the correlation matrix corresponding to that dimension is copied instead. 

**Processing the MC samples in parallel**

At the start of this section we already saw that the optional `parallel_cores` keyword can be used to running the MC
samples one-by-one through the measurement function rather than all at once as in the standard case. It is also possible
to use the same keyword to use parallel processing. Here, only the processing of the input quantities through the measurement
function is done in parallel. Generating the samples and calculating the covariance matrix etc is still done as normal.
Punpy uses the multiprocessing module which comes standard with your python distribution.
The gain by using parallel processing only really outweighs the overhead if the measurement function is relatively slow
(of the order of 0.1 s or slower for one set of input quantities).

Parallel processing can be done as follows::

   if __name__ == "__main__":
      prop = punpy.MCPropagation(10000,parallel_cores=4)
      ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3])
      us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3])

Note that the use of 'if __name__ == "__main__":' is required when using a Windows machine for multiprocessing and is generally good practise.
When processing in parallel, child processes are generated from the parent code, and the above statement is necessary in Windows to avoid the child processes to generate children themselves.
Everything using the results of the multiprocessing needs to be inside the 'if __name__ == "__main__"'.
However the measurement function itself needs to be outside this since the child processes need to find this.


Principles of Uncertainty Analysis
###################################

The Guide to the expression of Uncertainty in Measurement (GUM 2008)
provides a framework for how to determine and express the uncertainty of
the measured value of a given measurand (the quantity which is being
measured). The International Vocabulary of Metrology (VIM 2008) defines
measurement uncertainty as:

   *"a non-negative parameter characterizing the dispersion of the
   quantity values being attributed to a measurand, based on the information used."*

The standard uncertainty is the measurement uncertainty expressed as a
standard deviation. Please note this is a separate concept to
measurement error, which is also defined in the VIM as:

   *"the measured quantity value minus a reference quantity value."*

Generally, the "reference quantity" is considered to be the "true value"
of the measurand and is therefore unknown. Figure 1 illustrates these
concepts.

.. image:: images/image1.png

*Figure 1 - Diagram illustrating the different concepts of measured value and true value, uncertainty and error.*
 
Within the GUM framework uncertainty analysis begins with understanding
the measurement function. The measurement function establishes the
mathematical relationship between all known input quantities (e.g.
instrument counts) and the measurand itself (e.g. radiance). Generally,
this may be written as

.. math:: y = f\left( x_{i},\ldots,\ x_{N} \right)

where:

-  :math:`y` is the measurand;

-  :math:`x_{i}` are the input quantities.

Uncertainty analysis is then performed by considering in turn each of
these different input quantities to the measurement function, this
process is represented in Figure 2. Each input quantity may be
influenced by one or more error effects which are described by an
uncertainty distribution. These separate distributions may then be
combined to determine the uncertainty of the measurand,
:math:`u^{2}(Y)`, using the *Law of Propagation of Uncertainties* (GUM,
2008),

.. math:: u^{2}\left( y \right) = \mathbf{\text{cS}}\left( \mathbf{x} \right)\mathbf{c}^{T}

where:

-  :math:`\mathbf{C}` is the vector of sensitivity coefficients,
   :math:`\partial Y/\partial X_{i}`;

-  :math:`\mathbf{S(x)}` is the error covariance matrix for the input
   quantities.


.. image:: images/image2.png

*Figure 2 - Conceptual process of uncertainty propagation.*

In a series of measurements (for example each pixel in a remote sensing
Level 1 (L1) data product) it is vital to consider how the errors
between the measurements in the series are correlated. This is crucial
when evaluating the uncertainty of a result derived from these data (for
example a Level 2 (L2) retrieval of geophysical parameter from a L1
product). In their vocabulary the Horizon 2020 FIDUCEO [1]_ (Fidelity
and uncertainty in climate data records from Earth observations) project
(see FIDUCEO Vocabulary, 2018) define three broad categories of error
correlation effects important to satellite data products, as follows:

-  **Random effects**: *"those causing errors that cannot be corrected
   for in a single measured value, even in principle, because the effect
   is stochastic. Random effects for a particular measurement process
   vary unpredictably from (one set of) measurement(s) to (another set
   of) measurement(s). These produce random errors which are entirely
   uncorrelated between measurements (or sets of measurements) and
   generally are reduced by averaging."*


-  **Structured random effects**: *"means those that across many
   observations there is a deterministic pattern of errors whose
   amplitude is stochastically drawn from an underlying probability
   distribution; "structured random" therefore implies "unpredictable"
   and "correlated across measurements"..."*


-  **Systematic (or common) effects**: *"those for a particular
   measurement process that do not vary (or vary coherently) from (one
   set of) measurement(s) to (another set of) measurement(s) and
   therefore produce systematic errors that cannot be reduced by
   averaging."*

.. [1] See: https://www.fiduceo.eu


.. _Monte Carlo Method
Monte Carlo Method
########################
For a detailed description of the Monte Carlo (MC) method, we refer to `Supplement 1 to the
"Guide to the expression of uncertainty in measurement" — Propagation of distributions
using a Monte Carlo method <https://www.bipm.org/utils/common/documents/jcgm/JCGM_101_2008_E.pdf>`_.

Here we summarise the main steps and detail how these were implemented.
The main stages consist of:

#. Formulation: Defining the measurand (output quantity Y), the input quantities X = (X1, . . . , XN ), 
and the measurement function (as a model relating Y and X). One also needs to asign Probability 
Density Functions (PDF) of each of the input quantities, as well as define the correlation between them 
(through joint PDF).

#. Propagation: propagate the PDFs for the Xi through the model to obtain the PDF for Y.

#. Summarizing: Use the PDF for Y to obtain the expectation of Y, the standard uncertainty u(Y) 
associated with Y (from the standard deviation), and the covariance between the different values in Y.

The MC method implemented in punpy consists of generating joint PDF from the provided 
uncertainties and correlation matrices or covariances. Punpy then propagates the PDFs for the Xi to Y
and then summarises the results through returning the uncertainties and correlation matrices.

As punpy is meant to be widely applicable, the user can define the measurand, input quantities 
and measurement function themselves. Within punpy, the input quantities and measurand will often 
be provided as python arrays (or scalars) and the measurement function in particular needs to be 
a python function that can take the input quantities as function arguments and returns the measurand.

To generate the PDF, punpy generates samples of draws from the PDF for each of the input quantities (total number of
draws is set by keyword `MCsteps`). Currently, punpy assumes all PDF to be gaussian, but this can 
easily be expanded upon in future work. Internally, punpy always generates independent random normally distributed
samples first and then correlates them where necessary using the Cholesky decomposition method (see paragraph below). 
Using this Cholesky decomposition correlates the PDF of the input quantities which means the joint PDF are defined. 
Each draw in the sample is then run through the measurement function and as a result we can a sample (and thus the 
PDF) of the measurand Y. Punpy then calculated the uncertainties from the standard deviation in the sample and the 
correlation matrix from the correlation coefficients between the different values in Y. 

Cholesky decomposition is a usefull method from linear algebra, which allows to efficiently draw samples from a 
multivariate probability distribution (joint PDF). The Cholesky decomposition is a decomposition of a 
positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose. The positive-definite
matrix being decomposed here is the correlation or covriance matrix () and R is the upper triangular matrix given by the 
Cholesky decomposition:

:math:`S(X)=R^T R`.

When sampling from the joint pdf, one can first draw samples Zi = (Z1, ... , ZN) for the input quantities Xi from the
independent PDF for the input quantities (i.e. as if they were uncorrelated). These samples Z can then be combined 
with the decomposition matrix R to obtain the correlated samples Ei = (E1, ... , EN):

:math:`E = X + R^T Z`.

The measurand pdf is then defined by processing each draw Ei to Y:

:math:`Y = f(E)`.


.. _Jacobian Method
Jacobian Method
########################
In progress

