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

The punpy package uses a Monte Carlo (MC) approach to propagate various types of uncertainty through a given measurement function. 
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

In order to do propagate uncertainties, punpy generates MC samples from the input quantities
(which can be individually correlated or not), and then propagates these samples through the
measurement function. This is typically done by passing an array consisting of all MC steps of an
input quantity instead of the input quantity themselve for each of the input quantities. Here it is assumed
the measurement function can deal with these higher dimensional arrays by just perfurming numpy operations.
However, this is not always the case. If the inputs to the measurement function are less flexible,
We can instead pass each MC sample individually tothe measurement function by setting the optional
parallel_cores keyword to 1. At the end of this section we'll also see how to use this keyword for parallel processing.

A number of methods can then be used to propagate uncertainties, depending on the kind of uncertainties that need to be propagated.
Internally, punpy always generates random normally distributed samples first and then correlates
them where necessary using the Cholesky decomposition method. For more details see the Monte
Carlo Approach section below.

We start by showing how to propagating random and systematic uncertainties individually.
When given values (arrays or numbers) for the input quantities xn, and their random (ur_xn) or systematic (us_xn) uncertainties, punpy can be used to propage these uncertainties as follows::

   y = measurement_function(x1, x2, x3)
   ur_y = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3])
   us_y = prop.propagate_systematic(measurement_function, [x1, x2, x3], [us_x1, us_x2, us_x3])

Note that these function return as standard product only the uncertainties on the measurand (because the correlation matrices are trivial in this case).
However these functions have an optional return_corr argument that can be set to True. 

It is also possible to include both types of uncertainty at the same time, or to combine random uncertainties in one input quantity with systematic uncertainties in another::

   ub_y, corrb_y = prop.propagate_both(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], [us_x1, us_x2, us_x3])
   ut_y, corrt_y = prop.propagate_type(measurement_function, [x1, x2, x3], [ur_x1, us_x2, us_x3], ['rand','syst','syst'])

Here, in addition to the uncertainties on the measurand, we also provide a correlation matrix between the elements in the measurand.
If required, this correlation matrix can easily be converted to a covariance matrix as::

   covb_y = prop.convert_corr_to_cov(corrb_y, ub_y)


**Propagating uncertainties when measurements are correlated (within input quantity)**

Sometimes the elements of an input quantity xn are not simply independent (random uncertainties) or fully correlated (systematic uncertainty), but rather a combination of the two.
In this case, it is possible to specify a covariance matrix cov_xn between all the elements of xn. If such a covariance matrix is known for every xn, punpy can use them to propage the combined uncertainty::

   uc_y, corrc_y = prop.propagate_cov(measurement_function, [x1, x2, x3], [cov_x1, cov_x2, cov_x3])
   

**Propagating uncertainties when input quantities are correlated (between input quantity)**

In addition to the elements within an input quantity being correlated, it is also possible the input quantities are correlated to eachother.
If this is the case, this functionality can be included in each of the functions specified above by giving an argument to the optional keyword corr_between.
This keyword needs to be set to the correlation matrix between the input quantities, and thus needs to have the appropriate shape (e.g. 3 x 3 array for 3 input quantities)::

   ur2_y = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], corr_between = corr_x1x2x3)
   uc2_y, corrc2_y = prop.propagate_cov(measurement_function, [x1, x2, x3], [cov_x1, cov_x2, cov_x3], corr_between = corr_x1x2x3)


It is also possible to return the generated samples by setting the optional return_samples keyword to True::

   ur_y, samplesr_y, samplesr_x = prop.propagate_random(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], corr_between=corr_x1x2x3, return_samples=True)
   ub_y, corrb_y, samplesr_y, samplesr_x = prop.propagate_both(measurement_function, [x1, x2, x3], [ur_x1, ur_x2, ur_x3], [us_x1, us_x2, us_x3], return_samples=True)

Further examples for different shapes of input quantities are given on the 'examples <https://punpy.readthedocs.io/en/latest/content/examples.html>'_ page.

**Processing the MC samples in parallel**
At the start of this section we already saw that the optional parallel_cores keyword can be used to running the MC
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
      print(ur_y)
      print(us_y)

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

|image0|

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

|image1|

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


Monte Carlo Apprach
########################
in progress










.. |image0| image:: ../../images/image1.png
   :width: 3.97506in
   :height: 2.46154in
.. |image1| image:: ../../images/image2.png
   :width: 4.61478in
   :height: 2.66265in
