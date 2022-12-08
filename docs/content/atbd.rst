.. atbd - algorithm theoretical basis
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _atbd:

Algorithm Theoretical Basis
===========================

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



.. _LPU Method:
Law of Propagation of Uncertainty Method
#########################################

Within the GUM framework uncertainty analysis begins with understanding
the measurement function. The measurement function establishes the
mathematical relationship between all known input quantities (e.g.
instrument counts) and the measurand itself (e.g. radiance). Generally,
this may be written as

.. math:: Y = f\left( X_{i},\ldots,\ X_{N} \right)

where:

-  :math:`Y` is the measurand;

-  :math:`X_{i}` are the input quantities.

Uncertainty analysis is then performed by considering in turn each of
these different input quantities to the measurement function, this
process is represented in Figure 2. Each input quantity may be
influenced by one or more error effects which are described by an
uncertainty distribution. These separate distributions may then be
combined to determine the uncertainty of the measurand,
:math:`u^{2}(Y)`, using the *Law of Propagation of Uncertainties* (GUM,
2008),

.. math:: u^{2}\left( Y \right) = \mathbf{\text{C\ S}}\left( \mathbf{X} \right)\mathbf{C}^{T}

where:

-  :math:`\mathbf{C}` is the vector of sensitivity coefficients,
   :math:`\partial Y/\partial X_{i}`;

-  :math:`\mathbf{S(X)}` is the error covariance matrix for the input
   quantities.

This can be extended to a measurement function with a measurand vector (rather than scalar) :math:`\mathbf{Y}=(Y_{i},\ldots,\ Y_{N})`. 
The uncertainties are then given by:

.. math:: \mathbf{S(Y)}=\mathbf{J}\ \mathbf{S(X)} \mathbf{J}^T	

where:

-  :math:`\mathbf{J}` is the Jacobian matrix of sensitivity coefficients, :math:`J_{ni} = \partial Y_{n}/\partial X_{i}`;
-  :math:`\mathbf{S(Y)}` is the error covariance matrix (n*n) for the measurand;
-  :math:`\mathbf{S(X)}` is the error covariance matrix (i*i) for the input quantities.

The error covariances matrices define the uncertainties (from the diagonal elements) as well as 
the correlation between the different quantities (off-diagonal elements).

.. image:: images/image2.png

*Figure 2 - Conceptual process of uncertainty propagation.*


.. _Monte Carlo Method:

Monte Carlo Method
########################
For a detailed description of the Monte Carlo (MC) method, we refer to `Supplement 1 to the
"Guide to the expression of uncertainty in measurement" - Propagation of distributions
using a Monte Carlo method <https://www.bipm.org/utils/common/documents/jcgm/JCGM_101_2008_E.pdf>`_.

Here we summarise the main steps and detail how these were implemented.
The main stages consist of:

-  Formulation: Defining the measurand (output quantity Y), the input quantities :math:`X = (X_{i},\ldots,\ X_{N})`, and the measurement function (as a model relating Y and X). One also needs to asign Probability Density Functions (PDF) of each of the input quantities, as well as define the correlation between them (through joint PDF).

-  Propagation: propagate the PDFs for the :math:`X_i` through the model to obtain the PDF for Y.

-  Summarizing: Use the PDF for Y to obtain the expectation of Y, the standard uncertainty u(Y) associated with Y (from the standard deviation), and the covariance between the different values in Y.

The MC method implemented in punpy consists of generating joint PDF from the provided 
uncertainties and correlation matrices or covariances. Punpy then propagates the PDFs for the :math:`X_i` to Y
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
matrix being decomposed here is the correlation or covriance matrix (S(X)) and R is the upper triangular matrix given by the 
Cholesky decomposition:

:math:`S(X)=R^T R`.

When sampling from the joint pdf, one can first draw samples :math:`Z = (Z_{i},\ldots,\ Z_{N})` for the input quantities :math:`X_i` from the
independent PDF for the input quantities (i.e. as if they were uncorrelated). These samples :math:`Z_i` can then be combined 
with the decomposition matrix R to obtain the correlated samples :math:`\xi = (\xi_1, ... , \xi_N)`:

:math:`\xi = X + R^T Z`.

The measurand pdf is then defined by processing each draw :math:`\xi_i` to Y:

:math:`Y = f(\xi)`.

