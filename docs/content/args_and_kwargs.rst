.. Args and kwargs
   Author: Pieter De Vis
   Email: pieter.de.vis@npl.co.uk
   Created: 15/04/20

.. _args:

Arguments and optional keyword arguments for punpy
===================================================

Punpy as a standalone package
#############################

We here list the arguments (args) and keyword arguments (kwargs) that can be provided when creating a MCPropagation and LPUPropagation object.

.. list-table:: arguments_MCPropagation
   :widths: 25 25 50
   :header-rows: 1

   * - argument
     - type
     - default (empty if non-keyword argument)
     - description
   * - steps
     - int
     -
     - number of MC iterations
   * - parallel_cores
     - int, optional
     - 0
     - number of CPU to be used in parallel processing
   * - dtype
     - numpy dtype, optional
     - None
     - numpy dtype for output variables
   * - verbose
     - bool, optional
     - False
     - bool to set if logging info should be printed
   * - MCdimlast
     - bool, optional
     - True
     - bool to set whether the MC dimension should be moved to the last dimension when running through the measurment function (when parallel_cores==0). This can be useful for broadcasting within the measurement function.

.. list-table:: arguments_LPUPropagation
   :widths: 25 25 50
   :header-rows: 1
   * - argument
     - type
     - default (empty if non-keyword argument)
     - description
   * - parallel_cores
     - int, optional
     - 0
     - number of CPU to be used in parallel processing
   * - Jx_diag
     - bool, optional
     - False
     - Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements.
   * - step
     - float, optional
     - None
     - Defines the spacing used when calculating the Jacobian with numdifftools
   * - verbose
     - bool, optional
     - False
     - bool to set if logging info should be printed

For details on the args and kwargs for the different functions in the MCPropagation and LPUPropagation classes, we refer to
the :ref:`API_MC` and :ref:`API_LPU` Sections.

