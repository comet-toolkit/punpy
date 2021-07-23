"""Use Monte Carlo to propagate uncertainties"""

import punpy.lpu.lpu_propagation as lpuprop
import punpy.mc.mc_propagation as mcprop

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class FiduceoPropagation:
    def __init__(self, mc=True, parallel_cores=0, dtype=None,steps=100, Jx_diag=None, step= None):
        """
        Initialise MC Propagator

        :param steps: number of MC iterations
        :type steps: int
        :param parallel_cores: number of CPU to be used in parallel processing
        :type parallel_cores: int
        """
        if mc:
            self.prop = mcprop.MCPropagation(steps,parallel_cores=parallel_cores,dtype=dtype)
        else:
            self.prop = lpuprop.LPUPropagation(parallel_cores=parallel_cores,Jx_diag=Jx_diag, step= step)

        # self.parallel_cores = parallel_cores
        # self.dtype = dtype
        # if parallel_cores>1:
        #     self.pool = Pool(self.parallel_cores)

    def propagate_random(
        self,
        func,
        x,
        u_x,
        corr_x=None,
        param_fixed=None,
        corr_between=None,
        return_corr=False,
        return_samples=False,
        repeat_dims=-99,
        corr_axis=-99,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
    ):
        """
        Propagate random uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors (1d-array) or images (2d-array).
        Random uncertainties arise when there is no correlation between repeated measurements.
        It is possible (though rare) that there is a correlation in one of the dimensions that is not one of the repeat_dims.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of random uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis, defaults to None. Can optionally be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array], optional
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param corr_between: correlation matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """

        return self.prop.propagate_random(
        self,
        func,
        x,
        u_x,
        corr_x=corr_x,
        param_fixed=param_fixed,
        corr_between=corr_between,
        return_corr=return_corr,
        return_samples=return_samples,
        repeat_dims=repeat_dims,
        corr_axis=corr_axis,
        fixed_corr_var=fixed_corr_var,
        output_vars=output_vars,
        PD_corr=PD_corr,
        )




    def propagate_rectangle_absolute(self,func,x,u_x,corr_x=None,param_fixed=None,
            corr_between=None,return_corr=False,return_samples=False,repeat_dims=-99,
            corr_axis=-99,fixed_corr_var=False,output_vars=1,PD_corr=True,):
        """
        Propagate systematic uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors (1d-array) or images (2d-array).
        Systematic uncertainties arise when there is full correlation between repeated measurements.
        There is a often also a correlation between measurements along the dimensions that is not one of the repeat_dims.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis, defaults to None. Can optionally be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array], optional
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param corr_between: correlation matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        return self.prop.propagate_systematic(
        self,
        func,
        x,
        u_x,
        corr_x=corr_x,
        param_fixed=param_fixed,
        corr_between=corr_between,
        return_corr=return_corr,
        return_samples=return_samples,
        repeat_dims=repeat_dims,
        corr_axis=corr_axis,
        fixed_corr_var=fixed_corr_var,
        output_vars=output_vars,
        PD_corr=PD_corr)


    def propagate_triangle_relative(
        self,
        func,
        x,
        cov_x,
        param_fixed=None,
        corr_between=None,
        return_corr=True,
        return_samples=False,
        repeat_dims=-99,
        corr_axis=-99,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
    ):
        """
                Propagate uncertainties with given covariance matrix through measurement function with n input quantities.
                Input quantities can be floats, vectors (1d-array) or images (2d-array).
                The covariance matrix can represent the full covariance matrix between all measurements in all dimensions.
                Alternatively if there are repeated measurements specified in repeat_dims, the covariance matrix is given
                for the covariance along the dimension that is not one of the repeat_dims.

                :param func: measurement function
                :type func: function
                :param x: list of input quantities (usually numpy arrays)
                :type x: list[array]
                :param cov_x: list of covariance matrices on input quantities (usually numpy arrays). In case the input quantity is an array of shape (m,o), the covariance matrix needs to be given as an array of shape (m*o,m*o).
                :type cov_x: list[array]
                :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
                :type param_fixed: list of bools, optional
                :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
                :type corr_between: array, optional
                :param return_corr: set to True to return correlation matrix of measurand, defaults to True
                :type return_corr: bool, optional
                :param return_samples: set to True to return generated samples, defaults to False
                :type return_samples: bool, optional
                :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
                :type repeat_dims: integer or list of 2 integers, optional
                :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
                :type corr_axis: integer, optional
                :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
                :type fixed_corr_var: bool or integer, optional
                :param output_vars: number of output parameters in the measurement function. Defaults to 1.
                :type output_vars: integer, optional
                :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
                :type PD_corr: bool, optional
                :return: uncertainties on measurand
                :rtype: array
                """

        self.prop.propagate_cov(
        self,
        func,
        x,
        cov_x,
        param_fixed=param_fixed,
        corr_between=corr_between,
        return_corr=return_corr,
        return_samples=return_samples,
        repeat_dims=repeat_dims,
        corr_axis=corr_axis,
        fixed_corr_var=fixed_corr_var,
        output_vars=output_vars,
        PD_corr=PD_corr,
        )

        # bell_shaped_relative
        # repeating_rectangles
        # repeating_bell_shapes
        # stepped_triangle_absolute
        # Exponential_decay