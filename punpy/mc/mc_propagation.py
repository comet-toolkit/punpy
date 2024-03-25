"""Use Monte Carlo to propagate uncertainties"""

import time
import warnings
from multiprocessing import Pool

import comet_maths as cm
import numpy as np

import punpy.utilities.utilities as util

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MCPropagation:
    def __init__(
        self, steps, parallel_cores=0, dtype=None, verbose=False, MCdimlast=True
    ):
        """
        Initialise MC Propagator

        :param steps: number of MC iterations
        :type steps: int
        :param parallel_cores: number of CPU to be used in parallel processing
        :type parallel_cores: int
        :param dtype: numpy dtype for output variables
        :type dtype: numpy dtype
        :param verbose: bool to set if logging info should be printed
        :type verbose: bool
        :param MCdimlast: bool to set whether the MC dimension should be moved to the last dimension when running through the measurment function (when parallel_cores==0). This can be useful for broadcasting within the measurement function. defaults to False
        :type MCdimlast: bool
        """

        self.MCsteps = steps
        self.parallel_cores = parallel_cores
        self.dtype = dtype
        if parallel_cores > 1:
            self.pool = Pool(self.parallel_cores)
        self.verbose = verbose
        self.starttime = time.time()
        self.MCdimlast = MCdimlast

    def propagate_random(
        self,
        func,
        x,
        u_x,
        corr_x=None,
        param_fixed=None,
        corr_between=None,
        samples=None,
        return_corr=False,
        return_samples=False,
        repeat_dims=-99,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
        refyvar=0,
        pdf_shape="gaussian",
        pdf_params=None,
        allow_some_nans=True,
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
        :param samples: allows to provide a Monte Carlo sample previously generated. This sample of input quantities will be used instead of generating one from the uncertainties and error-correlation. Defaults to None
        :type samples: list[array], optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True.
        :type PD_corr: bool, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims).
        :type refyvar: int, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian.
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters).
        :type pdf_params: dict, optional
        :param allow_some_nans: set to False to ignore any MC sample which has any nan's in the measurand. Defaults to True, in which case only MC samples with only nan's are ignored.
        :type allow_some_nans: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        if corr_x is None:
            corr_x = ["rand"] * len(x)
        for i in range(len(x)):
            if corr_x[i] is None:
                corr_x[i] = "rand"

        return self.propagate_standard(
            func,
            x,
            u_x,
            corr_x,
            param_fixed=param_fixed,
            corr_between=corr_between,
            samples=samples,
            return_corr=return_corr,
            return_samples=return_samples,
            repeat_dims=repeat_dims,
            corr_dims=corr_dims,
            separate_corr_dims=separate_corr_dims,
            fixed_corr_var=fixed_corr_var,
            output_vars=output_vars,
            PD_corr=PD_corr,
            refyvar=refyvar,
            pdf_shape=pdf_shape,
            pdf_params=pdf_params,
            allow_some_nans=allow_some_nans,
        )

    def propagate_systematic(
        self,
        func,
        x,
        u_x,
        corr_x=None,
        param_fixed=None,
        corr_between=None,
        samples=None,
        return_corr=False,
        return_samples=False,
        repeat_dims=-99,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
        refyvar=0,
        pdf_shape="gaussian",
        pdf_params=None,
        allow_some_nans=True,
    ):
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
        :param samples: allows to provide a Monte Carlo sample previously generated. This sample of input quantities will be used instead of generating one from the uncertainties and error-correlation. Defaults to None
        :type samples: list[array], optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :param allow_some_nans: set to False to ignore any MC sample which has any nan's in the measurand. Defaults to True, in which case only MC samples with only nan's are ignored.
        :type allow_some_nans: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        if corr_x is None:
            corr_x = ["syst"] * len(x)
        for i in range(len(x)):
            if corr_x[i] is None:
                corr_x[i] = "syst"

        return self.propagate_standard(
            func,
            x,
            u_x,
            corr_x,
            param_fixed=param_fixed,
            corr_between=corr_between,
            samples=samples,
            return_corr=return_corr,
            return_samples=return_samples,
            repeat_dims=repeat_dims,
            corr_dims=corr_dims,
            separate_corr_dims=separate_corr_dims,
            fixed_corr_var=fixed_corr_var,
            output_vars=output_vars,
            PD_corr=PD_corr,
            refyvar=refyvar,
            pdf_shape=pdf_shape,
            pdf_params=pdf_params,
            allow_some_nans=allow_some_nans,
        )

    def propagate_cov(
        self,
        func,
        x,
        cov_x,
        param_fixed=None,
        corr_between=None,
        samples=None,
        return_corr=True,
        return_samples=False,
        repeat_dims=-99,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
        refyvar=0,
        pdf_shape="gaussian",
        pdf_params=None,
        allow_some_nans=True,
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
        :param cov_x: list of covariance matrices on input quantities (usually numpy arrays). In case the input quantity is an array of shape (m,o), the covariance matrix  is typically given as an array of shape (m*o,m*o).
        :type cov_x: list[array]
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param samples: allows to provide a Monte Carlo sample previously generated. This sample of input quantities will be used instead of generating one from the uncertainties and error-correlation. Defaults to None
        :type samples: list[array], optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :param allow_some_nans: set to False to ignore any MC sample which has any nan's in the measurand. Defaults to True, in which case only MC samples with only nan's are ignored.
        :type allow_some_nans: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        x = [np.atleast_1d(x[i]) for i in range(len(x))]
        cov_x = [np.atleast_2d(cov_x[i]) for i in range(len(x))]
        u_x = [
            cm.uncertainty_from_covariance(cov_x[i]).reshape(x[i].shape)
            for i in range(len(x))
        ]
        corr_x = [cm.correlation_from_covariance(cov_x[i]) for i in range(len(x))]

        return self.propagate_standard(
            func,
            x,
            u_x,
            corr_x,
            param_fixed=param_fixed,
            corr_between=corr_between,
            samples=samples,
            return_corr=return_corr,
            return_samples=return_samples,
            repeat_dims=repeat_dims,
            corr_dims=corr_dims,
            separate_corr_dims=separate_corr_dims,
            fixed_corr_var=fixed_corr_var,
            output_vars=output_vars,
            PD_corr=PD_corr,
            refyvar=refyvar,
            pdf_shape=pdf_shape,
            pdf_params=pdf_params,
            allow_some_nans=allow_some_nans,
        )

    def propagate_standard(
        self,
        func,
        x,
        u_x,
        corr_x,
        param_fixed=None,
        corr_between=None,
        samples=None,
        return_corr=False,
        return_samples=False,
        repeat_dims=-99,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
        refyvar=0,
        pdf_shape="gaussian",
        pdf_params=None,
        allow_some_nans=True,
    ):
        """
        Propagate uncertainties through measurement function with n input quantities. Correlations must be specified in corr_x.
        Input quantities can be floats, vectors (1d-array) or images (2d-array).
        Systematic uncertainties arise when there is full correlation between repeated measurements.
        There is a often also a correlation between measurements along the dimensions that is not one of the repeat_dims.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis. Can be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array]
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param corr_between: correlation matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param samples: allows to provide a Monte Carlo sample previously generated. This sample of input quantities will be used instead of generating one from the uncertainties and error-correlation. Defaults to None
        :type samples: list[array], optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :param allow_some_nans: set to False to ignore any MC sample which has any nan's in the measurand. Defaults to True, in which case only MC samples with only nan's are ignored.
        :type allow_some_nans: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        if self.verbose:
            print(
                "starting propagation (%s s since creation of prop object)"
                % (time.time() - self.starttime)
            )

        # check if mcsteps is 0, if so don't propagate uncertainties and just return None
        if self.MCsteps == 0:
            return self.return_no_unc(return_corr, return_samples)

        (
            yshapes,
            x,
            u_x,
            corr_x,
            n_repeats,
            repeat_shape,
            repeat_dims,
            corr_dims,
            fixed_corr,
        ) = self.perform_checks(
            func,
            x,
            u_x,
            corr_x,
            repeat_dims,
            corr_dims,
            separate_corr_dims,
            output_vars,
            fixed_corr_var,
            param_fixed,
            refyvar,
        )

        if n_repeats > 0:
            xb, u_xb = util.select_repeated_x(
                x, u_x, param_fixed, 0, repeat_dims, repeat_shape
            )
            out_0 = self.propagate_standard(
                func,
                xb,
                u_xb,
                corr_x,
                param_fixed,
                corr_between,
                samples,
                return_corr,
                return_samples,
                -99,
                corr_dims=corr_dims,
                separate_corr_dims=separate_corr_dims,
                fixed_corr_var=fixed_corr_var,
                output_vars=output_vars,
                PD_corr=False,
                pdf_shape=pdf_shape,
                pdf_params=pdf_params,
            )

            outs = self.make_new_outs(
                out_0,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            outs = self.add_repeated_outs(
                outs,
                0,
                out_0,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            for i in range(1, n_repeats):
                xb, u_xb = util.select_repeated_x(
                    x, u_x, param_fixed, i, repeat_dims, repeat_shape
                )
                out_i = self.propagate_standard(
                    func,
                    xb,
                    u_xb,
                    corr_x,
                    param_fixed,
                    corr_between,
                    samples,
                    return_corr,
                    return_samples,
                    -99,
                    corr_dims=corr_dims,
                    separate_corr_dims=separate_corr_dims,
                    fixed_corr_var=fixed_corr_var,
                    output_vars=output_vars,
                    PD_corr=False,
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                )

                outs = self.add_repeated_outs(
                    outs,
                    i,
                    out_i,
                    yshapes,
                    len(x),
                    n_repeats,
                    repeat_shape,
                    repeat_dims,
                    return_corr,
                    return_samples,
                    output_vars,
                    refyvar,
                )
                if self.verbose:
                    print(
                        "repeated measurement %s out of %s processed (%s s since creation of prop object)"
                        % (i, n_repeats, time.time() - self.starttime)
                    )

            outs = self.finish_repeated_outs(
                outs,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            return outs

        else:
            if samples is not None:
                MC_x = samples
            elif all([not np.any(u_xi) for u_xi in u_x]):
                return self.return_no_unc(return_corr, return_samples)
            else:
                MC_x = self.generate_MC_sample(
                    x,
                    u_x,
                    corr_x,
                    corr_between=corr_between,
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                )

            MC_y = self.run_samples(
                func, MC_x, output_vars=output_vars, allow_some_nans=allow_some_nans
            )

            return self.process_samples(
                MC_x,
                MC_y,
                return_corr,
                return_samples,
                yshapes,
                corr_dims,
                separate_corr_dims,
                fixed_corr,
                PD_corr,
                output_vars,
            )

    def generate_MC_sample(
        self,
        x,
        u_x,
        corr_x,
        corr_between=None,
        pdf_shape="gaussian",
        pdf_params=None,
        comp_list=False,
    ):
        """
        function to generate MC sample for input quantities

        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis. Can be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array]
        :param corr_between: correlation matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :param comp_list: boolean to define whether u_x and corr_x are given as a list or individual uncertainty components. Defaults to False, in which case a single combined uncertainty component is given per input quantity.
        :type comp_list: bool, optional
        :return: MC sample for input quantities
        :rtype: list[array]
        """
        MC_data = np.empty(len(x), dtype=np.ndarray)

        for i in range(len(x)):
            MC_data[i] = cm.generate_sample(
                self.MCsteps,
                x,
                u_x,
                corr_x,
                i,
                pdf_shape=pdf_shape,
                pdf_params=pdf_params,
                comp_list=comp_list,
            )

        if corr_between is not None:
            MC_data = cm.correlate_sample_corr(MC_data, corr_between)
        if self.verbose:
            print(
                "samples generated (%s s since creation of prop object)"
                % (time.time() - self.starttime)
            )
        return MC_data

    def generate_MC_sample_cov(
        self, x, cov_x, corr_between=None, pdf_shape="gaussian", pdf_params=None
    ):
        """
        function to generate MC sample for input quantities from covariance matrix

        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis. Can be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array]
        :param corr_between: correlation matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :return: MC sample for input quantities
        :rtype:
        """

        MC_data = np.empty(len(x), dtype=np.ndarray)
        for i in range(len(x)):
            if not hasattr(x[i], "__len__"):
                MC_data[i] = cm.generate_sample_systematic(
                    self.MCsteps,
                    x[i],
                    cov_x[i],
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                )
            else:
                MC_data[i] = cm.generate_sample_cov(
                    self.MCsteps,
                    x[i].ravel(),
                    cov_x[i],
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                ).reshape(x[i].shape + (self.MCsteps,))

        if corr_between is not None:
            MC_data = cm.correlate_sample_corr(MC_data, corr_between)

        return MC_data

    def propagate_cov_flattened(
        self,
        func,
        x,
        cov_x,
        param_fixed=None,
        corr_between=None,
        samples=None,
        return_corr=True,
        return_samples=False,
        repeat_dims=-99,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr_var=False,
        output_vars=1,
        PD_corr=True,
        refyvar=0,
        pdf_shape="gaussian",
        pdf_params=None,
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
        :param samples: allows to provide a Monte Carlo sample previously generated. This sample of input quantities will be used instead of generating one from the uncertainties and error-correlation. Defaults to None
        :type samples: list[array], optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int, optional
        :param pdf_shape: string identifier of the probability density function shape, defaults to gaussian
        :type pdf_shape: str, optional
        :param pdf_params: dictionaries defining optional additional parameters that define the probability density function, Defaults to None (gaussian does not require additional parameters)
        :type pdf_params: dict, optional
        :return: uncertainties on measurand
        :rtype: array
        """

        (
            yshapes,
            x,
            u_x,
            corr_x,
            n_repeats,
            repeat_shape,
            repeat_dims,
            corr_dims,
            fixed_corr,
        ) = self.perform_checks(
            func,
            x,
            cov_x,
            None,
            repeat_dims,
            corr_dims,
            separate_corr_dims,
            output_vars,
            fixed_corr_var,
            param_fixed,
            refyvar,
        )

        if n_repeats > 0:
            xb, _ = util.select_repeated_x(
                x, x, param_fixed, 0, repeat_dims, repeat_shape
            )
            out_0 = self.propagate_cov(
                func,
                xb,
                cov_x,
                param_fixed,
                corr_between,
                return_corr,
                return_samples,
                -99,
                corr_dims=corr_dims,
                separate_corr_dims=separate_corr_dims,
                output_vars=output_vars,
                PD_corr=False,
                pdf_shape=pdf_shape,
                pdf_params=pdf_params,
            )

            outs = self.make_new_outs(
                out_0,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            outs = self.add_repeated_outs(
                outs,
                0,
                out_0,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            for i in range(1, n_repeats):
                xb, _ = util.select_repeated_x(
                    x, x, param_fixed, i, repeat_dims, repeat_shape
                )

                out_i = self.propagate_cov(
                    func,
                    xb,
                    cov_x,
                    param_fixed,
                    corr_between,
                    return_corr,
                    return_samples,
                    -99,
                    corr_dims=corr_dims,
                    separate_corr_dims=separate_corr_dims,
                    output_vars=output_vars,
                    PD_corr=False,
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                )

                outs = self.add_repeated_outs(
                    outs,
                    i,
                    out_i,
                    yshapes,
                    len(x),
                    n_repeats,
                    repeat_shape,
                    repeat_dims,
                    return_corr,
                    return_samples,
                    output_vars,
                    refyvar,
                )

            outs = self.finish_repeated_outs(
                outs,
                yshapes,
                len(x),
                n_repeats,
                repeat_shape,
                repeat_dims,
                return_corr,
                return_samples,
                output_vars,
                refyvar,
            )

            return outs

        else:
            if samples is not None:
                MC_data = samples
            else:
                MC_data = self.generate_MC_sample_cov(
                    x,
                    cov_x,
                    corr_between=corr_between,
                    pdf_shape=pdf_shape,
                    pdf_params=pdf_params,
                )

        return self.process_samples(
            func,
            MC_data,
            return_corr,
            return_samples,
            yshapes,
            corr_dims,
            separate_corr_dims,
            fixed_corr,
            PD_corr,
            output_vars,
        )

    def perform_checks(
        self,
        func,
        x,
        u_x,
        corr_x,
        repeat_dims,
        corr_dims,
        separate_corr_dims,
        output_vars,
        fixed_corr_var,
        param_fixed,
        refyvar,
    ):
        """
        Perform checks on the input parameters and set up the appropriate keywords for further processing

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis, defaults to None. Can optionally be set to "rand" (diagonal correlation matrix), "syst" (correlation matrix of ones) or a custom correlation matrix.
        :type corr_x: list[array], optional
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer or list, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int
        :return: yshape,u_x,repeat_axis,repeat_dims,corr_dims,fixed_corr
        :rtype: tuple, list[array], int, int, int, array
        """

        # Set up repeat_axis and repeat_dims for proper use in recursive function.
        if isinstance(repeat_dims, int):
            repeat_dims = [repeat_dims]

        # find the shape
        y = func(*x)
        if output_vars == 1:
            yshape = np.array(y).shape
            yshapes = [np.array(y).shape]
        else:
            yshape = np.array(y[refyvar]).shape
            yshapes = [np.array(y[i]).shape for i in range(output_vars)]

        shapewarning = False
        for i in range(len(x)):
            try:
                if not (np.all(np.isfinite(u_x[i]))):
                    warnings.warn(
                        "One of your uncertainties has nans (of inf). This can cause issues in the uncertainty propagation. Please remove or replace your nans."
                    )

                if not (np.all(np.isfinite(corr_x[i]))):
                    warnings.warn(
                        "One of your error correlation matrices has nans (of inf). This can cause issues in the uncertainty propagation. Please remove or replace your nans."
                    )
            except:
                pass

            if hasattr(x[i], "__shape__"):
                if param_fixed is not None:
                    if (
                        x[i].shape != yshape
                        and self.parallel_cores == 0
                        and not param_fixed[i]
                    ):
                        shapewarning = True
                else:
                    if x[i].shape != yshape and self.parallel_cores == 0:
                        shapewarning = True
            else:
                if self.parallel_cores == 0 and not hasattr(x[i], "__len__"):
                    shapewarning = True

        if shapewarning:
            warnings.warn(
                "It looks like one of your input quantities is not an array or does not "
                "have the same shape as the measurand. This is not a problem, but means "
                "you likely cannot use array operations in your measurement function. "
                "You might need to set parallel_cores to 1 or higher when creating "
                "your MCPropagation object."
            )

        # Check for which input quantities there is no uncertainty,
        # replacing Nones with zeros where necessary.
        # Count the number of non-zero uncertainties. If this number is one, the
        # correlation matrix for the measurand will be the same as for this input qty.

        count = 0
        for i in range(len(x)):
            if u_x[i] is None:
                if hasattr(x[i], "__len__"):
                    u_x[i] = np.zeros(x[i].shape)
                else:
                    u_x[i] = 0.0

            if u_x is not None:
                if u_x[i] is not None:
                    if self.dtype is not None:
                        u_x[i] = np.array(u_x[i], dtype=self.dtype)

            if np.sum(u_x[i]) != 0 and fixed_corr_var == True:
                count += 1
                var = i

            if corr_x is not None:
                if corr_x[i] is not None:
                    if not (isinstance(corr_x[i], str) or isinstance(corr_x[i], dict)):
                        if np.any(corr_x[i] > 1.000001):
                            raise ValueError(
                                "punpy.mc_propagation: One of the provided correlation matrices "
                                "has elements >1."
                            )

        if count == 1:
            fixed_corr_var = var
        else:
            fixed_corr_var = -99

        if fixed_corr_var >= 0 and corr_x is not None:
            if isinstance(corr_x[fixed_corr_var], str):
                if corr_x[fixed_corr_var] == "rand":
                    fixed_corr = np.eye(len(u_x[fixed_corr_var]))
                elif corr_x[fixed_corr_var] == "syst":
                    fixed_corr = np.ones(
                        (len(u_x[fixed_corr_var]), len(u_x[fixed_corr_var]))
                    )
            else:
                fixed_corr = corr_x[fixed_corr_var]

        else:
            fixed_corr = None

        if (isinstance(corr_dims, int)) or (isinstance(corr_dims, str)):
            corr_dims = [corr_dims]

        if len(repeat_dims) == 1:
            if repeat_dims[0] >= 0:
                n_repeats = yshape[repeat_dims[0]]
                for ico in range(len(corr_dims)):
                    if corr_dims[ico] > repeat_dims[0]:
                        corr_dims[ico] -= 1
                    elif corr_dims[ico] == repeat_dims[0]:
                        raise ValueError(
                            "punpy.mc_propagation: corr_dims and repeat_axis keywords should not be the same."
                        )
            else:
                n_repeats = 0
            repeat_shape = (n_repeats,)  # repeat_dims = -99
        else:
            repeat_dims = -np.sort(-np.array(repeat_dims))
            repeat_shape = tuple([yshape[repeat_dim] for repeat_dim in repeat_dims])
            n_repeats = np.prod(repeat_shape)
            for repeat_dim in repeat_dims:
                for ico in range(len(corr_dims)):
                    if corr_dims[ico] > repeat_dim:
                        corr_dims[ico] -= 1
                    elif corr_dims[ico] == repeat_dim:
                        raise ValueError(
                            "punpy.mc_propagation: corr_dims and repeat_axis keywords should not be the same."
                        )

        if output_vars > 1 and separate_corr_dims:
            if output_vars != len(corr_dims):
                raise ValueError(
                    "The provided corr_dims was not a list with the corr_dims for each output variable. This needs to be the case when setting separate_corr_dims to True"
                )

        return (
            yshapes,
            x,
            u_x,
            corr_x,
            n_repeats,
            repeat_shape,
            repeat_dims,
            corr_dims,
            fixed_corr,
        )

    def make_new_outs(
        self,
        out_0,
        yshapes,
        lenx,
        n_repeats,
        repeat_shape,
        repeat_dims,
        return_corr,
        return_samples,
        output_vars,
        refyvar,
    ):
        """
        Prepare arrays for storing the results from the individual repeated measurements.

        :param out_0: output array of the first repeated measurement
        :type out_0: array
        :type yshapes: tuple
        :param lenx: number of input quantities
        :type lenx: int
        :param n_repeats: number of repeated measurements
        :type n_repeats: int
        :param repeat_shape: shape along which the measurements are repeated
        :type repeat_shape: tuple
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_Jacobian: set to True to return Jacobian, defaults to False
        :type return_Jacobian: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int
        :return: array of outputs of the repeated measurements
        :rtype: array
        """
        if not return_corr and output_vars == 1 and not return_samples:
            outs = np.empty((n_repeats,) + out_0.shape, dtype=self.dtype)

        else:
            outs = np.empty(len(out_0), dtype=object)

            if output_vars == 1:
                outs[0] = np.empty((n_repeats,) + out_0[0].shape, dtype=self.dtype)

            elif output_vars > 1 and not return_corr and not return_samples:
                outs[0] = np.empty(
                    (output_vars,),
                    dtype=object,
                )
                for i in range(output_vars):
                    outs[0][i] = np.empty(
                        (n_repeats,) + out_0[i].shape, dtype=self.dtype
                    )

            # elif (out_0[0][0].ndim) > 1:
            #     outs[0] = np.empty((output_vars,len(out_0[0][0]),n_repeats,)+out_0[0][0].shape)

            else:
                outs[0] = np.empty(
                    (output_vars,),
                    dtype=object,
                )
                for i in range(output_vars):
                    outs[0][i] = np.empty(
                        (n_repeats,) + out_0[0][i].shape, dtype=self.dtype
                    )

            extra_index = 0
            if return_corr:
                if output_vars == 1:
                    outs[1] = np.zeros_like(out_0[1], dtype=self.dtype)
                    extra_index += 1

                else:
                    outs[1] = np.empty((output_vars,), dtype=object)
                    for ii in range(output_vars):
                        outs[1][ii] = np.zeros_like(out_0[1][ii], dtype=self.dtype)

                    extra_index += 1
                    if all(
                        [yshapes[i] == yshapes[refyvar] for i in range(len(yshapes))]
                    ):
                        corr_out = np.zeros_like(
                            out_0[1 + extra_index], dtype=self.dtype
                        )
                    else:
                        corr_out = None
                    outs[1 + extra_index] = corr_out
                    extra_index += 1

            if return_samples:
                outs[1 + extra_index] = np.empty(lenx, dtype=object)
                outs[2 + extra_index] = np.empty(lenx, dtype=object)
        return outs

    def add_repeated_outs(
        self,
        outs,
        i,
        out_i,
        yshapes,
        lenx,
        n_repeats,
        repeat_shape,
        repeat_dims,
        return_corr,
        return_samples,
        output_vars,
        refyvar,
    ):
        """
        Add the results for a single repeated measurement to the combined array

        :param outs: array with outputs of the repeated measurements
        :type outs: list[array]
        :param i: index of the individual measurements
        :type i: int
        :param out_i: array with outputs of the individual measurements
        :type out_i: list[array]
        :param yshapes: shape of the measurand
        :type yshapes: tuple
        :param lenx: number of input quantities
        :type lenx: int
        :param n_repeats: number of repeated measurements
        :type n_repeats: int
        :param repeat_shape: shape along which the measurements are repeated
        :type repeat_shape: tuple
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_Jacobian: set to True to return Jacobian, defaults to False
        :type return_Jacobian: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int
        :return: combined outputs
        :rtype: list[array]
        """

        if not return_corr and output_vars == 1 and not return_samples:
            outs[i] = out_i

        elif output_vars == 1:
            outs[0][i] = out_i[0]

        elif output_vars > 1 and not return_corr and not return_samples:
            outs[0]
            for ii in range(output_vars):
                outs[0][ii][i] = out_i[ii]

        # elif (out_i[0][0].ndim) > 1:
        #     outs[0] = np.empty(output_vars,dtype=object)
        #     for ii in range(output_vars):
        #         for iii in range(len(out_i[0][ii])):
        #             outs[0][ii][iii][i] = out_i[0][ii][iii]

        else:
            for ii in range(output_vars):
                outs[0][ii][i] = out_i[0][ii]

        extra_index = 0
        if return_corr:
            if output_vars == 1:
                corr = out_i[1]
                if np.isnan(corr).any():
                    warnings.warn(
                        "one of the measurements along the repeat_dim has nans"
                    )

                else:
                    outs[1] += corr
                extra_index += 1

            else:
                for ii in range(output_vars):
                    corr = out_i[1][ii]
                    if np.isnan(corr).any():
                        warnings.warn(
                            "one of the measurements along the repeat_dim has nans"
                        )

                    else:
                        outs[1][ii] += corr

                extra_index += 1
                if all([yshapes[i] == yshapes[refyvar] for i in range(len(yshapes))]):
                    corr_out = out_i[1 + extra_index]
                    if np.isnan(corr).any():
                        warnings.warn(
                            "one of the measurements along the repeat_dim has nans"
                        )

                    else:
                        outs[1 + extra_index] += corr_out
                extra_index += 1

        if return_samples:
            for k in range(lenx):
                if i == 0:
                    outs[1 + extra_index][k] = out_i[1 + extra_index][k]
                    outs[2 + extra_index][k] = out_i[2 + extra_index][k]
                else:
                    outs[1 + extra_index][k] = np.vstack(
                        [outs[1 + extra_index][k], out_i[1 + extra_index][k]]
                    )
                    outs[2 + extra_index][k] = np.vstack(
                        [outs[2 + extra_index][k], out_i[2 + extra_index][k]]
                    )

        return outs

    def finish_repeated_outs(
        self,
        outs,
        yshapes,
        lenx,
        n_repeats,
        repeat_shape,
        repeat_dims,
        return_corr,
        return_samples,
        output_vars,
        refyvar,
    ):
        """
        Do final operations to output arrays with repeated measurements (e.g. dividing by number of repeats to take mean).

        :param outs: array of outputs of the repeated measurements
        :type outs: array
        :type yshapes: tuple
        :param lenx: number of input quantities
        :type lenx: int
        :param n_repeats: number of repeated measurements
        :type n_repeats: int
        :param repeat_shape: shape along which the measurements are repeated
        :type repeat_shape: tuple
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_Jacobian: set to True to return Jacobian, defaults to False
        :type return_Jacobian: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param refyvar: Index of output variable with reference shape (only relevant when output_vars>1; should be output variable with most dimensions; affects things like repeat_dims)
        :type refyvar: int
        :return: finalised array of outputs of the repeated measurements
        :rtype: array
        """
        if not return_corr and output_vars == 1 and not return_samples:
            u_func = np.array(outs, dtype=self.dtype)
        else:
            u_func = np.array(outs[0])
        if len(repeat_dims) == 1:
            if output_vars == 1:
                u_func = np.moveaxis(u_func, 0, repeat_dims[0])
            else:
                u_funcb = u_func[:]
                u_func = np.empty(output_vars, dtype=object)
                if all([yshapes[i] == yshapes[refyvar] for i in range(len(yshapes))]):
                    for i in range(output_vars):
                        u_func[i] = np.moveaxis(u_funcb[i], 0, repeat_dims[0])
                else:
                    for i in range(output_vars):
                        u_func[i] = np.empty(yshapes[i], dtype=self.dtype)
                        if len(yshapes[i]) == len(u_funcb[i].shape):
                            repeat_dim_corr = repeat_dims[0]
                            for idim, dim in enumerate(yshapes[refyvar]):
                                if dim not in yshapes[i] and idim < repeat_dims[0]:
                                    repeat_dim_corr -= 1
                            u_func[i] = np.moveaxis(u_funcb[i], 0, repeat_dim_corr)
                        elif len(yshapes[i]) == 2 and len(u_funcb[i].shape) == 1:
                            for ii in range(yshapes[i][0]):
                                for iii in range(yshapes[i][1]):
                                    u_func[i][ii, iii] = u_funcb[i][iii][ii]
                        elif len(yshapes[i]) == 2 and len(u_funcb[i].shape) == 3:
                            if yshapes[i][0] == u_funcb[i].shape[0]:
                                for ii in range(yshapes[i][0]):
                                    for iii in range(yshapes[i][1]):
                                        u_func[i][ii, iii] = u_funcb[i][ii][0][iii]
                        else:
                            print(yshapes, i, u_funcb[i].shape)
                            raise ValueError(
                                "punpy.mc_propagation: this shape is not supported"
                            )

        else:
            if output_vars == 1:
                u_func = u_func.reshape(repeat_shape + u_func[0].shape)
                for repeat_dim in repeat_dims:
                    u_func = np.moveaxis(u_func, 0, repeat_dim)
            else:
                u_funcb = u_func[:]
                u_func = np.empty(output_vars, dtype=object)
                for i in range(output_vars):
                    u_func[i] = u_funcb[i].reshape(repeat_shape + (-1,))
                    u_func[i] = np.moveaxis(u_func[i], 0, repeat_dims[0])
                    u_func[i] = np.moveaxis(u_func[i], 0, repeat_dims[1])

        if (output_vars == 1 and u_func.shape != yshapes[0]) or (
            output_vars > 1
            and (u_func[0].shape != yshapes[0] or u_func[1].shape != yshapes[1])
        ):
            print(u_func.shape, u_func[0].shape, yshapes)
            raise ValueError(
                "punpy.mc_propagation: The shape of the uncertainties does not match the shape"
                "of the measurand. This is likely a problem with combining"
                "repeated measurements (repeat_dims keyword)."
            )

        if not return_corr and not return_samples:
            return u_func

        else:
            n_masked = len([uf for uf in u_func.T if np.isnan(uf).any()])
            outs[0] = u_func
            if return_corr:
                outs[1] = outs[1] / (n_repeats - n_masked)
                if output_vars == 1:
                    if not cm.isPD(outs[1]):
                        outs[1] = cm.nearestPD_cholesky(
                            outs[1], corr=True, return_cholesky=False
                        )

                else:
                    for j in range(output_vars):
                        if not cm.isPD(outs[1][j]):
                            outs[1][j] = cm.nearestPD_cholesky(
                                outs[1][j], corr=True, return_cholesky=False
                            )

                    if all(
                        [yshapes[i] == yshapes[refyvar] for i in range(len(yshapes))]
                    ):
                        outs[2] = outs[2] / (n_repeats - n_masked)
                        if not cm.isPD(outs[2]):
                            outs[2] = cm.nearestPD_cholesky(
                                outs[2], corr=True, return_cholesky=False
                            )

        return outs

    def run_samples(
        self,
        func,
        MC_x,
        output_vars=1,
        start=None,
        end=None,
        sli=None,
        allow_some_nans=True,
    ):
        """
        process all the MC samples of input quantities through the measurand function

        :param func: measurement function
        :type func: function
        :param MC_x: MC-generated samples of input quantities
        :type MC_x: array[array]
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param start: set this parameter to propagate the input quantities through the measurement function starting from a specfic index. All input quantities before this index are ignored. Defaults to None, in which case no input quantities are ignored.
        :type start: integer, optional
        :param end: set this parameter to propagate the input quantities through the measurement function up until a specfic index. All input quantities after this index are ignored. Defaults to None, in which case no input quantities are ignored.
        :type end: integer, optional
        :param sli: set this parameter to a slice to set which input quantities will be processed through the measurment function. All other input quantities are ignored. Can only be used if start and end are not set. Defaults to None, in which case no input quantities are ignored.
        :type sli: slice, optional
        :param allow_some_nans: set to False to ignore any MC sample which has any nan's in the measurand. Defaults to True, in which case only MC samples with only nan's are ignored.
        :type allow_some_nans: bool, optional
        :return: MC sample of measurand
        :rtype: array[array]
        """
        if (start is not None) or (end is not None):
            sli = slice(start, end)
            indices = range(*sli.indices(self.MCsteps))
        else:
            sli = slice(sli)
            indices = range(*sli.indices(self.MCsteps))

        if self.parallel_cores == 0:
            if self.MCdimlast:
                MC_y = np.moveaxis(
                    func(*[np.moveaxis(dat[sli], 0, -1) for dat in MC_x]), -1, 0
                )
            else:
                MC_y = func(*[x[sli] for x in MC_x])

        elif self.parallel_cores == 1:
            MC_y = list(map(func, *[x[sli] for x in MC_x]))

        else:
            MC_x2 = np.empty(len(indices), dtype=object)
            for i, index in enumerate(indices):
                MC_x2[i] = [MC_x[j][index, ...] for j in range(len(MC_x))]
            MC_y = self.pool.starmap(func, MC_x2)

        if output_vars == 1:
            if allow_some_nans:
                MC_y_out = np.array(
                    [
                        MC_y[i]
                        for i in range(len(indices))
                        if np.any(np.isfinite(MC_y[i]))
                    ],
                    dtype=self.dtype,
                )
            else:
                MC_y_out = np.array(
                    [
                        MC_y[i]
                        for i in range(len(indices))
                        if np.all(np.isfinite(MC_y[i]))
                    ],
                    dtype=self.dtype,
                )
        else:
            if allow_some_nans:
                try:
                    MC_y_out = np.array(
                        [
                            MC_y[i]
                            for i in range(len(indices))
                            if np.all(
                                [
                                    np.any(np.isfinite(MC_y[i][ivar]))
                                    for ivar in range(output_vars)
                                ]
                            )
                        ],
                        dtype=self.dtype,
                    )
                except:
                    MC_y_out = np.array(
                        [
                            MC_y[i]
                            for i in range(len(indices))
                            if np.all(
                                [
                                    np.any(np.isfinite(MC_y[i][ivar]))
                                    for ivar in range(output_vars)
                                ]
                            )
                        ],
                        dtype=object,
                    )

            else:
                try:
                    MC_y_out = np.array(
                        [
                            MC_y[i]
                            for i in range(len(indices))
                            if np.all(
                                [
                                    np.all(np.isfinite(MC_y[i][ivar]))
                                    for ivar in range(output_vars)
                                ]
                            )
                        ],
                        dtype=self.dtype,
                    )
                except:
                    MC_y_out = np.array(
                        [
                            MC_y[i]
                            for i in range(len(indices))
                            if np.all(
                                [
                                    np.all(np.isfinite(MC_y[i][ivar]))
                                    for ivar in range(output_vars)
                                ]
                            )
                        ],
                        dtype=object,
                    )

        if len(MC_y_out) < len(indices):
            if allow_some_nans:
                print(
                    "%s of the %s MC samples were not processed correctly (contained only nans) and will be ignored in the punpy output"
                    % (len(indices) - len(MC_y_out), len(indices))
                )

            else:
                print(
                    "%s of the %s MC samples were not processed correctly (contained some nans) and will be ignored in the punpy output"
                    % (len(indices) - len(MC_y_out), len(indices))
                )

        if len(MC_y_out) == 0:
            MC_y_out = np.array(MC_y, dtype=self.dtype)

        if self.verbose:
            print(
                "samples propagated (%s s since creation of prop object)"
                % (time.time() - self.starttime)
            )

        return MC_y_out

    def combine_samples(self, MC_samples):
        return np.concatenate(MC_samples, axis=0)

    def process_samples(
        self,
        MC_x,
        MC_y,
        return_corr=False,
        return_samples=False,
        yshapes=None,
        corr_dims=-99,
        separate_corr_dims=False,
        fixed_corr=None,
        PD_corr=True,
        output_vars=1,
    ):
        """
        Run the MC-generated samples of input quantities through the measurement function and calculate
        correlation matrix if required.

        :param MC_x: MC-generated samples of input quantities
        :type MC_x: array[array]
        :param MC_y: MC sample of measurand
        :type MC_y: array[array]
        :param return_corr: set to True to return correlation matrix of measurand
        :type return_corr: bool
        :param return_samples: set to True to return generated samples
        :type return_samples: bool
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated. When the combined correlation of 2 or more (but not all) dimensions is required, they can be provided as a string containing the different dimension integers, separated by a dot (e.g. "0.2"). When multiple error_correlations should be calculated, they can be provided as a list.
        :type corr_dims: integer, optional
        :param separate_corr_dims: When set to True and output_vars>1, corr_dims should be a list providing the corr_dims for each output variable, each following the format defined in the corr_dims description. Defaults to False
        :type separate_corr_dims: bool, optional
        :param fixed_corr: correlation matrix to be copied without changing, defaults to None (correlation matrix is calculated rather than copied)
        :type fixed_corr: array
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """

        # if hasattr(MC_y[0,0], '__len__'):
        #     print(yshape,np.array(MC_y[0,0]).shape,np.array(MC_y[1,0]).shape,np.array(MC_y[2,0]).shape,np.array(MC_y[3,0]).shape)
        if yshapes is None:
            if output_vars > 1:
                yshapes = [MC_y[0][i].shape for i in range(output_vars)]
            elif len(MC_y) > 0:
                yshapes = MC_y[0].shape

        if len(MC_y) == 0:
            if output_vars == 1:
                u_func = np.nan * np.zeros(yshapes[0])
            else:
                u_func = np.empty(output_vars, dtype=object)
                for i in range(output_vars):
                    u_func[i] = np.nan * np.zeros(yshapes[i])
        if output_vars == 1:
            u_func = np.std(MC_y, axis=0, dtype=self.dtype)
        else:
            complex_shapes = True
            if yshapes is None:
                complex_shapes = False
            elif all([yshapes[i] == yshapes[0] for i in range(len(yshapes))]):
                complex_shapes = False

            if complex_shapes:
                MC_y2 = np.empty(output_vars, dtype=object)
                u_func = np.empty(output_vars, dtype=object)

                for i in range(output_vars):
                    MC_y2[i] = np.empty((self.MCsteps,) + yshapes[i])
                    for j in range(self.MCsteps):
                        MC_y2[i][j] = MC_y[j, i]
                    u_func[i] = np.std(np.array(MC_y2[i]), axis=0, dtype=self.dtype)

            else:
                u_func = np.std(MC_y, axis=0, dtype=self.dtype)

        if self.verbose:
            print(
                "std calculated (%s s since creation of prop object)"
                % (time.time() - self.starttime)
            )

        if not return_corr:
            if return_samples:
                return u_func, MC_y, MC_x
            else:
                return u_func
        else:
            if output_vars == 1:
                if fixed_corr is None:
                    # if separate_corr_dims, the corr_dims are wrapped in an additional list
                    if separate_corr_dims:
                        corrdims = corr_dims[0]
                    else:
                        corrdims = corr_dims
                    corr_y = cm.calculate_corr(
                        MC_y, corrdims, PD_corr, dtype=self.dtype
                    )
                else:
                    corr_y = fixed_corr

                if self.verbose:
                    print(
                        "corr calculated (%s s since creation of prop object)"
                        % (time.time() - self.starttime)
                    )

                if return_samples:
                    return u_func, corr_y, MC_y, MC_x
                else:
                    return u_func, corr_y

            else:
                # create an empty arrays and then populate it with the correlation matrix for each output parameter individually
                corr_ys = np.empty(output_vars, dtype=object)

                for i in range(output_vars):
                    if separate_corr_dims:
                        corrdims = corr_dims[i]
                    else:
                        corrdims = corr_dims

                    if fixed_corr is None:
                        if complex_shapes:
                            corr_ys[i] = cm.calculate_corr(
                                MC_y2[i], corrdims, PD_corr, self.dtype
                            )
                        else:
                            corr_ys[i] = cm.calculate_corr(
                                MC_y[:, i], corrdims, PD_corr, self.dtype
                            )

                    else:
                        corr_ys[i] = fixed_corr

                if complex_shapes:
                    corr_out = None
                else:
                    # calculate correlation matrix between the different outputs produced by the measurement function.
                    MC_y2 = MC_y.reshape((self.MCsteps, output_vars, -1))

                    corr_out = np.mean(
                        [
                            cm.calculate_corr(
                                MC_y2[:, :, i], PD_corr=PD_corr, dtype=self.dtype
                            )
                            for i in range(len(MC_y2[0, 0]))
                        ],
                        axis=0,
                    )

                if self.verbose:
                    print(
                        "corr calculated for output_var>1 (%s s since creation of prop object)"
                        % (time.time() - self.starttime)
                    )

                if return_samples:
                    return u_func, corr_ys, corr_out, MC_y, MC_x
                else:
                    return u_func, corr_ys, corr_out

    def return_no_unc(self, return_corr, return_samples):
        """
        function to generate outputs in right format when there are no valid uncertainties

        :param return_corr: set to True to return correlation matrix of measurand
        :type return_corr: bool
        :param return_samples: set to True to return generated samples
        :type return_samples: bool
        :return: outputs (None) in right format
        """
        out = None
        if return_corr:
            out = [None, None]
            if return_samples:
                out = [None, None, None, None]
        elif return_samples:
            out = [None, None, None]
        return out
