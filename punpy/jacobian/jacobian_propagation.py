"""Use Monte Carlo to propagate uncertainties"""

import numpy as np
from multiprocessing import Pool
import numdifftools as nd
import punpy.utilities.utilities as util

'''___Authorship___'''
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

class JacobianPropagation:
    def __init__(self,parallel_cores=0):
        """
        Initialise MC Propagator

        :param parallel_cores: number of CPU to be used in parallel processing
        :type parallel_cores: int
        """

        self.parallel_cores = parallel_cores

    def propagate_random(self,func,x,u_x,corr_x=None,param_fixed=None,corr_between=None,
                         return_corr=False,repeat_dims=-99,corr_axis=-99,
                         fixed_corr_var=False,output_vars=1,Jx=None,Jx_diag=False):
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
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param Jx: Jacobian matrix, evaluated at x. This allows to give a precomputed jacobian matrix, which could potentially be calculated using analytical prescription. Defaults to None, in which case Jx is calculated numerically as part of the propagation.
        :rtype Jx: array, optional
        :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
        :rtype Jx_diag: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        fun,xflat,u_xflat,yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats = yshape[repeat_axis]
            if self.parallel_cores>1:
                inputs= np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,u_xb = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    inputs[i]=[func,xb,u_xb,corr_x,param_fixed,
                                                    corr_between,return_corr,
                                                    repeat_dims,corr_axis,
                                                    fixed_corr_var,output_vars,Jx]
                pool = Pool(self.parallel_cores)
                outs=pool.starmap(self.propagate_random,inputs)
                pool.close()

            else:
                outs = np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,u_xb = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    outs[i] = self.propagate_random(func,xb,u_xb,corr_x,param_fixed,
                                                    corr_between,return_corr,
                                                    repeat_dims,corr_axis,
                                                    fixed_corr_var,output_vars,Jx)

            return self.combine_repeated_outs(outs,yshape,n_repeats,repeat_axis,
                                              return_corr,output_vars)

        else:
            if Jx is None:
                Jx=util.calculate_Jacobian(fun,xflat,Jx_diag)

            if corr_between is None:
                corr_between = np.eye(len(x))

            if corr_x is None:
                if corr_axis>=0:
                    corrs=[np.eye(xi.shape[corr_axis]) for xi in x]
                else:
                    corrs=[np.eye(len(xi.flatten())) for xi in x]
            else:
                corrs=np.empty(len(x),dtype=object)
                for i in range(len(x)):
                    if corr_x[i] is None or corr_x[i]=="rand":
                        corrs[i]=np.eye(len(x[i].flatten()))
                    elif corr_x[i]=="syst":
                        corrs[i]=np.ones((len(x[i].flatten()),len(x[i].flatten())))
                    else:
                        corrs[i]=corr_x[i]
            corr_x=util.calculate_flattened_corr(corrs,corr_between)
            cov_x=util.convert_corr_to_cov(corr_x,u_xflat)

            return self.process_jacobian(Jx,cov_x,yshape,return_corr,
                                         fixed_corr,output_vars)

    def propagate_systematic(self,func,x,u_x,corr_x=None,param_fixed=None,
                             corr_between=None,return_corr=False,repeat_dims=-99,
                             corr_axis=-99,fixed_corr_var=False,output_vars=1,
                             Jx=None,Jx_diag=False):
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
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param Jx: Jacobian matrix, evaluated at x. This allows to give a precomputed jacobian matrix, which could potentially be calculated using analytical prescription. Defaults to None, in which case Jx is calculated numerically as part of the propagation.
        :rtype Jx: array, optional
        :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
        :rtype Jx_diag: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        fun,xflat,u_xflat,yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats = yshape[repeat_axis]
            if self.parallel_cores>1:
                inputs= np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,u_xb = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    inputs[i]=[func,xb,u_xb,corr_x,param_fixed,
                                                    corr_between,return_corr,
                                                    repeat_dims,corr_axis,
                                                    fixed_corr_var,output_vars,Jx]
                pool = Pool(self.parallel_cores)
                outs=pool.starmap(self.propagate_systematic,inputs)
                pool.close()

            else:
                outs = np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,u_xb = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    outs[i] = self.propagate_systematic(func,xb,u_xb,corr_x,param_fixed,
                                                    corr_between,return_corr,repeat_dims,
                                                    corr_axis,fixed_corr_var,output_vars,Jx)

            return self.combine_repeated_outs(outs,yshape,n_repeats,repeat_axis,
                                              return_corr,output_vars)

        else:
            if Jx is None:
                Jx = util.calculate_Jacobian(fun,xflat,Jx_diag)

            if corr_between is None:
                corr_between = np.eye(len(x))

            if corr_x is None:
                if corr_axis >= 0:
                    corrs = [np.ones((len(xi.flatten()),len(xi.flatten()))) for xi in x]
                else:
                    corrs = [np.ones((len(xi.flatten()),len(xi.flatten()))) for xi in x]
            else:
                corrs = np.empty(len(x),dtype=object)
                for i in range(len(x)):
                    if corr_x[i] is None or corr_x[i] == "syst":
                        corrs[i] = np.ones((len(x[i].flatten()),len(x[i].flatten())))
                    elif corr_x[i] == "rand":
                        corrs[i] = np.eye(len(x[i].flatten()))
                    else:
                        corrs[i] = corr_x[i]
            corr_x = util.calculate_flattened_corr(corrs,corr_between)
            cov_x = util.convert_corr_to_cov(corr_x,u_xflat)

            return self.process_jacobian(Jx,cov_x,yshape,return_corr,fixed_corr,
                                         output_vars)

    def propagate_cov(self,func,x,cov_x,param_fixed=None,corr_between=None,
                      return_corr=False,repeat_dims=-99,corr_axis=-99,
                      fixed_corr_var=False,output_vars=1,Jx=None,Jx_diag=False):
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
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param Jx: Jacobian matrix, evaluated at x. This allows to give a precomputed jacobian matrix, which could potentially be calculated using analytical prescription. Defaults to None, in which case Jx is calculated numerically as part of the propagation.
        :rtype Jx: array, optional
        :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
        :rtype Jx_diag: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        u_x = [util.uncertainty_from_covariance(cov_x[i]) for i in range(len(x))]
        corr_x = [util.correlation_from_covariance(cov_x[i]) for i in range(len(x))]

        fun,xflat,u_xflat,yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats = yshape[repeat_axis]
            if self.parallel_cores > 1:
                inputs = np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,u_xb = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    inputs[i] = [func,xb,cov_x,param_fixed,
                                                 corr_between,return_corr,repeat_dims,
                                                 corr_axis,fixed_corr_var,output_vars,Jx]
                pool = Pool(self.parallel_cores)
                outs = pool.starmap(self.propagate_cov,inputs)
                pool.close()

            else:
                outs = np.empty(n_repeats,dtype=object)
                for i in range(n_repeats):
                    xb,_ = util.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,
                                                     n_repeats)
                    outs[i] = self.propagate_cov(func,xb,cov_x,param_fixed,
                                                 corr_between,return_corr,repeat_dims,
                                                 corr_axis,fixed_corr_var,output_vars,Jx)

            return self.combine_repeated_outs(outs,yshape,n_repeats,repeat_axis,
                                              return_corr,output_vars)

        else:
            if Jx is None:
                Jx=util.calculate_Jacobian(fun,xflat,Jx_diag)

            if corr_between is None:
                corr_between = np.eye(len(x))

            corr_x = util.calculate_flattened_corr(corr_x,corr_between)
            cov_x = util.convert_corr_to_cov(corr_x,u_xflat)

            return self.process_jacobian(Jx,cov_x,yshape,return_corr,fixed_corr,
                                         output_vars)

    def process_jacobian(self,J,covx,shape_y,return_corr,fixed_corr=None,output_vars=1):
        covy=np.dot(np.dot(J,covx),J.T)
        u_func=np.diag(covy)**0.5
        if fixed_corr is None:
            corr_y=util.convert_cov_to_corr(covy,u_func)
        else:
            corr_y=fixed_corr
        if not return_corr:
            if output_vars==1:
                return u_func.reshape(shape_y)
            else:
                return u_func.reshape((output_vars,)+shape_y)

        else:
            if output_vars==1:
                return u_func.reshape(shape_y),corr_y
            else:
                #create an empty arrays and then populate it with the correlation matrix for each output parameter individually
                corr_ys,corr_out=util.separate_flattened_corr(corr_y,output_vars)

                return u_func.reshape((output_vars,)+shape_y),corr_ys,corr_out

    def perform_checks(self,func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,
                       fixed_corr_var):
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
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :return: yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr
        :rtype: tuple, list[array], int, int, int, array
        """

        if output_vars==1:
            fun = lambda c: func(*c.reshape(len(x),-1))
        else:
            fun = lambda c: np.concatenate(func(*c.reshape(len(x),-1)))

        # find the shape
        if output_vars == 1:
            yshape = np.array(func(*x)).shape
        else:
            yshape = np.array(func(*x)[0]).shape

        # shapewarning=False
        # for i in range(len(x)):
        #     if hasattr(x[i], "__len__"):
        #         if x[i].shape != yshape and self.parallel_cores == 0:
        #             shapewarning=True
        #     elif self.parallel_cores == 0:
        #         shapewarning=True
        #
        # if shapewarning:
        #     warnings.warn(
        #         "It looks like one of your input quantities is not an array or does not "
        #         "have the same shape as the measurand. This is not a problem, but means "
        #         "you likely cannot use array operations in your measurement function. "
        #         "You might need to set parallel_cores to 1 or higher when creating "
        #         "your MCPropagation object.")

        # Check for which input quantities there is no uncertainty,
        # replacing Nones with zeros where necessary.
        # Count the number of non-zero uncertainties. If this number is one, the
        # correlation matrix for the measurand will be the same as for this input qty.

        count=0
        for i in range(len(x)):
            if u_x[i] is None:
                if hasattr(x[i],"__len__"):
                    u_x[i] = np.zeros(x[i].shape)
                else:
                    u_x[i] = 0.
            if np.sum(u_x[i])!=0 and fixed_corr_var==True:
                count+=1
                var=i
            if corr_x is not None:
                if corr_x[i] is not None:
                    if not isinstance(corr_x[i],str):
                        if np.any(corr_x[i]>1.000001):
                            raise ValueError("One of the provided correlation matrices "
                                             "has elements >1.")

        if count==1:
            fixed_corr_var=var
        else:
            fixed_corr_var=-99


        if fixed_corr_var >= 0 and corr_x is not None:
            if corr_x[fixed_corr_var] == "rand":
                fixed_corr = np.eye(len(u_x[fixed_corr_var]))
            elif corr_x[fixed_corr_var] == "syst":
                fixed_corr = np.ones((len(u_x[fixed_corr_var]),len(u_x[fixed_corr_var])))
            else:
                fixed_corr = corr_x[fixed_corr_var]

        else:
            fixed_corr = None

        # Set up repeat_axis and repeat_dims for proper use in recursive function.
        if isinstance(repeat_dims,int):
            repeat_axis = repeat_dims
            repeat_dims = -99
        else:
            repeat_axis = repeat_dims[0]
            repeat_dims = repeat_dims[1]
            if repeat_axis<repeat_dims:
                repeat_dims -= 1

        if repeat_axis >= 0:
            if corr_axis > repeat_axis:
                corr_axis -= 1
            elif corr_axis == repeat_axis:
                print("corr_axis and repeat_axis keywords should not be the same.")
                exit()

        xflat = np.concatenate([xi.flatten() for xi in x])
        u_xflat = np.concatenate([u_xi.flatten() for u_xi in u_x])

        return fun,xflat,u_xflat,yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr

    def combine_repeated_outs(self,outs,yshape,n_repeats,repeat_axis,return_corr,
                              output_vars):
        """
        Combine the outputs of the repeated measurements into one results array

        :param outs: list of outputs of the repeated measurements
        :type outs: list[array]
        :param yshape: shape of the measurand
        :type yshape: tuple
        :param n_repeats: number of repeated measurements
        :type n_repeats: int
        :param lenx: number of input quantities
        :type lenx: int
        :param repeat_axis: dimension along which the measurements are repeated
        :type repeat_axis: int
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: combined outputs
        :rtype: list[array]
        """
        if not return_corr:
            if output_vars == 1:
                u_func = np.zeros(yshape)
                if repeat_axis == 0:
                    for i in range(len(outs)):
                        u_func[i] = outs[i]
                elif repeat_axis == 1:
                    for i in range(len(outs)):
                        if len(outs[i].shape) > 1:
                            if outs[i].shape[1] == 1:
                                u_func[:,i] = outs[i][:,0]
                            else:
                                u_func[:,i] = outs[i]
                        else:
                            u_func[:,i] = outs[i]

                elif repeat_axis == 2:
                    for i in range(len(outs)):
                        u_func[:,:,i] = outs[i]
            else:
                u_func = np.zeros((output_vars,)+yshape)
                if repeat_axis == 0:
                    for i in range(len(outs)):
                        u_func[:,i] = outs[i]
                elif repeat_axis == 1:
                    for i in range(len(outs)):
                        u_func[:,:,i] = np.array(outs[i])
                elif repeat_axis == 2:
                    for i in range(len(outs)):
                        u_func[:,:,:,i] = outs[i]
            return u_func

        else:
            if output_vars == 1:
                u_func = np.zeros(yshape)
                if repeat_axis == 0:
                    for i in range(len(outs)):
                        u_func[i] = outs[i][0]
                elif repeat_axis == 1:
                    for i in range(len(outs)):
                        if len(outs[i][0].shape) > 1:
                            if outs[i][0].shape[1] == 1:
                                u_func[:,i] = outs[i][0][:,0]
                            else:
                                u_func[:,i] = outs[i][0]
                        else:
                            u_func[:,i] = outs[i][0]
                elif repeat_axis == 2:
                    for i in range(len(outs)):
                        u_func[:,:,i] = outs[i][0]
            else:
                u_func = np.zeros((output_vars,)+yshape)
                if repeat_axis == 0:
                    for i in range(len(outs)):
                        u_func[:,i] = outs[i][0]
                elif repeat_axis == 1:
                    for i in range(len(outs)):
                        u_func[:,:,i] = outs[i][0]
                elif repeat_axis == 2:
                    for i in range(len(outs)):
                        u_func[:,:,:,i] = outs[i][0]

            returns = np.empty(len(outs[0]),dtype=object)
            returns[0] = u_func
            extra_index = 0
            if return_corr:
                corr = np.mean([outs[i][1] for i in range(n_repeats)],axis=0)
                returns[1] = corr
                extra_index += 1

            if output_vars > 1:
                corr_out = np.mean([outs[i][1+extra_index] for i in range(n_repeats)],
                                   axis=0)
                returns[1+extra_index] = corr_out
                extra_index += 1

            return returns