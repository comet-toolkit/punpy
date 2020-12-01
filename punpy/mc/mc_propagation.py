"""Use Monte Carlo to propagate uncertainties"""

import numpy as np
from multiprocessing import Pool
import warnings
import punpy.utilities.utilities as util

'''___Authorship___'''
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

class MCPropagation:
    def __init__(self,steps,parallel_cores=0):
        """
        Initialise MC Propagator

        :param steps: number of MC iterations
        :type steps: int
        """

        self.MCsteps = steps
        self.parallel_cores = parallel_cores

    def propagate_random(self,func,x,u_x,corr_x=None,param_fixed=None,corr_between=None,
                         return_corr=False,return_samples=False,repeat_dims=-99,
                         corr_axis=-99,fixed_corr_var=False,output_vars=1,PD_corr=True):
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
        :param fixed_corr_var: set to integer to copy the correlation matrix of the dimiension the integer refers to. Set to True to automatically detect if only one uncertainty is present and the correlation matrix of that dimension should be copied. Defaults to False.
        :type fixed_corr_var: bool or integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats=yshape[repeat_axis]
            outs = np.empty(n_repeats,dtype=object)
            for i in range(n_repeats):
                xb, u_xb = self.select_repeated_x(x,u_x,param_fixed,i,
                                                  repeat_axis,n_repeats)

                outs[i] = self.propagate_random(func,xb,u_xb,corr_x,param_fixed,
                                                corr_between,return_corr,
                                                return_samples,repeat_dims,
                                                corr_axis=corr_axis,
                                                output_vars=output_vars,
                                                fixed_corr_var = fixed_corr_var,
                                                PD_corr=False)

            return self.combine_repeated_outs(outs,yshape,n_repeats,len(x),repeat_axis,
                                              return_corr,return_samples,output_vars)

        else:
            MC_data = np.empty(len(x),dtype=np.ndarray)
            for i in range(len(x)):
                if corr_x is None:
                    MC_data[i] = self.generate_samples_random(x[i],u_x[i])
                elif corr_x[i] is None or corr_x[i]=="rand":
                    MC_data[i] = self.generate_samples_random(x[i],u_x[i])
                elif corr_x[i]=="syst":
                    MC_data[i] = self.generate_samples_systematic(x[i],u_x[i])
                else:
                    MC_data[i] = self.generate_samples_correlated(x,u_x,corr_x,i)

            if corr_between is not None:
                MC_data = self.correlate_samples_corr(MC_data,corr_between)

            return self.process_samples(func,MC_data,return_corr,return_samples,
                                        corr_axis,fixed_corr,PD_corr,output_vars)


    def propagate_systematic(self,func,x,u_x,corr_x=None,param_fixed=None,
                             corr_between=None,return_corr=False,return_samples=False,
                             repeat_dims=-99,corr_axis=-99,fixed_corr_var=False,
                             output_vars=1,PD_corr=True):
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
        yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,u_x,corr_x,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats=yshape[repeat_axis]
            outs = np.empty(n_repeats,dtype=object)
            for i in range(n_repeats):
                xb,u_xb = self.select_repeated_x(x,u_x,param_fixed,i,repeat_axis,n_repeats)
                outs[i] = self.propagate_systematic(func,xb,u_xb,corr_x,param_fixed,
                                                    corr_between,return_corr,
                                                    return_samples,repeat_dims,
                                                    corr_axis=corr_axis,
                                                    fixed_corr_var=fixed_corr_var,
                                                    output_vars=output_vars,
                                                    PD_corr=False)
            return self.combine_repeated_outs(outs,yshape,n_repeats,len(x),repeat_axis,
                                              return_corr,return_samples,output_vars)

        else:
            MC_data = np.empty(len(x),dtype=np.ndarray)
            for i in range(len(x)):
                if corr_x is None:
                    MC_data[i] = self.generate_samples_systematic(x[i],u_x[i])
                elif corr_x[i] is None or corr_x[i] == "syst":
                    MC_data[i] = self.generate_samples_systematic(x[i],u_x[i])
                elif corr_x[i] == "rand":
                    MC_data[i] = self.generate_samples_random(x[i],u_x[i])
                else:
                    MC_data[i] = self.generate_samples_correlated(x,u_x,corr_x,i)

            if corr_between is not None:
                MC_data = self.correlate_samples_corr(MC_data,corr_between)

            return self.process_samples(func,MC_data,return_corr,return_samples,
                                        corr_axis,fixed_corr,PD_corr,output_vars)

    def propagate_cov(self,func,x,cov_x,param_fixed=None,corr_between=None,
                      return_corr=True,return_samples=False,repeat_dims=-99,
                      corr_axis=-99,fixed_corr_var=False,output_vars=1,PD_corr=True):
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
        yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr = self.perform_checks(
            func,x,cov_x,None,repeat_dims,corr_axis,output_vars,fixed_corr_var)

        if repeat_axis >= 0:
            n_repeats=yshape[repeat_axis]
            outs = np.empty(n_repeats,dtype=object)
            for i in range(n_repeats):
                xb,_ = self.select_repeated_x(x,x,param_fixed,i,repeat_axis,n_repeats)
                outs[i] = self.propagate_cov(func,xb,cov_x,param_fixed,corr_between,
                                                return_corr,return_samples,repeat_dims,
                                                corr_axis=corr_axis,
                                                output_vars=output_vars,PD_corr=False)
            return self.combine_repeated_outs(outs,yshape,n_repeats,len(x),repeat_axis,
                                              return_corr,return_samples,output_vars)
        else:
            MC_data = np.empty(len(x),dtype=np.ndarray)
            for i in range(len(x)):
                if not hasattr(x[i],"__len__"):
                    MC_data[i] = self.generate_samples_systematic(x[i],cov_x[i])
                elif (all((cov_x[i]==0).flatten())): #This is the case if one of the variables has no uncertainty
                    MC_data[i] = np.tile(x[i].flatten(),(self.MCsteps,1)).T
                elif param_fixed is not None:
                    if param_fixed[i] and (len(x[i].shape) == 2):
                        MC_data[i] = np.array([self.generate_samples_cov(
                            x[i][:,j].flatten(),cov_x[i]).reshape
                            (x[i][:,j].shape+(self.MCsteps,)) for j in
                                               range(x[i].shape[1])]).T
                        MC_data[i] = np.moveaxis(MC_data[i],0,1)
                    else:
                        MC_data[i] = self.generate_samples_cov(x[i].flatten(),cov_x[i])\
                                         .reshape(x[i].shape+(self.MCsteps,))
                else:
                    MC_data[i] = self.generate_samples_cov(x[i].flatten(),cov_x[i])\
                                     .reshape(x[i].shape+(self.MCsteps,))

            if corr_between is not None:
                MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,
                                    fixed_corr,PD_corr,output_vars)

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


        # find the shape
        if output_vars == 1:
            yshape = np.array(func(*x)).shape
        else:
            yshape = np.array(func(*x)[0]).shape

        shapewarning=False
        for i in range(len(x)):
            if hasattr(x[i], "__len__"):
                if x[i].shape != yshape and self.parallel_cores == 0:
                    shapewarning=True
            elif self.parallel_cores == 0:
                shapewarning=True

        if shapewarning:
            warnings.warn(
                "It looks like one of your input quantities is not an array or does not "
                "have the same shape as the measurand. This is not a problem, but means "
                "you likely cannot use array operations in your measurement function. "
                "You might need to set parallel_cores to 1 or higher when creating "
                "your MCPropagation object.")

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


        return yshape,u_x,repeat_axis,repeat_dims,corr_axis,fixed_corr


    def select_repeated_x(self,x,u_x,param_fixed,i,repeat_axis,n_repeats):
        """
        Select one (index i) of multiple repeated entries and return the input quantities and uncertainties for that entry.

        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param i: index of the input quantity (in x)
        :type i: int
        :param repeat_axis: dimension along which the measurements are repeated
        :type repeat_axis: int
        :param n_repeats: number of repeated measurements
        :type n_repeats: int
        :return: list of input quantities, list of uncertainties for single measurement
        :rtype: list[array]. list[array]
        """
        xb=np.zeros(len(x),dtype=object)
        u_xb=np.zeros(len(u_x),dtype=object)
        for j in range(len(x)):
            selected=False
            if param_fixed is not None:
                if param_fixed[j] == True:
                    xb[j] = x[j]
                    u_xb[j] = u_x[j]
                    selected=True
            if not selected:
                if len(x[j].shape) > repeat_axis:
                    if (x[j].shape[repeat_axis]!=n_repeats):
                        xb[j] = x[j]
                        u_xb[j] = u_x[j]
                    elif repeat_axis == 0 :
                        xb[j]=x[j][i]
                        u_xb[j] = u_x[j][i]
                    elif repeat_axis == 1:
                        xb[j] = x[j][:,i]
                        u_xb[j] = u_x[j][:,i]
                    elif repeat_axis == 2:
                        xb[j] = x[j][:,:,i]
                        u_xb[j] = u_x[j][:,:,i]
                    else:
                        warnings.warn("The repeat axis is too large to be dealt with by the"
                                      "current version of punpy.")
                else:
                    if (len(x[j])==n_repeats):
                        xb[j] = x[j][i]
                        u_xb[j] = u_x[j][i]
                    else:
                        xb[j] = x[j]
                        u_xb[j] = u_x[j]
        return xb, u_xb

    def combine_repeated_outs(self,outs,yshape,n_repeats,lenx,repeat_axis,return_corr,return_samples,output_vars):
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
        if not return_corr and not return_samples:
            if output_vars == 1:
                u_func = np.zeros(yshape)
                if repeat_axis == 0:
                    for i in range(len(outs)):
                        u_func[i] = outs[i]
                elif repeat_axis == 1:
                    for i in range(len(outs)):
                        if len(outs[i].shape)>1:
                            if outs[i].shape[1]==1:
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
                        if len(outs[i][0].shape)>1:
                            if outs[i][0].shape[1]==1:
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
            returns[0]=u_func
            extra_index=0
            if return_corr:
                corr=np.mean([outs[i][1] for i in range(n_repeats)],axis=0)
                if output_vars > 1:
                    for j in range(output_vars):
                        if not util.isPD(corr[j]):
                            corr[j] = util.nearestPD_cholesky(corr[j] ,corr=True,
                                                           return_cholesky=False)
                else:
                    if not util.isPD(corr):
                        corr=util.nearestPD_cholesky(corr,corr=True,return_cholesky=False)
                returns[1]=corr
                extra_index+=1

            if output_vars>1:
                corr_out = np.mean([outs[i][1+extra_index] for i in range(n_repeats)],axis=0)
                if not util.isPD(corr_out):
                    corr_out=util.nearestPD_cholesky(corr_out,corr=True,return_cholesky=False)
                returns[1+extra_index]=corr_out
                extra_index+=1

            if return_samples:
                returns[1+extra_index] = [np.vstack([outs[i][1+extra_index][k] for i in range(n_repeats)]) for k in range(lenx)]
                returns[2+extra_index] = [np.vstack([outs[i][2+extra_index][k] for i in range(n_repeats)]) for k in range(lenx)]

            return returns

    def process_samples(self,func,data,return_corr,return_samples,corr_axis=-99,
                        fixed_corr=None,PD_corr=True,output_vars=1):
        """
        Run the MC-generated samples of input quantities through the measurement function and calculate
        correlation matrix if required.

        :param func: measurement function
        :type func: function
        :param data: MC-generated samples of input quantities
        :type data: array[array]
        :param return_corr: set to True to return correlation matrix of measurand
        :type return_corr: bool
        :param return_samples: set to True to return generated samples
        :type return_samples: bool
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param fixed_corr: correlation matrix to be copied without changing, defaults to None (correlation matrix is calculated rather than copied)
        :type fixed_corr: array
        :param PD_corr: set to True to make sure returned correlation matrices are positive semi-definite, default to True
        :type PD_corr: bool, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        if self.parallel_cores==0:
            MC_y = np.array(func(*data))
        elif self.parallel_cores==1:
            # In order to Process the MC iterations separately, the array with the input quantities has to be reordered
            # so that it has the same length (i.e. the first dimension) as the number of MCsteps.
            # First we move the axis with the same length as self.MCsteps from the last dimension to the fist dimension
            data2 = [np.moveaxis(dat,-1,0) for dat in data]
            # The function can then be applied to each of these MCsteps
            MC_y2 = list(map(func,*data2))
            # We then reorder to bring it back to the original shape
            MC_y = np.moveaxis(MC_y2,0,-1)

        else:
            # We again need to reorder the input quantities samples in order to be able to pass them to p.starmap
            # We here use lists to iterate over and order them slightly different as the case above.
            data2=[[data[j][...,i] for j in range(len(data))] for i in range(self.MCsteps)]
            pool = Pool(self.parallel_cores)
            MC_y2=np.array(pool.starmap(func,data2))
            del pool
            MC_y = np.moveaxis(MC_y2,0,-1)

        u_func = np.std(MC_y,axis=-1)

        if not return_corr:
            if return_samples:
                return u_func,MC_y,data
            else:
                return u_func
        else:
            if output_vars==1:
                if fixed_corr is None:
                    corr_y = self.calculate_corr(MC_y,corr_axis).astype("float32")
                    if PD_corr:
                        if not util.isPD(corr_y):
                            corr_y = util.nearestPD_cholesky(corr_y,corr=True,
                                                           return_cholesky=False)
                else:
                    corr_y = fixed_corr
                if return_samples:
                    return u_func,corr_y,MC_y,data
                else:
                    return u_func,corr_y

            else:
                #create an empty arrays and then populate it with the correlation matrix for each output parameter individually
                corr_ys=np.empty(output_vars,dtype=object)
                for i in range(output_vars):
                    if fixed_corr is None:
                        corr_ys[i] = self.calculate_corr(MC_y[i],corr_axis)\
                                            .astype("float32")
                        if PD_corr:
                            if not util.isPD(corr_ys[i]):
                                corr_ys[i] = util.nearestPD_cholesky(corr_ys[i],
                                                        corr=True,return_cholesky=False)
                    else:
                        corr_ys[i] = fixed_corr
                #calculate correlation matrix between the different outputs produced by the measurement function.
                corr_out=np.corrcoef(MC_y.reshape((output_vars,-1))).astype("float32")
                if PD_corr:
                    if not util.isPD(corr_out):
                        corr_out = util.nearestPD_cholesky(corr_out,corr=True,
                                                         return_cholesky=False)
                if return_samples:
                    return u_func,corr_ys,corr_out,MC_y,data
                else:
                    return u_func,corr_ys,corr_out

    def calculate_corr(self,MC_y,corr_axis=-99):
        """
        Calculate the correlation matrix between the MC-generated samples of output quantities.
        If corr_axis is specified, this axis will be the one used to calculate the correlation matrix (e.g. if corr_axis=0 and x.shape[0]=n, the correlation matrix will have shape (n,n)).
        This will be done for each combination of parameters in the other dimensions and the resulting correlation matrices are averaged.

        :param MC_y: MC-generated samples of the output quantities (measurands)
        :type MC_y: array
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :return: correlation matrix
        :rtype: array
        """
        #print("the shape is:",MC_y.shape)

        if len(MC_y.shape) <3:
                corr_y = np.corrcoef(MC_y)

        elif len(MC_y.shape) == 3:
            if corr_axis==0:
                corr_ys = np.empty(len(MC_y[0]),dtype=object)
                for i in range(len(MC_y[0])):
                    corr_ys[i] = np.corrcoef(MC_y[:,i])
                corr_y = np.mean(corr_ys,axis=0)

            elif corr_axis==1:
                corr_ys = np.empty(len(MC_y),dtype=object)
                for i in range(len(MC_y)):
                    corr_ys[i] = np.corrcoef(MC_y[i])
                corr_y = np.mean(corr_ys,axis=0)

            else:
                MC_y = MC_y.reshape((MC_y.shape[0]*MC_y.shape[1],self.MCsteps))
                corr_y = np.corrcoef(MC_y)

        elif len(MC_y.shape) == 4:
            if corr_axis == 0:
                corr_ys = np.empty(len(MC_y[0])*len(MC_y[0,0]),dtype=object)
                for i in range(len(MC_y[0])):
                    for j in range(len(MC_y[0,0])):
                        corr_ys[i+j*len(MC_y[0])] = np.corrcoef(MC_y[:,i,j])
                corr_y = np.mean(corr_ys,axis=0)

            elif corr_axis == 1:
                corr_ys = np.empty(len(MC_y)*len(MC_y[0,0]),dtype=object)
                for i in range(len(MC_y)):
                    for j in range(len(MC_y[0,0])):
                        corr_ys[i+j*len(MC_y)] = np.corrcoef(MC_y[i,:,j])
                corr_y = np.mean(corr_ys,axis=0)

            elif corr_axis == 2:
                corr_ys = np.empty(len(MC_y)*len(MC_y[0]),dtype=object)
                for i in range(len(MC_y)):
                    for j in range(len(MC_y[0])):
                        corr_ys[i+j*len(MC_y)] = np.corrcoef(MC_y[i,j])
                corr_y = np.mean(corr_ys,axis=0)
            else:
                MC_y = MC_y.reshape((MC_y.shape[0]*MC_y.shape[1]*MC_y.shape[2],self.MCsteps))
                corr_y = np.corrcoef(MC_y)
        else:
            print("MC_y has too high dimensions. Reduce the dimensionality of the input data")
            exit()

        return corr_y

    def generate_samples_correlated(self,x,u_x,corr_x,i):
        """
        Generate correlated MC samples of input quantity with given uncertainties and correlation matrix.
        Samples are generated using generate_samples_cov() after matching up the uncertainties to the right correlation matrix.
        It is possible to provide one correlation matrix to be used for each measurement (which each have an uncertainty) or a correlation matrix per measurement.

        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_x: list of correlation matrices (n,n) along non-repeating axis, or list of correlation matrices for each repeated measurement.
        :type corr_x: list[array], optional
        :param i: index of the input quantity (in x)
        :type i: int
        :return: generated samples
        :rtype: array
        """
        if (len(x[i].shape) == 2):
            if len(corr_x[i]) == len(u_x[i]):
                MC_data = np.zeros((u_x[i].shape)+(self.MCsteps,))
                for j in range(len(u_x[i][0])):
                    cov_x = util.convert_corr_to_cov(corr_x[i],u_x[i][:,j])
                    MC_data[:,j,:] = self.generate_samples_cov(x[i][:,j].flatten(),
                                     cov_x).reshape(x[i][:,j].shape+(self.MCsteps,))
            else:
                MC_data = np.zeros((u_x[i].shape)+(self.MCsteps,))
                for j in range(len(u_x[i][:,0])):
                    cov_x = util.convert_corr_to_cov(corr_x[i],u_x[i][j])
                    MC_data[j,:,:] = self.generate_samples_cov(x[i][j].flatten(),
                                     cov_x).reshape(x[i][j].shape+(self.MCsteps,))
        else:
            cov_x = util.convert_corr_to_cov(corr_x[i],u_x[i])
            MC_data = self.generate_samples_cov(x[i].flatten(),cov_x).reshape(
                        x[i].shape+(self.MCsteps,))

        return MC_data

    def generate_samples_random(self,param,u_param):
        """
        Generate MC samples of input quantity with random (Gaussian) uncertainties.

        :param param: values of input quantity (mean of distribution)
        :type param: float or array
        :param u_param: uncertainties on input quantity (std of distribution)
        :type u_param: float or array
        :return: generated samples
        :rtype: array
        """
        if not hasattr(param,"__len__"):
            return np.random.normal(size=self.MCsteps)*u_param+param
        elif len(param.shape) == 1:
            return np.random.normal(size=(len(param),self.MCsteps))*u_param[:,None]+param[:,None]
        elif len(param.shape) == 2:
            return np.random.normal(size=param.shape+(self.MCsteps,))*u_param[:,:,None]+param[:,:,None]
        elif len(param.shape) == 3:
            return np.random.normal(size=param.shape+(self.MCsteps,))*u_param[:,:,:,None]+param[:,:,:,None]
        else:
            print("parameter shape not supported")
            exit()


    def generate_samples_systematic(self,param,u_param):
        """
        Generate correlated MC samples of input quantity with systematic (Gaussian) uncertainties.

        :param param: values of input quantity (mean of distribution)
        :type param: float or array
        :param u_param: uncertainties on input quantity (std of distribution)
        :type u_param: float or array
        :return: generated samples
        :rtype: array
        """
        if not hasattr(param,"__len__"):
            return np.random.normal(size=self.MCsteps)*u_param+param
        elif len(param.shape) == 1:
            return np.dot(u_param[:,None],np.random.normal(size=self.MCsteps)[None,:])+param[:,None]
        elif len(param.shape) == 2:
            return np.dot(u_param[:,:,None],np.random.normal(size=self.MCsteps)[:,None,None])[:,:,:,0]+param[:,:,None]
        elif len(param.shape) == 3:
            return np.dot(u_param[:,:,:,None],np.random.normal(size=self.MCsteps)[:,None,None,None])[:,:,:,:,0,0]+param[:,:,:,None]
        else:
            print("parameter shape not supported")
            exit()

    def generate_samples_cov(self,param,cov_param):
        """
        Generate correlated MC samples of input quantity with a given covariance matrix.
        Samples are generated independent and then correlated using Cholesky decomposition.

        :param param: values of input quantity (mean of distribution)
        :type param: array
        :param cov_param: covariance matrix for input quantity
        :type cov_param: array
        :return: generated samples
        :rtype: array
        """
        try:
            L = np.linalg.cholesky(cov_param)
        except:
            L = util.nearestPD_cholesky(cov_param)

        return np.dot(L,np.random.normal(size=(len(param),self.MCsteps)))+param[:,None]

    def correlate_samples_corr(self,samples,corr):
        """
        Method to correlate independent samples of input quantities using correlation matrix and Cholesky decomposition.

        :param samples: independent samples of input quantities
        :type samples: array[array]
        :param corr: correlation matrix between input quantities
        :type corr: array
        :return: correlated samples of input quantities
        :rtype: array[array]
        """
        if np.max(corr) > 1.000001 or len(corr) != len(samples):
            raise ValueError("The correlation matrix between variables is not the right shape or has elements >1.")
        else:
            try:
                L = np.array(np.linalg.cholesky(corr))
            except:
                L = util.nearestPD_cholesky(corr)

            #Cholesky needs to be applied to Gaussian distributions with mean=0 and std=1,
            #We first calculate the mean and std for each input quantity
            means = np.array([np.mean(samples[i]) for i in range(len(samples))])
            stds = np.array([np.std(samples[i]) for i in range(len(samples))])

            #We normalise the samples with the mean and std, then apply Cholesky, and finally reapply the mean and std.
            if all(stds!=0):
                return np.dot(L,(samples-means)/stds)*stds+means

            #If any of the variables has no uncertainty, the normalisation will fail. Instead we leave the parameters without uncertainty unchanged.
            else:
                samples_out=samples[:]
                id_nonzero=np.where(stds!=0)
                samples_out[id_nonzero]=np.dot(L[id_nonzero][:,id_nonzero],(samples[id_nonzero]-means[id_nonzero])/stds[id_nonzero])[:,0]*stds[id_nonzero]+means[id_nonzero]
                return samples_out