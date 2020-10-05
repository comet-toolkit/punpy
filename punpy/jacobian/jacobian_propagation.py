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

        :param steps: number of MC iterations
        :type steps: int
        """

        self.parallel_cores = parallel_cores

    def propagate_random(self,func,x,u_x,corr_between=None,return_corr=False,corr_axis=-99,output_vars=1):
        """
        Propagate random uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors or images.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of random uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        yshape=np.shape(func(*x))
        if output_vars==1:
            fun = lambda c: func(*c.reshape(len(x),-1))
        else:
            fun = lambda c: np.concatenate(func(*c.reshape(len(x),-1)))
        Jx=util.calculate_Jacobian(fun,x)
        if corr_between is None:
            corr_x = np.eye(len(x.flatten()))
        else:
            corrs=[np.eye(len(xi.flatten())) for xi in x]
            corr_x=self.calculate_flattened_corr(corrs,corr_between)
        cov_x=util.convert_corr_to_cov(corr_x,u_x)
        return self.process_jacobian(Jx,cov_x,yshape,return_corr,corr_axis,output_vars)

    def calculate_flattened_corr(self,corrs,corr_between):
        totcorrlen=0
        for i in range(len(corrs)):
            totcorrlen += len(corrs[i])
        totcorr=np.eye(totcorrlen)
        for i in range(len(corrs)):
            for j in range(len(corrs)):
                totcorr[i*len(corrs[i]):(i+1)*len(corrs[i]),j*len(corrs[j]):(j+1)*len(corrs[j])]=corr_between[i,j]*corrs[i]**0.5*corrs[j]**0.5
        return totcorr

    def process_jacobian(self,J,covx,shape_y,return_corr,corr_axis=-99,output_vars=1):
        covy=np.dot(np.dot(J,covx),J.T)
        u_func=np.diag(covy)**0.5
        corr_y=util.convert_cov_to_corr(covy,u_func)
        #print("test",J,covx,covy)
        if not return_corr:
            return u_func.reshape(shape_y)
        else:
            if output_vars==1:
                return u_func.reshape(shape_y),corr_y
            else:
                #create an empty arrays and then populate it with the correlation matrix for each output parameter individually
                corr_ys=np.empty(output_vars,dtype=object)
                for i in range(output_vars):
                    corr_ys[i] = corr_y[int(i*len(corr_y)/output_vars):
                                        int((i+1)*len(corr_y)/output_vars),
                                        int(i*len(corr_y)/output_vars):
                                        int((i+1)*len(corr_y)/output_vars)]

                # #calculate correlation matrix between the different outputs produced by the measurement function.
                # corr_out=np.corrcoef(MC_y.reshape((output_vars,-1)))

                return u_func.reshape(shape_y),corr_ys#,corr_out

    def propagate_systematic(self,func,x,u_x,corr_between=None,return_corr=False,corr_axis=-99,output_vars=1):
        """
        Propagate systematic uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors or images.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to False
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        yshape = np.shape(func(*x))
        if output_vars == 1:
            fun = lambda c:func(*c.reshape(len(x),-1))
        else:
            fun = lambda c:np.concatenate(func(*c.reshape(len(x),-1)))
        Jx = util.calculate_Jacobian(fun,x)
        if corr_between is None:
            corr_x = np.ones((len(x.flatten()),len(x.flatten())))
        else:
            corrs = [np.ones((len(xi.flatten()),len(xi.flatten()))) for xi in x]
            corr_x = self.calculate_flattened_corr(corrs,corr_between)
        cov_x = util.convert_corr_to_cov(corr_x,u_x)
        return self.process_jacobian(Jx,cov_x,yshape,return_corr,corr_axis,output_vars)


    def propagate_both(self,func,x,u_x_rand,u_x_syst,corr_between=None,return_corr=True,corr_axis=-99,output_vars=1):
        """
        Propagate random and systematic uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors or images.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x_rand: list of random uncertainties on input quantities (usually numpy arrays)
        :type u_x_rand: list[array]
        :param u_x_syst: list of systematic uncertainties on input quantities (usually numpy arrays)
        :type u_x_syst: list[array]
        :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        yshape = np.shape(func(*x))
        if output_vars == 1:
            fun = lambda c:func(*c.reshape(len(x),-1))
        else:
            fun = lambda c:np.concatenate(func(*c.reshape(len(x),-1)))
        Jx = util.calculate_Jacobian(fun,x)
        if corr_between is None:
            corr_x = np.eye(len(x.flatten()))
        else:
            corrs = [np.eye(len(xi.flatten())) for xi in x]
            corr_x = self.calculate_flattened_corr(corrs,corr_between)
        cov_x = util.convert_corr_to_cov(corr_x,u_x)
        return self.process_jacobian(Jx,cov_x,yshape,return_corr,corr_axis,output_vars)

    def propagate_cov(self,func,x,cov_x,corr_between=None,return_corr=True,corr_axis=-99,output_vars=1):
        """
        Propagate uncertainties with given covariance matrix through measurement function with n input quantities.
        Input quantities can be floats, vectors or images.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param cov_x: list of covariance matrices on input quantities (usually numpy arrays). In case the input quantity is an array of shape (m,o), the covariance matrix needs to be given as an array of shape (m*o,m*o).
        :type cov_x: list[array]
        :param corr_between: covariance matrix (n,n) between input quantities, defaults to None
        :type corr_between: array, optional
        :param return_corr: set to True to return correlation matrix of measurand, defaults to True
        :type return_corr: bool, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        yshape = np.shape(func(*x))
        if output_vars == 1:
            fun = lambda c:func(*c.reshape(len(x),-1))
        else:
            fun = lambda c:np.concatenate(func(*c.reshape(len(x),-1)))
        Jx = util.calculate_Jacobian(fun,x)
        if corr_between is None:
            corr_between = np.eye(len(x))
        corr_x  = [util.convert_cov_to_corr(corr_x[i],u_x[i]) for i in range(len(x))]
        corr_x_full = self.calculate_flattened_corr(corr_x,corr_between)

        cov_x_full = util.convert_corr_to_cov(corr_x_full,u_x)
        return self.process_jacobian(Jx,cov_x_full,yshape,return_corr,corr_axis,output_vars)
