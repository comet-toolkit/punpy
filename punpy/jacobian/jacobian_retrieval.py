"""Run MCMC for retrieval"""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import numpy as np
import emcee
from multiprocessing import Pool
import time
import punpy.utilities.utilities as util
import os
from scipy.optimize import minimize

os.environ["OMP_NUM_THREADS"] = "1"

"""___NPL Modules___"""

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/03/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

inds_cache = {}


# lock = threading.Lock()
class JacobianRetrieval:
    def __init__(
        self,
        measurement_function,
        observed,
        syst_uncertainty=None,
        rand_uncertainty=None,
        cov=None,
        uplims=+np.inf,
        downlims=-np.inf,
    ):
        self.measurement_function = measurement_function
        self.observed = observed
        self.rand_uncertainty = np.array([rand_uncertainty])
        self.syst_uncertainty = np.array([syst_uncertainty])
        if cov is None:
            self.invcov = cov
        else:
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
            # print(observed,cov,self.invcov)

        self.uplims = np.array(uplims)
        self.downlims = np.array(downlims)

    def run_retrieval(self, theta_0, return_corr=True):
        res = minimize(self.find_chisum, theta_0)
        Jx = util.calculate_Jacobian(self.measurement_function, res.x)
        # print("wer",res.x,theta_0,Jx,util.calculate_Jacobian(self.measurement_function,theta_0))

        return tuple(res.x) + tuple(self.process_inverse_jacobian(Jx, return_corr))

    def process_inverse_jacobian(self, J, return_corr=True):
        print(
            self.invcov,
            J.T,
            np.dot(J.T, self.invcov),
            np.dot(np.dot(J.T, self.invcov), J),
            np.linalg.inv(np.dot(np.dot(J.T, self.invcov), J)),
        )
        covx = np.linalg.inv(np.dot(np.dot(J.T, self.invcov), J))
        u_func = np.diag(covx) ** 0.5
        corr_x = util.convert_cov_to_corr(covx, u_func)
        if return_corr:
            return u_func, corr_x
        else:
            return u_func
        # if not return_corr:
        #     return u_func.reshape(shape_y)
        # else:
        #     if output_vars == 1:
        #         return u_func.reshape(shape_y),corr_y
        #     else:
        #         #create an empty arrays and then populate it with the correlation matrix for each output parameter individually
        #         corr_ys = np.empty(output_vars,dtype=object)
        #         for i in range(output_vars):
        #             corr_ys[i] = corr_y[int(i*len(corr_y)/output_vars):
        #                                 int((i+1)*len(corr_y)/output_vars),
        #                          int(i*len(corr_y)/output_vars):
        #                          int((i+1)*len(corr_y)/output_vars)]
        #
        #         # #calculate correlation matrix between the different outputs produced by the measurement function.
        #         # corr_out=np.corrcoef(MC_y.reshape((output_vars,-1)))
        #
        #         return u_func.reshape(shape_y),corr_ys  #,corr_out

    def find_chisum(self, theta):
        model = self.measurement_function(theta)
        diff = model - self.observed
        if np.isfinite(np.sum(diff)):
            if self.invcov is None:
                return np.sum((diff) ** 2 / self.rand_uncertainty ** 2)
            else:
                # print(diff,np.linalg.inv(self.cov),np.dot(np.dot(diff.T,self.invcov),diff))
                return np.dot(np.dot(diff.T, self.invcov), diff)
        else:
            return np.inf
