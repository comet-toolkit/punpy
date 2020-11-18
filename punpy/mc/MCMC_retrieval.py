"""Run MCMC for retrieval"""

'''___Built-In Modules___'''

'''___Third-Party Modules___'''
import numpy as np
import emcee
from multiprocessing import Pool
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"

'''___NPL Modules___'''

'''___Authorship___'''
__author__ = "Pieter De Vis"
__created__ = "01/03/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

inds_cache = {}


# lock = threading.Lock()
class MCMCRetrieval:
    def __init__(self,measurement_function,observed,syst_uncertainty=None,rand_uncertainty=None,cov=None,parallel_cores=4,uplims=+np.inf,downlims=-np.inf):
        self.measurement_function=measurement_function
        self.observed = observed
        self.rand_uncertainty = np.array([rand_uncertainty])
        self.syst_uncertainty = np.array([syst_uncertainty])
        if cov is None:
            self.invcov = cov
        else:
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
        self.uplims = np.array(uplims)
        self.downlims = np.array(downlims)
        self.parallel_cores=parallel_cores

    def run_retrieval(self,theta_0,nwalkers,steps,burn_in,return_samples=True,return_corr=False):
        # if self.syst_uncertainty is not None:
        #     theta_0=np.append([0.],theta_0)
        ndimw = len(theta_0)
        pos = [theta_0*np.random.normal(1.0,0.1,theta_0.shape)+np.random.normal(0.0,0.001,theta_0.shape) for i in
               range(nwalkers)]
        #print(self.measurement_function(theta_0))
        p = Pool(self.parallel_cores)
        sampler = emcee.EnsembleSampler(nwalkers,ndimw,self.lnprob,pool=p)
        sampler.run_mcmc(pos,steps,progress=False)

        samples = sampler.chain[:,:,:].reshape((-1,ndimw))[burn_in::]
        medians = np.median(samples,axis=0)
        unc_up = (np.percentile(samples,84,axis=0)-medians)
        unc_down = -(np.percentile(samples,16,axis=0)-medians)
        unc_avg= (unc_up+unc_down)/2.
        corr=(np.corrcoef(samples.T))


        if return_samples:
            if return_corr:
                return medians, unc_avg, corr, samples
            else:
                return medians,unc_avg,samples

        else:
            if return_corr:
                return medians, unc_avg, corr
            else:
                return medians,unc_avg

    def find_chisum(self,theta):
        model=self.measurement_function(*theta)
        diff = model-self.observed
        if np.isfinite(np.sum(diff)):
            if self.invcov is None:
                return np.sum((diff)**2/self.rand_uncertainty**2)
            else:
                #print(diff,np.linalg.inv(self.cov),np.dot(np.dot(diff.T,self.invcov),diff))
                return np.dot(np.dot(diff.T,self.invcov),diff)
        else:
            return np.inf

    # def upper_triangular_to_symmetric(self,ut):
    #     n = ut.shape[0]
    #     try:
    #         inds = inds_cache[n]
    #     except KeyError:
    #         inds = np.tri(n, k=-1, dtype=np.bool)
    #         inds_cache[n] = inds
    #     ut[inds] = ut.T[inds]

    # def fast_positive_definite_inverse(self,m):
    #     cholesky, info = lapack.dpotrf(m)
    #     if info != 0:
    #         raise ValueError('dpotrf failed on input {}'.format(m))
    #     inv, info = lapack.dpotri(cholesky)
    #     if info != 0:
    #         raise ValueError('dpotri failed on input {}'.format(cholesky))
    #     self.upper_triangular_to_symmetric(inv)
    #     return inv

    def lnlike(self,theta):
        #print(theta,self.find_chisum(theta))
        return -0.5*(self.find_chisum(theta))

    def lnprior(self,theta):
        if all(self.downlims < theta[1::]) and all(self.uplims > theta[1::]):
            #if self.syst_uncertainty[0] is None:
                return 0
            # else:
            #     return -0.5*(theta[0]**2/self.syst_uncertainty**2)
        else:
            return -np.inf

    def lnprob(self,theta):
        lp_prior = self.lnprior(theta)
        if not np.isfinite(lp_prior):
            return -np.inf
        lp = self.lnlike(theta)
        # if lp < -99999999999:
        #     return -np.inf
        return lp_prior+lp
