"""Use Monte Carlo to propagate uncertainties"""

import numpy as np
from multiprocessing import Pool

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

    def propagate_random(self,func,x,u_x,corr_between=None,return_corr=False,return_samples=False,corr_axis=-99,output_vars=1):
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
        MC_data = np.empty(len(x),dtype=np.ndarray)
        for i in range(len(x)):
            if u_x[i] is None:
                u_x[i]=np.zeros_like(x[i])
            #print(x[i].shape,u_x[i].shape)
            MC_data[i] = self.generate_samples_random(x[i],u_x[i])
            #print(MC_data[i].nbytes)

        if corr_between is not None:
            MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,output_vars)

    def propagate_systematic(self,func,x,u_x,corr_between=None,return_corr=False,return_samples=False,corr_axis=-99,output_vars=1):
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
        MC_data = np.empty(len(x),dtype=np.ndarray)
        for i in range(len(x)):
            if u_x[i] is None:
                u_x[i] = np.zeros_like(x[i])

            MC_data[i] = self.generate_samples_systematic(x[i],u_x[i])

        if corr_between is not None:
            MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,output_vars)

    def propagate_both(self,func,x,u_x_rand,u_x_syst,corr_between=None,return_corr=True,return_samples=False,corr_axis=-99,output_vars=1):
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
        MC_data = np.empty(len(x),dtype=np.ndarray)
        for i in range(len(x)):
            if u_x_rand[i] is None:
                u_x_rand[i] = np.zeros_like(x[i])
            if u_x_syst[i] is None:
                u_x_syst[i] = np.zeros_like(x[i])

            MC_data[i] = self.generate_samples_both(x[i],u_x_rand[i],u_x_syst[i])

        if corr_between is not None:
            MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,output_vars)

    def propagate_type(self,func,x,u_x,u_type,corr_between=None,return_corr=True,return_samples=False,corr_axis=-99,output_vars=1):
        """
        Propagate random or systematic uncertainties through measurement function with n input quantities.
        Input quantities can be floats, vectors or images.

        :param func: measurement function
        :type func: function
        :param x: list of input quantities (usually numpy arrays)
        :type x: list[array]
        :param u_x: list of uncertainties on input quantities (usually numpy arrays)
        :type u_x: list[array]
        :param u_type: sting identifiers whether uncertainties are random or systematic
        :type u_type: list[str]
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
        MC_data = np.empty(len(x),dtype=np.ndarray)
        for i in range(len(x)):
            if u_x[i] is None:
                u_x[i] = np.zeros_like(x[i])
            if u_type[i].lower() == 'rand' or u_type[i].lower() == 'random' or u_type[i].lower() == 'r':
                MC_data[i] = self.generate_samples_random(x[i],u_x[i])
            elif u_type[i].lower() == 'syst' or u_type[i].lower() == 'systematic' or u_type[i].lower() == 's':
                MC_data[i] = self.generate_samples_systematic(x[i],u_x[i])
            else:
                raise ValueError(
                    'Uncertainty type not understood. Use random ("random", "rand" or "r") or systematic ("systematic", "syst" or "s").')

        if corr_between is not None:
            MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,output_vars)

    def propagate_cov(self,func,x,cov_x,corr_between=None,return_corr=True,return_samples=False,corr_axis=-99,output_vars=1):
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
        :param return_samples: set to True to return generated samples, defaults to False
        :type return_samples: bool, optional
        :param corr_axis: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_axis: integer, optional
        :param output_vars: number of output parameters in the measurement function. Defaults to 1.
        :type output_vars: integer, optional
        :return: uncertainties on measurand
        :rtype: array
        """
        MC_data = np.empty(len(x),dtype=np.ndarray)
        for i in range(len(x)):
            if not hasattr(x[i],"__len__"):
                MC_data[i] = self.generate_samples_systematic(x[i],cov_x[i])
            elif (all((cov_x[i]==0).flatten())): #This is the case if one of the variables has no uncertainty
                MC_data[i] = np.tile(x[i].flatten(),(self.MCsteps,1)).T
            else:
                MC_data[i] = self.generate_samples_cov(x[i].flatten(),cov_x[i]).reshape(x[i].shape+(self.MCsteps,))
        if corr_between is not None:
            MC_data = self.correlate_samples_corr(MC_data,corr_between)

        return self.process_samples(func,MC_data,return_corr,return_samples,corr_axis,output_vars)

    def process_samples(self,func,data,return_corr,return_samples,corr_axis=-99,output_vars=1):
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
            with Pool(self.parallel_cores) as p:
                MC_y2=np.array(p.starmap(func,data2))
            MC_y = np.moveaxis(MC_y2,0,-1)

        u_func = np.std(MC_y,axis=-1)
        if not return_corr:
            if return_samples:
                return u_func,MC_y,data
            else:
                return u_func
        else:
            if output_vars==1:
                corr_y = self.calculate_corr(MC_y,corr_axis)
                if return_samples:
                    return u_func,corr_y,MC_y,data
                else:
                    return u_func,corr_y

            else:
                #create an empty arrays and then populate it with the correlation matrix for each output parameter individually
                corr_ys=np.empty(output_vars,dtype=object)
                for i in range(output_vars):
                    corr_ys[i] = self.calculate_corr(MC_y[i],corr_axis)

                #calculate correlation matrix between the different outputs produced by the measurement function.
                corr_out=np.corrcoef(MC_y.reshape((output_vars,-1)))

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

    def generate_samples_both(self,param,u_param_rand,u_param_syst):
        """
        Generate correlated MC samples of the input quantity with random and systematic (Gaussian) uncertainties.

        :param param: values of input quantity (mean of distribution)
        :type param: float or array
        :param u_param_rand: random uncertainties on input quantity (std of distribution)
        :type u_param_rand: float or array
        :param u_param_syst: systematic uncertainties on input quantity (std of distribution)
        :type u_param_syst: float or array
        :return: generated samples
        :rtype: array
        """
        if not hasattr(param,"__len__"):
            return np.random.normal(size=self.MCsteps)*u_param_rand+np.random.normal(
                size=self.MCsteps)*u_param_syst+param
        elif len(param.shape) == 1:
            return np.random.normal(size=(len(param),self.MCsteps))*u_param_rand[:,None]+np.dot(u_param_syst[:,None],
                np.random.normal(size=self.MCsteps)[None,:])+param[:,None]
        elif len(param.shape) == 2:
            return np.random.normal(size=param.shape+(self.MCsteps,))*u_param_rand[:,:,None]+np.dot(
                u_param_syst[:,:,None],np.random.normal(size=self.MCsteps)[:,None,None])[:,:,:,0]+param[:,:,None]
        elif len(param.shape) == 3:
            return np.random.normal(size=param.shape+(self.MCsteps,))*u_param_rand[:,:,:,None]+np.dot(u_param_syst[:,:,:,None],np.random.normal(size=self.MCsteps)[:,None,None,None])[:,:,:,:,0,0]+param[:,:,:,None]
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
            L = self.nearestPD_cholesky(cov_param)

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
        if np.max(corr) > 1 or len(corr) != len(samples):
            raise ValueError("The correlation matrix between variables is not the right shape or has elements >1.")
        else:
            try:
                L = np.array(np.linalg.cholesky(corr))
            except:
                L = self.nearestPD_cholesky(corr)

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

    @staticmethod
    def nearestPD_cholesky(A):
        """
        Find the nearest positive-definite matrix

        :param A: correlation matrix or covariance matrix
        :type A: array
        :return: nearest positive-definite matrix
        :rtype: array

        Copied and adapted from [1] under BSD license.
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], which
        credits [3].
        [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
        [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A+A.T)/2
        _,s,V = np.linalg.svd(B)

        H = np.dot(V.T,np.dot(np.diag(s),V))

        A2 = (B+H)/2

        A3 = (A2+A2.T)/2

        try:
            return np.linalg.cholesky(A3)
        except:

            spacing = np.spacing(np.linalg.norm(A))

            I = np.eye(A.shape[0])
            k = 1
            while not MCPropagation.isPD(A3):
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I*(-mineig*k**2+spacing)
                k += 1

            if np.any(abs(A-A3)/(A+0.0001) > 0.0001):
                raise ValueError(
                    "One of the provided covariance matrix is not postive definite. Covariance matrices need to be at least positive semi-definite. Please check your covariance matrix.")
            else:
                print(
                    "One of the provided covariance matrix is not positive definite. It has been slightly changed (less than 0.01% in any element) to accomodate our method.")
                return np.linalg.cholesky(A3)

    @staticmethod
    def isPD(B):
        """
        Returns true when input is positive-definite, via Cholesky

        :param B: matrix
        :type B: array
        :return: true when input is positive-definite
        :rtype: bool
        """
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def convert_corr_to_cov(corr,u):
        """
        Convert correlation matrix to covariance matrix

        :param corr: correlation matrix
        :type corr: array
        :param u: uncertainties
        :type u: array
        :return: covariance matrix
        :rtype: array
        """
        return u.flatten()*corr*u.flatten().T

    @staticmethod
    def convert_cov_to_corr(cov,u):
        """
        Convert covariance matrix to correlation matrix

        :param corr: covariance matrix
        :type corr: array
        :param u: uncertainties
        :type u: array
        :return: correlation matrix
        :rtype: array
        """
        return 1/u.flatten()*cov/u.flatten().T