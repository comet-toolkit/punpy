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

