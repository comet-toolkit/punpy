"""Use Monte Carlo to propagate uncertainties"""

from abc import ABC,abstractmethod

import numpy as np

import punpy.fiduceo.fiduceo_correlations as fc
from punpy.lpu.lpu_propagation import LPUPropagation
from punpy.mc.mc_propagation import MCPropagation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class FiduceoMeasurementFunction(ABC):
    def __init__(self,variables=[],corr_between=None,param_fixed=None,output_vars=1,repeat_dims=-99,corr_axis=-99,mc=True,steps=10000,parallel_cores=0,dtype=None, Jx_diag=False, step=None):
        """
        Initialise FiduceoMeasurementFunction
        """
        self.variables=variables
        if mc:
            self.prop=MCPropagation(steps, parallel_cores, dtype)
        else:
            self.prop=LPUPropagation(parallel_cores, Jx_diag, step)

        self.corr_between=corr_between
        self.param_fixed=param_fixed
        self.output_vars=output_vars
        self.repeat_dims=repeat_dims
        self.corr_axis=corr_axis


    @abstractmethod
    def function(self):
        pass

    def run(self,*args,expand=True):
        input_qty = self.get_input_qty(args,expand=expand)
        if self.repeat_dims is None:
            return self.function(*input_qty)
        else:
            return self.function(*input_qty)


    def propagate_u(self,form,*args,expand=True):
        input_qty=self.get_input_qty(args,expand=expand)
        input_unc = self.get_input_unc(form,args,expand=expand)
        input_corr = self.get_input_corr(form,args)
        return self.prop.propagate_standard(self.function,input_qty,input_unc,input_corr,
                                     param_fixed=self.param_fixed,
                                     corr_between=self.corr_between,
                                     return_corr=False,return_samples=False,
                                     repeat_dims=self.repeat_dims,
                                     corr_axis=self.corr_axis,
                                     output_vars=self.output_vars)

    def get_input_qty(self,*args,expand=True):
        if len(self.variables)==0:
            raise ValueError("Variables have not been specified.")
            exit()
        else:
            inputs = np.empty(len(self.variables),dtype=object)
            for iv,var in enumerate(self.variables):
                found = False
                for dataset in args[0]:
                    try:
                        inputs[iv]=dataset[var].values
                        found = True
                    except:
                        continue
                if not found:
                    raise ValueError("Variable %s not found in provided datasets."%(var))

            if expand:
                datashape = inputs[0].shape
                for i in range(len(inputs)):
                    if len(inputs[i].shape) < len(datashape):
                        if inputs[i].shape[0] == datashape[1]:
                            inputs[i] = np.tile(inputs[i],(datashape[0],1))
                        elif inputs[i].shape[0] == datashape[0]:
                            inputs[i] = np.tile(inputs[i],(datashape[1],1)).T

            return inputs

    def get_input_unc(self,form,*args,expand=True):
        inputs_unc = np.empty(len(self.variables), dtype=object)
        for iv,var in enumerate(self.variables):
            uvar = "u_%s_%s"%(form,var)
            found = False
            for dataset in args[0]:
                try:
                    inputs_unc[iv]=dataset[uvar].values
                    if "rel_" in form:
                        inputs_unc[iv] *= dataset[var].values
                    found = True
                except:
                    continue
            if not found:
                inputs_unc[iv] = None
                print(
                    "%s uncertainty for variable %s (%s) not found in provided datasets. Zero uncertainty assumed."%(form,var,uvar))

        if expand:
            datashape = inputs_unc[0].shape
            for i in range(len(inputs_unc)):
                if inputs_unc[i] is not None:
                    if len(inputs_unc[i].shape) < len(datashape):
                        inputs_unc[i] = np.tile(
                            inputs_unc[i],(datashape[1],1)).T
        return inputs_unc

    def get_input_corr(self,form,*args):
        inputs_corr = np.empty(len(self.variables),dtype=object)
        for iv,var in enumerate(self.variables):
            found = False
            for dataset in args[0]:
                try:
                    inputs_corr[iv] = fc.calculate_corr(var,form,dataset,self.repeat_dims)
                    found = True
                except:
                    continue
            if not found:
                raise ValueError("Correlation for variable %s not found in provided datasets."%(var))

        return inputs_corr

