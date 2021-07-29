"""Use Monte Carlo to propagate uncertainties"""

from multiprocessing import Pool

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class FiduceoParser:
    def __init__(self, steps, parallel_cores=0, dtype=None):
        """
        Initialise MC Propagator

        :param steps: number of MC iterations
        :type steps: int
        :param parallel_cores: number of CPU to be used in parallel processing
        :type parallel_cores: int
        """

        self.MCsteps = steps
        self.parallel_cores = parallel_cores
        self.dtype = dtype
        if parallel_cores>1:
            self.pool = Pool(self.parallel_cores)

    def parse_and_propagate_variable(self,func,xarr,corr_between=None,corrdim=None,repeat_dims=None):
        vars=self.parse_vars(func)
        x=self.parse_x(xarr,vars)
        u_x=self.parse_u_x(xarr,vars)

        corr_x=[]
        for ivar,var in enumerate(vars):
            dims=xx
            propdims=[]
            propform=[]
            propparams=[]
            i_repeat_dims=[]
            for idim,dim in enumerate(dims):
                if dim not in repeat_dims:
                    propdims.append(idim)
                    propform.append(form[dim])
                    propparams.append(propparams[dim])
                else:
                    i_repeat_dims.append(idim)

            if all(propform == "random"):
                corr_x.append("rand")

            elif all(propform=="rectangular_absolute"):
                corr_x.append("syst")

            else:
                corr_x=self.calculate_corr_from_form(propdims,propform,propparams)

        u_prop = self.prop.propagate_standard(self,func,x,u_x,corr_x,
            param_fixed = param_fixed,
            corr_between = corr_between,
            return_corr = True,
            return_samples = False,
            repeat_dims = i_repeat_dims,
            corr_axis = corrdim,
            output_vars = output_vars,)

    def calculate_corr_from_form(self,propdims,propform,propparams):
        """

        :param propdims:
        :type propdims:
        :param propform:
        :type propform:
        :param propparams:
        :type propparams:
        :return:
        :rtype:
        """

    def parse_var(self,func):
        vars = func["variables"]
        return vars

    def parse_x(self, xarr,vars):
        x=np.empty(len(vars),dtype=object)
        for ivar,var in enumerate(vars):
            x[ivar]=xarr[vars]
        return x

    def parse_u_x(self,xarr,vars):
        u_x = np.empty(len(vars),dtype=object)
        for ivar,var in enumerate(vars):
            u_x[ivar] = xarr["unc_"+vars]
        return u_x