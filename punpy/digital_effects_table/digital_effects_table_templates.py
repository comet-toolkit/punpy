"""Class to make templates for digital effects tables for measurand"""

from abc import ABC

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class DigitalEffectsTableTemplates(ABC):
    def __init__(
        self,
        yvariable,
        yunit,
        repeat_dims=-99
    ):
        """
        Initialise DigitalEffectsTableTemplates

        :param yvariable: name of the measurand variable
        :type yvariable: string
        :param yunit: unit of the measurand
        :type yunit: string
        :param repeat_dims: set to positive integer(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: integer or list of 2 integers, optional
        """
        self.yvariable = yvariable
        self.yunit = yunit
        if isinstance(repeat_dims, int):
            self.repeat_dims = [repeat_dims]
        else:
            self.repeat_dims = repeat_dims

    def make_template_all(self,dims,u_xvar_ref):
        """
        Make the digital effects table template for the case where random, systematic and structured uncertainties are propagated seperately

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr_list,custom_err_corr_dict,dim_sizes=self.determine_template_shape(dims,u_xvar_ref,"err_corr_str_"+ self.yvariable)

        template = {
            self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "unc_comps": ["u_ran_"+ self.yvariable,"u_sys_"+ self.yvariable,"u_str_"+ self.yvariable]
                }
            },
            "u_ran_"+ self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "err_corr": [
                        {
                            "dim": dim,
                            "form": "random",
                            "params": [],
                            "units": []
                        }
                        for dim in dims
                    ]
                }
            },
            "u_sys_"+ self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "err_corr": [
                        {
                            "dim": dim,
                            "form": "systematic",
                            "params": [],
                            "units": []
                        }
                        for dim in dims
                    ]
                }
            },
            "u_str_"+ self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "err_corr": err_corr_list
                }
            },
        }

        if custom_err_corr_dict is not None:
            template["err_corr_str_"+ self.yvariable]=custom_err_corr_dict

        return template,dim_sizes

    def make_template_tot(self,dims,u_xvar_ref):
        """
        Make the digital effects table template for the case where uncertainties are combined and only the total uncertainty is returned.

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr_list,custom_err_corr_dict,dim_sizes=self.determine_template_shape(dims,u_xvar_ref,"err_corr_tot_"+ self.yvariable)

        template = {
            self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "unc_comps": ["u_tot_"+ self.yvariable]
                }
            },
            "u_tot_"+ self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "err_corr": err_corr_list
                }
            },
        }
        if custom_err_corr_dict is not None:
            template["err_corr_tot_"+ self.yvariable]=custom_err_corr_dict

        return template,dim_sizes

    def determine_template_shape(self,dims,u_xvar_ref,err_corr_name):
        """

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :param err_corr_name:
        :type err_corr_name:
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """

        # define dim_size_dict to specify size of arrays
        dim_sizes = {}
        for dim in dims:
            dim_sizes[dim]=len(u_xvar_ref[dim])

        err_corr_list=[]
        custom_corr_dims=[]
        custom_err_corr_dict=None



        #loop through all dimensions, and copy the ones that are repeat dims
        for i in range(len(dims)):
            if i in self.repeat_dims:
                err_corr_list.append(
                    {
                        "dim": dims[i],
                        "form": u_xvar_ref.attrs["err_corr_"+str(i+1)+"_form"],
                        "params": u_xvar_ref.attrs["err_corr_"+str(i+1)+"_params"],
                        "units": u_xvar_ref.attrs["err_corr_"+str(i+1)+"_units"]
                    }
                )
                if not dims[i]==u_xvar_ref.attrs["err_corr_"+str(i+1)+"_dim"]:
                    raise ValueError
            else:
                custom_corr_dims.append(dims[i])

        # make a combined custom form for the variables that are not repeated dims
        if len(err_corr_list)>0:
            err_corr_list.append(
                {
                    "dim": custom_corr_dims,
                    "form": "custom",
                    "params": [err_corr_name],
                    "units": []
                }
            )

        if len(custom_corr_dims)>0:
            corrdim=''.join(custom_corr_dims)
            dim_sizes[corrdim]=1
            for cust_dim in custom_corr_dims:
                dim_sizes[corrdim]*=len(u_xvar_ref[cust_dim])

            custom_err_corr_dict = {
                "dtype": np.float32,
                "dim": [corrdim, corrdim],
                "attributes": {"units": ""},
            }

        return err_corr_list,custom_err_corr_dict,dim_sizes

