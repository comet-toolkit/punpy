"""Class to make templates for digital effects tables for measurand"""

from abc import ABC

import numpy as np
import xarray as xr

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class DigitalEffectsTableTemplates(ABC):
    def __init__(self, yvariable, yunit):
        """
        Initialise DigitalEffectsTableTemplates

        :param yvariable: name of the measurand variable
        :type yvariable: string
        :param yunit: unit of the measurand
        :type yunit: string
        """
        self.yvariable = yvariable
        self.yunit = yunit

    def make_template_main(
        self,
        dims,
        dim_sizes,
        str_repeat_corr_dims=[],
        store_unc_percent=False,
        repeat_dim_err_corrs=[],
    ):
        """
        Make the digital effects table template for the case where random, systematic and structured uncertainties are propagated seperately

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :param str_repeat_corr_dims: set to (list of) string(s) with dimension name(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to [], for which there is no reduction in dimensionality..
        :type str_repeat_corr_dims: str or list of str, optional
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr, custom_err_corr = self.set_errcorr_shape(
            dims,
            dim_sizes,
            "err_corr_tot_" + self.yvariable,
            str_repeat_corr_dims=str_repeat_corr_dims,
            repeat_dim_err_corr=repeat_dim_err_corrs,
        )

        if store_unc_percent:
            units = "%"
        else:
            units = self.yunit

        template = {
            self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "unc_comps": [
                        self.make_ucomp_name(
                            "ran", store_unc_percent=store_unc_percent
                        ),
                        self.make_ucomp_name(
                            "sys", store_unc_percent=store_unc_percent
                        ),
                        self.make_ucomp_name(
                            "str", store_unc_percent=store_unc_percent
                        ),
                    ],
                },
            },
            self.make_ucomp_name("ran", store_unc_percent=store_unc_percent): {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": units,
                    "err_corr": [
                        {"dim": dim, "form": "random", "params": [], "units": []}
                        for dim in dims
                    ],
                },
            },
            self.make_ucomp_name("sys", store_unc_percent=store_unc_percent): {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": units,
                    "err_corr": [
                        {"dim": dim, "form": "systematic", "params": [], "units": []}
                        for dim in dims
                    ],
                },
            },
            self.make_ucomp_name("str", store_unc_percent=store_unc_percent): {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {"units": units, "err_corr": err_corr},
            },
        }

        if custom_err_corr is not None:
            template["err_corr_str_" + self.yvariable] = custom_err_corr

        return template

    def make_template_tot(
        self,
        dims,
        dim_sizes,
        str_repeat_corr_dims=[],
        store_unc_percent=False,
        repeat_dim_err_corrs=[],
    ):
        """
        Make the digital effects table template for the case where uncertainties are combined and only the total uncertainty is returned.

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr, custom_err_corr = self.set_errcorr_shape(
            dims,
            dim_sizes,
            "err_corr_tot_" + self.yvariable,
            str_repeat_corr_dims=str_repeat_corr_dims,
            repeat_dim_err_corr=repeat_dim_err_corrs,
        )

        if store_unc_percent:
            units = "%"
        else:
            units = self.yunit

        template = {
            self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "unc_comps": [
                        self.make_ucomp_name("tot", store_unc_percent=store_unc_percent)
                    ],
                },
            },
            self.make_ucomp_name("tot", store_unc_percent=store_unc_percent): {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {"units": units, "err_corr": err_corr},
            },
        }
        if custom_err_corr is not None:
            template["err_corr_tot_" + self.yvariable] = custom_err_corr

        return template

    def make_template_specific(
        self,
        comp_list,
        dims,
        dim_sizes,
        str_repeat_corr_dims=[],
        store_unc_percent=False,
        repeat_dim_err_corrs=[],
        simple_random=True,
        simple_systematic=True,
    ):
        """
        Make the digital effects table template for the case where uncertainties are combined and only the total uncertainty is returned.

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        if store_unc_percent:
            units = "%"
        else:
            units = self.yunit

        template = {
            self.yvariable: {
                "dtype": np.float32,
                "dim": dims,
                "attributes": {
                    "units": self.yunit,
                    "unc_comps": [
                        self.make_ucomp_name(comp, store_unc_percent=store_unc_percent)
                        for comp in comp_list
                    ],
                },
            },
        }

        for ic, comp in enumerate(comp_list):
            if comp == "random" and simple_random:
                template[
                    self.make_ucomp_name(comp, store_unc_percent=store_unc_percent)
                ] = {
                    "dtype": np.float32,
                    "dim": dims,
                    "attributes": {
                        "units": units,
                        "err_corr": [
                            {"dim": dim, "form": "random", "params": [], "units": []}
                            for dim in dims
                        ],
                    },
                }

            elif comp == "systematic" and simple_systematic:
                template[
                    self.make_ucomp_name(comp, store_unc_percent=store_unc_percent)
                ] = {
                    "dtype": np.float32,
                    "dim": dims,
                    "attributes": {
                        "units": units,
                        "err_corr": [
                            {
                                "dim": dim,
                                "form": "systematic",
                                "params": [],
                                "units": [],
                            }
                            for dim in dims
                        ],
                    },
                }

            else:
                err_corr, custom_err_corr = self.set_errcorr_shape(
                    dims,
                    dim_sizes,
                    "err_corr_" + comp + "_" + self.yvariable,
                    str_repeat_corr_dims=str_repeat_corr_dims,
                    repeat_dim_err_corr=repeat_dim_err_corrs[ic],
                )

                template[
                    self.make_ucomp_name(comp, store_unc_percent=store_unc_percent)
                ] = {
                    "dtype": np.float32,
                    "dim": dims,
                    "attributes": {"units": units, "err_corr": err_corr},
                }
                if custom_err_corr is not None:
                    template[
                        "err_corr_" + comp + "_" + self.yvariable
                    ] = custom_err_corr

        return template

    def make_ucomp_name(self, ucomp, store_unc_percent=False, var=None):
        if store_unc_percent:
            uvarname_start = "u_rel_"
        else:
            uvarname_start = "u_"
        if var is None:
            var = self.yvariable
        return uvarname_start + ucomp + "_" + var

    def set_errcorr_shape(
        self,
        dims,
        dim_sizes,
        err_corr_name,
        str_repeat_corr_dims=[],
        str_corr_dims=[],
        repeat_dim_err_corr=[],
    ):
        """
        Function to work out which di
        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :param err_corr_name:
        :type err_corr_name:
        :param str_repeat_corr_dims: set to (list of) string(s) with dimension name(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to [], for which there is no reduction in dimensionality..
        :type str_repeat_corr_dims: str or list of str, optional
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr_list = []
        custom_corr_dims = []
        custom_err_corr_dict = None

        # loop through all dimensions, and copy the ones that are repeat dims

        for dim in dims:
            if dim in str_repeat_corr_dims:
                err_corr_list.append(repeat_dim_err_corr[dim])
            else:
                custom_corr_dims.append(dim)

        if len(custom_corr_dims) > 0:
            if len(custom_corr_dims) == 1:
                corrdim = custom_corr_dims[0]
                custom_corr_dims = custom_corr_dims[0]
            else:
                corrdim = ".".join(custom_corr_dims)
                dim_sizes[corrdim] = 1
                for cust_dim in custom_corr_dims:
                    dim_sizes[corrdim] *= dim_sizes[cust_dim]

            custom_err_corr_dict = {
                "dtype": np.float32,
                "dim": [corrdim, corrdim],
                "attributes": {"units": ""},
            }

        # make a combined custom form for the variables that are not repeated dims
        if len(custom_corr_dims) > 0:
            err_corr_list.append(
                {
                    "dim": custom_corr_dims,
                    "form": "err_corr_matrix",
                    "params": [err_corr_name],
                    "units": [],
                }
            )

        return err_corr_list, custom_err_corr_dict

    def remove_unc_component(self, ds, variable, u_comp, err_corr_comp=None):
        """
        Function to remove an uncertainty component from a dataset

        :param ds: dataset from which uncertainty should be removed
        :type ds: xr.Dataset
        :param variable: variable from which uncertainty should be removed
        :type variable: str
        :param u_comp: name of uncertainty component that should be removed
        :type u_comp: str
        :param err_corr_comp: optionally, the name of the error correlation matrix component associated with this uncertainty component can be provided, so it is removed as well.
        :type err_corr_comp: str
        :return: dataset which uncertainty component removed
        :rtype: xr.Dataset
        """
        ds[variable].attrs["unc_comps"].remove(u_comp)
        try:
            ds = ds.drop(u_comp)
            if err_corr_comp is not None:
                ds = ds.drop(err_corr_comp)
        except:
            pass
        return ds

    def join_with_preexisting_ds(self, ds, ds_pre, drop=None):
        """
        Function to combine digital effects table, with previously populated dataset.
        Only the measurand is overwritten.

        :param ds: digital effects table for measurand, created by punpy
        :type ds: xarray.Dataset
        :param ds_pre: previously populated dataset, to be combined with digital effects table
        :type ds_pre: xarray.Dataset
        :param drop: list of variables to drop
        :param drop: List(str)
        :return: merged digital effects table
        :rtype: xarray.Dataset
        """
        if drop is not None:
            if drop in ds_pre.variables:
                ds_pre = ds_pre.drop(drop)
        for var in ds.variables:
            try:
                ds_pre = ds_pre.drop(var)
            except:
                continue
        ds_out = xr.merge([ds, ds_pre])
        ds_out.attrs = ds_pre.attrs

        return ds_out
