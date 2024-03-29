"""Class to make templates for digital effects tables for measurand"""

import warnings
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
    """
    DigitalEffectsTableTemplates class allows to make templates for digital effects table creation for measurand

    :param yvariable: name of the measurand variable
    :type yvariable: string
    :param yunit: unit of the measurand
    :type yunit: string
    """

    def __init__(self, yvariable, yunit, output_vars):
        if isinstance(yvariable, str):
            self.yvariable = [yvariable]
            self.yunit = [yunit]
        else:
            self.yvariable = yvariable
            self.yunit = yunit

        self.output_vars = output_vars

    def make_template_main(
        self,
        dims,
        dim_sizes,
        str_corr_dims=[],
        separate_corr_dims=False,
        str_repeat_noncorr_dims=[],
        store_unc_percent=False,
        repeat_dim_err_corrs=[],
    ):
        """
        Make the digital effects table template for the case where random, systematic and structured uncertainties are propagated seperately

        :param dims: list of dimensions
        :type dims: list
        :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
        :type u_xvar_ref: xarray.Variable
        :param str_repeat_noncorr_dims: set to (list of) string(s) with dimension name(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to [], for which there is no reduction in dimensionality..
        :type str_repeat_noncorr_dims: str or list of str, optional
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        template = {}

        for i in range(self.output_vars):
            err_corr, custom_err_corr = self._set_errcorr_shape(
                dims[i],
                dim_sizes,
                "err_corr_str_" + self.yvariable[i],
                str_repeat_noncorr_dims=str_repeat_noncorr_dims,
                str_corr_dims=str_corr_dims[i],
                repeat_dim_err_corr=repeat_dim_err_corrs,
            )

            if store_unc_percent:
                units = "%"
            else:
                units = self.yunit[i]

            template[self.yvariable[i]] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {
                    "units": self.yunit[i],
                    "unc_comps": [
                        self._make_ucomp_name(
                            self.yvariable[i],
                            "ran",
                            store_unc_percent=store_unc_percent,
                        ),
                        self._make_ucomp_name(
                            self.yvariable[i],
                            "sys",
                            store_unc_percent=store_unc_percent,
                        ),
                        self._make_ucomp_name(
                            self.yvariable[i],
                            "str",
                            store_unc_percent=store_unc_percent,
                        ),
                    ],
                },
            }
            template[
                self._make_ucomp_name(
                    self.yvariable[i], "ran", store_unc_percent=store_unc_percent
                )
            ] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {
                    "units": units,
                    "err_corr": [
                        {"dim": dim, "form": "random", "params": [], "units": []}
                        for dim in dims[i]
                    ],
                },
            }
            template[
                self._make_ucomp_name(
                    self.yvariable[i], "sys", store_unc_percent=store_unc_percent
                )
            ] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {
                    "units": units,
                    "err_corr": [
                        {"dim": dim, "form": "systematic", "params": [], "units": []}
                        for dim in dims[i]
                    ],
                },
            }
            template[
                self._make_ucomp_name(
                    self.yvariable[i], "str", store_unc_percent=store_unc_percent
                )
            ] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {"units": units, "err_corr": err_corr},
            }

            if custom_err_corr is not None:
                for key in custom_err_corr.keys():
                    if "err_corr_str_" + self.yvariable[i] in key:
                        template[key] = custom_err_corr[key]

        return template

    def make_template_tot(
        self,
        dims,
        dim_sizes,
        str_corr_dims=[],
        separate_corr_dims=False,
        str_repeat_noncorr_dims=[],
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
        template = {}

        for i in range(self.output_vars):
            err_corr, custom_err_corr = self._set_errcorr_shape(
                dims[i],
                dim_sizes,
                "err_corr_tot_" + self.yvariable[i],
                str_repeat_noncorr_dims=str_repeat_noncorr_dims,
                str_corr_dims=str_corr_dims[i],
                repeat_dim_err_corr=repeat_dim_err_corrs,
            )

            if store_unc_percent:
                units = "%"
            else:
                units = self.yunit[i]

            template[self.yvariable[i]] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {
                    "units": self.yunit[i],
                    "unc_comps": [
                        self._make_ucomp_name(
                            self.yvariable[i],
                            "tot",
                            store_unc_percent=store_unc_percent,
                        )
                    ],
                },
            }
            template[
                self._make_ucomp_name(
                    self.yvariable[i], "tot", store_unc_percent=store_unc_percent
                )
            ] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {"units": units, "err_corr": err_corr},
            }

            if custom_err_corr is not None:
                for key in custom_err_corr.keys():
                    if "err_corr_tot_" + self.yvariable[i] in key:
                        template[key] = custom_err_corr[key]

        # if self.output_vars>1:
        #     template["corr_between_vars"]={
        #         "dtype": np.float32,
        #         "dim": dims,
        #         "attributes": {"units": units, "err_corr": err_corr},
        #     }

        return template

    def make_template_specific(
        self,
        comp_list,
        dims,
        dim_sizes,
        str_corr_dims=[],
        separate_corr_dims=False,
        str_repeat_noncorr_dims=[],
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
        template = {}

        for i in range(self.output_vars):
            if store_unc_percent:
                units = "%"
            else:
                units = self.yunit[i]

            template[self.yvariable[i]] = {
                "dtype": np.float32,
                "dim": dims[i],
                "attributes": {
                    "units": self.yunit[i],
                    "unc_comps": [
                        self._make_ucomp_name(
                            self.yvariable[i], comp, store_unc_percent=store_unc_percent
                        )
                        for comp in comp_list
                    ],
                },
            }

            for ic, comp in enumerate(comp_list):
                if comp == "random" and simple_random:
                    template[
                        self._make_ucomp_name(
                            self.yvariable[i], comp, store_unc_percent=store_unc_percent
                        )
                    ] = {
                        "dtype": np.float32,
                        "dim": dims[i],
                        "attributes": {
                            "units": units,
                            "err_corr": [
                                {
                                    "dim": dim,
                                    "form": "random",
                                    "params": [],
                                    "units": [],
                                }
                                for dim in dims[i]
                            ],
                        },
                    }

                elif comp == "systematic" and simple_systematic:
                    template[
                        self._make_ucomp_name(
                            self.yvariable[i], comp, store_unc_percent=store_unc_percent
                        )
                    ] = {
                        "dtype": np.float32,
                        "dim": dims[i],
                        "attributes": {
                            "units": units,
                            "err_corr": [
                                {
                                    "dim": dim,
                                    "form": "systematic",
                                    "params": [],
                                    "units": [],
                                }
                                for dim in dims[i]
                            ],
                        },
                    }

                else:
                    err_corr, custom_err_corr = self._set_errcorr_shape(
                        dims[i],
                        dim_sizes,
                        "err_corr_" + comp + "_" + self.yvariable[i],
                        str_repeat_noncorr_dims=str_repeat_noncorr_dims,
                        str_corr_dims=str_corr_dims[i],
                        repeat_dim_err_corr=repeat_dim_err_corrs[ic],
                    )

                    template[
                        self._make_ucomp_name(
                            self.yvariable[i], comp, store_unc_percent=store_unc_percent
                        )
                    ] = {
                        "dtype": np.float32,
                        "dim": dims[i],
                        "attributes": {"units": units, "err_corr": err_corr},
                    }
                    if custom_err_corr is not None:
                        for key in custom_err_corr.keys():
                            if "err_corr_" + comp + "_" + self.yvariable[i] in key:
                                template[key] = custom_err_corr[key]

        return template

    def _make_ucomp_name(self, var, ucomp, store_unc_percent=False):
        """
        Return the name of the uncertainty component

        :param var: name of variable
        :type var: str
        :param ucomp: name of component
        :type ucomp: str
        :param store_unc_percent: boolean to indicate whether uncertainties are stored in percent, defaults to False.
        :type store_unc_percent: bool, optional
        :return: name of uncertainty component
        :rtype: str
        """
        if store_unc_percent:
            uvarname_start = "u_rel_"
        else:
            uvarname_start = "u_"
        return uvarname_start + ucomp + "_" + var

    def _set_errcorr_shape(
        self,
        dims,
        dim_sizes,
        err_corr_name,
        str_repeat_noncorr_dims=[],
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
        :param str_repeat_noncorr_dims: set to (list of) string(s) with dimension name(s) to select the axis which has repeated measurements. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to [], for which there is no reduction in dimensionality..
        :type str_repeat_noncorr_dims: str or list of str, optional
        :return: measurand digital effects table template to be used by obsarray
        :rtype: dict
        """
        err_corr_list = []
        custom_corr_dims = []
        custom_err_corr_dict = None

        # loop through all dimensions, and copy the ones that are repeat dims
        for dim in dims:
            if dim in str_repeat_noncorr_dims:
                err_corr_list.append(repeat_dim_err_corr[dim])
            else:
                custom_corr_dims.append(dim)

        if len(str_corr_dims) == 0 or all(elem is None for elem in str_corr_dims):
            if len(custom_corr_dims) > 0:
                if len(custom_corr_dims) == 1:
                    corrdim = custom_corr_dims[0]
                    custom_corr_dims = custom_corr_dims[0]
                else:
                    corrdim = ".".join(custom_corr_dims)
                    # dim_sizes[corrdim] = 1
                    # for cust_dim in custom_corr_dims:
                    #     dim_sizes[corrdim] *= dim_sizes[cust_dim]

                custom_err_corr_dict = {
                    err_corr_name: {
                        "dtype": np.float32,
                        "dim": [corrdim, corrdim],
                        "attributes": {"units": ""},
                    }
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
        else:
            custom_err_corr_dict = {}
            for corrdim in str_corr_dims:
                if corrdim is not None:
                    custom_err_corr_dict[err_corr_name + "_" + corrdim] = {
                        "dtype": np.float32,
                        "dim": [corrdim, corrdim],
                        "attributes": {"units": ""},
                    }

                    err_corr_list.append(
                        {
                            "dim": corrdim.split("."),
                            "form": "err_corr_matrix",
                            "params": [err_corr_name + "_" + corrdim],
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
        for var in ds.variables:
            ds[var].encoding = ds_pre[var].encoding
            ds[var].values = ds[var].values.astype(ds_pre[var].values.dtype)
            err_corr_warn = False
            for key in ds_pre[var].attrs.keys():
                if "err_corr" in key:
                    try:
                        if ds[var].attrs[key] == ds_pre[var].attrs[key]:
                            err_corr_warn = True
                    except:
                        err_corr_warn = True

                else:
                    ds[var].attrs[key] = ds_pre[var].attrs[key]
            if err_corr_warn:
                warnings.warn(
                    "The returned dataset has a different error correlation parameterisation for variable %s than the provided ds_out_pre (possibly just the order changed)."
                    % var
                )

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
