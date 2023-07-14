"""Use Monte Carlo to propagate uncertainties"""

import copy
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import obsarray

from punpy.digital_effects_table.digital_effects_table_templates import (
    DigitalEffectsTableTemplates,
)
from punpy.digital_effects_table.measurement_function_utils import (
    MeasurementFunctionUtils,
)
from punpy.mc.mc_propagation import MCPropagation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MeasurementFunction(ABC):
    def __init__(
        self,
        prop=None,
        xvariables=None,
        uncxvariables=None,
        yvariable=None,
        yunit="",
        corr_between=None,
        param_fixed=None,
        repeat_dims=-99,
        corr_dims=-99,
        allow_some_nans=True,
        ydims=None,
        refxvar=None,
        sizes_dict=None,
        use_err_corr_dict=False,
        broadcast_correlation="syst",
    ):
        """
        Initialise MeasurementFunction

        :param prop: punpy MC propagation or LPU propagation object. Defaults to None, in which case a MC propagation object with 100 MC steps is used.
        :type prop: punpy.MCPropagation or punpy. LPUPropagation
        :param xvariables: list of input quantity names, in same order as arguments in measurement function and with same exact names as provided in input datasets. Defaults to None, in which case get_argument_names function is used.
        :type xvariables: list(str), optional
        :param uncxvariables: list of input quantity names for which uncertainties should be propagated. Should be a subset of input quantity names. Defaults to None, in which case uncertainties on all input quantities are used.
        :type uncxvariables: list(str), optional
        :param yvariable: name of measurand. Defaults to None, in which case get_measurand_name_and_unit function is used.
        :type yvariable: str, optional
        :param yunit: unit of measurand. Defaults to "" (unitless).
        :type yunit: str, optional
        :param corr_between: Allows to specify the (average) error correlation coefficient between the various input quantities. Defaults to None, in which case no error-correlation is assumed.
        :type corr_between: numpy.ndarray , optional
        :param param_fixed: set to true or false to indicate for each input quantity whether it has to remain unmodified either when expand=true or when using repeated measurements, defaults to None (no inputs fixed).
        :type param_fixed: list of bools, optional
        :param repeat_dims: Used to select the axis which has repeated measurements. Axis can be specified using the name(s) of the dimension, or their index in the ydims. The calculations will be performed seperately for each of the repeated measurments and then combined, in order to save memory and speed up the process.  Defaults to -99, for which there is no reduction in dimensionality..
        :type repeat_dims: str or int or list(str) or list(int), optional
        :param corr_dims: set to positive integer to select the axis used in the correlation matrix. The correlation matrix will then be averaged over other dimensions. Defaults to -99, for which the input array will be flattened and the full correlation matrix calculated.
        :type corr_dims: integer, optional
        :param ydims: list of dimensions of the measurand, in correct order. list of list of dimensions when there are multiple measurands. Default to None, in which case it is assumed to be the same as refxvar (see below) input quantity.
        :type ydims: list(str), optional
        :param refxvar: name of reference input quantity that has the same shape as measurand. Defaults to None
        :type refxvar: string, optional
        :param sizes_dict: Dictionary with sizes of each of the dimensions of the measurand. Defaults to None, in which cases sizes come from input quantites.
        :type sizes_dict: dict, optional
        :param use_err_corr_dict: when possible, use dictionaries with separate error-correlation info per dimension in order to save memory
        :type use_err_corr_dict: bool, optional
        """
        if prop is None:
            self.prop = MCPropagation(100, dtype="float32")

        else:
            self.prop = prop

        self.xvariables = None
        if self.get_argument_names() is None:
            self.xvariables = xvariables
        else:
            self.xvariables = self.get_argument_names()
            if xvariables is not None:
                if xvariables != self.xvariables:
                    raise ValueError(
                        "punpy.MeasurementFunction: when specifying both xvariables and get_argument_names, they need to be the same!"
                    )

        if self.xvariables is None:
            raise ValueError(
                "punpy.MeasurementFunction: You need to specify xvariables as keyword when initialising MeasurementFunction object, or add get_argument_names() as a function of the class."
            )

        if uncxvariables is None:
            self.uncxvariables = self.xvariables
        else:
            if isinstance(uncxvariables, str):
                uncxvariables = [uncxvariables]

            if np.all([uncvar in self.xvariables for uncvar in uncxvariables]):
                self.uncxvariables = uncxvariables

            else:
                raise ValueError(
                    "punpy.MeasurementFunction: the provided uncxvariables are not a subset of the provided xvariables."
                )

        if yvariable is None:
            self.yvariable, yunit = self.get_measurand_name_and_unit()
        else:
            self.yvariable = yvariable
            yvar = self.get_measurand_name_and_unit()[0]
            if yvar != "measurand":
                if yvariable != yvar:
                    raise ValueError(
                        "punpy.MeasurementFunction: when specifying both yvariable and get_measurand_name_and_unit, they need to be the same!"
                    )

        self.corr_between = corr_between
        if isinstance(self.yvariable, str):
            self.output_vars = 1
        else:
            self.output_vars = len(self.yvariable)
        self.ydims = ydims

        if self.output_vars == 1:
            if isinstance(self.yvariable, str):
                self.yvariable = [self.yvariable]
            if isinstance(yunit, str):
                yunit = [yunit]
            if self.ydims is not None:
                if isinstance(self.ydims[0], str):
                    self.ydims = [ydims]

        self.templ = DigitalEffectsTableTemplates(
            self.yvariable, yunit, self.output_vars
        )
        self.sizes_dict = sizes_dict

        if refxvar is None:
            self.refxvar = self.xvariables[0]
        elif isinstance(refxvar, int):
            self.refxvar = self.xvariables[refxvar]
        else:
            self.refxvar = refxvar

        if isinstance(repeat_dims, int) or isinstance(repeat_dims, str):
            repeat_dims = [repeat_dims]

        self.repeat_dims = np.array(repeat_dims)

        self.num_repeat_dims = np.empty_like(self.repeat_dims, dtype=int)
        self.str_repeat_dims = np.empty_like(self.repeat_dims, dtype="<U30")

        if isinstance(corr_dims, int) or isinstance(corr_dims, str):
            corr_dims = [corr_dims]
        self.corr_dims = np.array(corr_dims, dtype=object)
        self.num_corr_dims = np.empty_like(self.corr_dims, dtype=object)
        self.str_corr_dims = np.empty_like(self.corr_dims, dtype="<U30")

        self.str_repeat_noncorr_dims = []

        self.param_fixed = param_fixed

        self.utils = MeasurementFunctionUtils(
            self.xvariables,
            self.uncxvariables,
            self.ydims,
            self.str_repeat_dims,
            self.str_repeat_noncorr_dims,
            self.prop.verbose,
            self.templ,
            use_err_corr_dict,
            broadcast_correlation,
            param_fixed,
        )

        self.use_err_corr_dict = use_err_corr_dict
        self.allow_some_nans = allow_some_nans

    @abstractmethod
    def meas_function(self, *args, **kwargs):
        """
        meas_function is the measurement function itself, to be used in the uncertainty propagation.
        This function must be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def get_argument_names(self):
        """
        This function allows to return the names of the input quantities as a list of strings.
        Can optionally be overwritten to provide names instead of providing xvariables as a keyword.

        :return: names of the input quantities
        :rtype: list of strings
        """
        return None

    def get_measurand_name_and_unit(self):
        """
        This function allows to return the name and unit of the measurand as strings.
        Can optionally be overwritten to provide names instead of providing yvariable as a keyword.

        :return: name of the measurand, unit
        :rtype: tuple(str, str)
        """
        return "measurand", ""

    def update_measurand(self, measurand, measurand_unit):
        self.yvariable = measurand
        self.templ = DigitalEffectsTableTemplates(self.yvariable, measurand_unit)

    def setup(self, *args, **kwargs):
        """
        This function is to provide a setup stage that can be run before propagating uncertainties.
        This allows to set up additional class attributes etc, without needing to edit the initialiser (which is quite specific to this class).
        This function can optionally be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def propagate_ds(
        self,
        *args,
        store_unc_percent=False,
        expand=False,
        ds_out_pre=None,
        use_ds_out_pre_unmodified=False,
        include_corr=True,
    ):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the combined random,
        systematic and structured uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to False (absolute uncertainties returned)
        :type store_unc_percent: bool (optional)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param ds_out_pre: Pre-existing output dataset in which the measurand & uncertainty variables should be saved. Defaults to None, in which case a new dataset is created.
        :type ds_out_pre: xarray.dataset (optional)
        :param include_corr: boolean to indicate whether the output dataset should include the correlation matrices. Defaults to True.
        :type include_corr: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        # first calculate the measurand and propagate the uncertainties
        y = self.check_sizes_and_run(*args, expand=expand, ds_out_pre=ds_out_pre)

        u_rand_y = self.propagate_random(*args, expand=expand)
        if self.prop.verbose:
            print(
                "propagate_random done (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        u_syst_y = self.propagate_systematic(*args, expand=expand)
        if self.prop.verbose:
            print(
                "propagate systematic done (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        if include_corr:
            if self.output_vars == 1:
                u_stru_y, corr_stru_y = self.propagate_structured(
                    *args, expand=expand, return_corr=include_corr
                )

            else:
                u_stru_y, corr_stru_y, corr_stru_y_between = self.propagate_structured(
                    *args, expand=expand, return_corr=include_corr
                )
        else:
            u_stru_y = self.propagate_structured(
                *args, expand=expand, return_corr=include_corr
            )
            corr_stru_y = None

        if use_ds_out_pre_unmodified:
            if ds_out_pre is not None:
                ds_out = ds_out_pre
            else:
                raise ValueError(
                    "punpy.MeasurementFunction: ds_out_pre needs to be provided when use_ds_out_pre_unmodified is set to True."
                )
        else:
            repeat_dim_err_corrs = self.utils.find_repeat_dim_corr(
                "str", *args, store_unc_percent=store_unc_percent, ydims=self.ydims
            )

            self.utils.set_repeat_dims_form(repeat_dim_err_corrs)
            template = self.templ.make_template_main(
                self.ydims,
                self.sizes_dict,
                store_unc_percent=store_unc_percent,
                str_repeat_noncorr_dims=self.str_repeat_noncorr_dims,
                repeat_dim_err_corrs=repeat_dim_err_corrs,
            )

            # create dataset template
            ds_out = obsarray.create_ds(template, self.sizes_dict)



        # add trivial first dimension to so we can loop over output_vars later
        if self.output_vars == 1:
            if u_rand_y is not None:
                u_rand_y = u_rand_y[None, ...]
            if u_syst_y is not None:
                u_syst_y = u_syst_y[None, ...]
            if u_stru_y is not None:
                u_stru_y = u_stru_y[None, ...]
            if corr_stru_y is not None:
                corr_stru_y = corr_stru_y[None, ...]

        for i in range(self.output_vars):
            ds_out[self.yvariable[i]].values = y[i]

            if store_unc_percent:
                ucomp_ran = "u_rel_ran_" + self.yvariable[i]
                ucomp_sys = "u_rel_sys_" + self.yvariable[i]
                ucomp_str = "u_rel_str_" + self.yvariable[i]
            else:
                ucomp_ran = "u_ran_" + self.yvariable[i]
                ucomp_sys = "u_sys_" + self.yvariable[i]
                ucomp_str = "u_str_" + self.yvariable[i]

            if u_rand_y is None:
                ds_out = self.templ.remove_unc_component(
                    ds_out, self.yvariable[i], ucomp_ran
                )
            else:
                if store_unc_percent:
                    ds_out[ucomp_ran].values = u_rand_y[i] / y[i] * 100
                else:
                    ds_out[ucomp_ran].values = u_rand_y[i]

            if u_syst_y is None:
                ds_out = self.templ.remove_unc_component(
                    ds_out, self.yvariable[i], ucomp_sys
                )
            else:
                if store_unc_percent:
                    ds_out[ucomp_sys].values = u_syst_y[i] / y[i] * 100
                else:
                    ds_out[ucomp_sys].values = u_syst_y[i]

            if u_stru_y is None:
                ds_out = self.templ.remove_unc_component(
                    ds_out,
                    self.yvariable[i],
                    ucomp_str,
                    err_corr_comp="err_corr_str_" + self.yvariable[i],
                )
            else:
                if store_unc_percent:
                    ds_out[ucomp_str].values = u_stru_y[i] / y[i] * 100
                else:
                    ds_out[ucomp_str].values = u_stru_y[i]

                if include_corr:
                    ds_out["err_corr_str_" + self.yvariable[i]].values = corr_stru_y[i]

                else:
                    ds_out.drop("err_corr_str_" + self.yvariable[i])

                    # ds_out.drop("err_corr_str_between")

            if (ds_out_pre is not None) and not use_ds_out_pre_unmodified:
                self.templ.join_with_preexisting_ds(
                    ds_out, ds_out_pre, drop=self.yvariable
                )

        if self.prop.verbose:
            print(
                "finishing propagate_ds (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_total(
        self,
        *args,
        store_unc_percent=False,
        expand=False,
        ds_out_pre=None,
        use_ds_out_pre_unmodified=False,
        include_corr=True,
    ):
        """
        Function to propagate the total uncertainties present in the digital effects
        tables in the input arguments, through the measurement function to produce
        an output digital effects table with the total uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_unc_percent: bool (optional)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param ds_out_pre: Pre-existing output dataset in which the measurand & uncertainty variables should be saved. Defaults to None, in which case a new dataset is created.
        :type ds_out_pre: xarray.dataset (optional)
        :param include_corr: boolean to indicate whether the output dataset should include the correlation matrices. Defaults to True.
        :type include_corr: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )
        y = self.check_sizes_and_run(*args, expand=expand, ds_out_pre=ds_out_pre)

        if include_corr:
            if self.output_vars == 1:
                u_tot_y, corr_tot_y = self.propagate_total(
                    *args, expand=expand, return_corr=include_corr
                )
            else:
                u_tot_y, corr_tot_y, corr_tot_y_between = self.propagate_total(
                    *args, expand=expand, return_corr=include_corr
                )
        else:
            u_tot_y = self.propagate_total(
                *args, expand=expand, return_corr=include_corr
            )
            corr_tot_y = None

        if use_ds_out_pre_unmodified:
            if ds_out_pre is not None:
                ds_out = ds_out_pre
            else:
                raise ValueError(
                    "punpy.MeasurementFunction: ds_out_pre needs to be provided when use_ds_out_pre_unmodified is set to True."
                )
        else:
            repeat_dim_err_corrs = self.utils.find_repeat_dim_corr(
                "tot", *args, store_unc_percent=store_unc_percent, ydims=self.ydims
            )

            self.utils.set_repeat_dims_form(repeat_dim_err_corrs)

            template = self.templ.make_template_tot(
                self.ydims,
                self.sizes_dict,
                store_unc_percent=store_unc_percent,
                str_corr_dims=self.str_corr_dims,
                str_repeat_noncorr_dims=self.str_repeat_noncorr_dims,
                repeat_dim_err_corrs=repeat_dim_err_corrs,
            )

            # create dataset template
            ds_out = obsarray.create_ds(template, self.sizes_dict)

        if self.output_vars == 1:
            u_tot_y = u_tot_y[None, ...]
            corr_tot_y = corr_tot_y[None, ...]

        for i in range(self.output_vars):
            ds_out[self.yvariable[i]].values = y[i]

            if store_unc_percent:
                ucomp = "u_rel_tot_" + self.yvariable[i]
            else:
                ucomp = "u_tot_" + self.yvariable[i]

            if u_tot_y is None:
                ds_out = self.templ.remove_unc_component(
                    ds_out,
                    self.yvariable[i],
                    ucomp,
                    err_corr_comp="err_corr_tot_" + self.yvariable[i],
                )
            else:
                if store_unc_percent:
                    ds_out[ucomp].values = u_tot_y[i] / y[i] * 100
                else:
                    ds_out[ucomp].values = u_tot_y[i]

                if include_corr:
                    if len(self.str_corr_dims) <= 1:
                        ds_out["err_corr_tot_" + self.yvariable[i]].values = corr_tot_y[
                            i
                        ]
                    else:
                        for ii in range(len(self.str_corr_dims)):
                            ds_out[
                                "err_corr_tot_"
                                + self.yvariable[i]
                                + "_"
                                + self.str_corr_dims[ii]
                            ].values = corr_tot_y[i][ii]

                else:
                    if len(self.str_corr_dims) == 1:
                        ds_out.drop("err_corr_tot_" + self.yvariable[i])
                    else:
                        for ii in range(len(self.str_corr_dims)):
                            ds_out.drop(
                                "err_corr_tot_"
                                + self.yvariable[i]
                                + "_"
                                + self.str_corr_dims[ii]
                            )

            if (ds_out_pre is not None) and not use_ds_out_pre_unmodified:
                self.templ.join_with_preexisting_ds(
                    ds_out, ds_out_pre, drop=self.yvariable[i]
                )

        # ds_out["err_corr_tot_between"]= corr_tot_y_between
        # ds_out.drop("err_corr_tot_between")

        if self.prop.verbose:
            print(
                "finishing propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_specific(
        self,
        comp_list,
        *args,
        comp_list_out=None,
        store_unc_percent=False,
        expand=False,
        ds_out_pre=None,
        use_ds_out_pre_unmodified=False,
        include_corr=True,
        simple_random=True,
        simple_systematic=True,
    ):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the uncertainties of specific
        components listed in comp_list.

        :param comp_list: list of uncertainty contributions to propagate
        :rtype comp_list: list of strings or string
        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_unc_percent: bool (optional)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param ds_out_pre: Pre-existing output dataset in which the measurand & uncertainty variables should be saved. Defaults to None, in which case a new dataset is created.
        :type ds_out_pre: xarray.dataset (optional)
        :param include_corr: boolean to indicate whether the output dataset should include the correlation matrices. Defaults to True.
        :type include_corr: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds_specific (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        if isinstance(comp_list, str):
            comp_list = [comp_list]

        if comp_list_out is None:
            comp_list_out = comp_list

        # first calculate the measurand and propagate the uncertainties
        y = self.check_sizes_and_run(*args, expand=expand, ds_out_pre=ds_out_pre)

        if use_ds_out_pre_unmodified:
            if ds_out_pre is not None:
                ds_out = ds_out_pre
            else:
                raise ValueError(
                    "punpy.MeasurementFunction: ds_out_pre needs to be provided when use_ds_out_pre_unmodified is set to True."
                )
        else:
            repeat_dim_err_corrs = [
                self.utils.find_repeat_dim_corr(
                    form, *args, store_unc_percent=store_unc_percent, ydims=self.ydims
                )
                for form in comp_list
            ]

            self.utils.set_repeat_dims_form(repeat_dim_err_corrs)

            template = self.templ.make_template_specific(
                comp_list_out,
                self.ydims,
                self.sizes_dict,
                store_unc_percent=store_unc_percent,
                str_repeat_noncorr_dims=self.str_repeat_noncorr_dims,
                repeat_dim_err_corrs=repeat_dim_err_corrs,
                simple_random=simple_random,
                simple_systematic=simple_systematic,
            )

            # create dataset template
            ds_out = obsarray.create_ds(template, self.sizes_dict)

        for i in range(self.output_vars):
            ds_out[self.yvariable[i]].values = y[i]

        for icomp, comp in enumerate(comp_list):
            corr_comp_y = None
            if comp == "random":
                u_comp_y = self.propagate_random(*args, expand=expand)
            elif comp == "systematic":
                u_comp_y = self.propagate_systematic(*args, expand=expand)

            else:
                if include_corr:
                    if self.output_vars == 1:
                        u_comp_y, corr_comp_y = self.propagate_specific(
                            comp, *args, return_corr=include_corr, expand=expand
                        )
                        corr_comp_y = corr_comp_y[None, ...]

                    else:
                        (
                            u_comp_y,
                            corr_comp_y,
                            corr_comp_y_between,
                        ) = self.propagate_specific(
                            comp, *args, return_corr=include_corr, expand=expand
                        )
                else:
                    u_comp_y = self.propagate_specific(
                        comp, *args, return_corr=include_corr, expand=expand
                    )

            if self.output_vars == 1:
                u_comp_y = u_comp_y[None, ...]

            for i in range(self.output_vars):
                if corr_comp_y is not None:
                    try:
                        ds_out[
                            "err_corr_" + comp_list_out[icomp] + "_" + self.yvariable[i]
                        ].values = corr_comp_y[i]
                    except:
                        warnings.warn(
                            "not able to set %s in the output dataset"
                            % (
                                "err_corr_"
                                + comp_list_out[icomp]
                                + "_"
                                + self.yvariable[i]
                            )
                        )
                else:
                    try:
                        ds_out.drop(
                            "err_corr_" + comp_list_out[icomp] + "_" + self.yvariable[i]
                        )
                    except:
                        pass

                if u_comp_y is None:
                    if store_unc_percent:
                        ds_out = self.templ.remove_unc_component(
                            ds_out,
                            self.yvariable[i],
                            "u_rel_" + comp_list_out[icomp] + "_" + self.yvariable[i],
                        )
                    else:
                        ds_out = self.templ.remove_unc_component(
                            ds_out,
                            self.yvariable[i],
                            "u_" + comp_list_out[icomp] + "_" + self.yvariable[i],
                        )
                else:
                    if store_unc_percent:
                        ds_out[
                            "u_rel_" + comp_list_out[icomp] + "_" + self.yvariable[i]
                        ].values = (u_comp_y[i] / y[i] * 100)
                    else:
                        ds_out[
                            "u_" + comp_list_out[icomp] + "_" + self.yvariable[i]
                        ].values = u_comp_y[i]

        for i in range(self.output_vars):
            if (ds_out_pre is not None) and not use_ds_out_pre_unmodified:
                ds_out = self.templ.join_with_preexisting_ds(
                    ds_out, ds_out_pre, drop=self.yvariable[i]
                )

        if self.prop.verbose:
            print(
                "finishing propagate_ds_specific (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_all(
        self,
        *args,
        store_unc_percent=False,
        expand=False,
        ds_out_pre=None,
        use_ds_out_pre_unmodified=False,
        include_corr=True,
    ):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the combined random,
        systematic and structured uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_unc_percent: bool (optional)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param ds_out_pre: Pre-existing output dataset in which the measurand & uncertainty variables should be saved. Defaults to None, in which case a new dataset is created.
        :type ds_out_pre: xarray.dataset (optional)
        :param include_corr: boolean to indicate whether the output dataset should include the correlation matrices. Defaults to True.
        :type include_corr: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """

        comp_list = []
        for iv, var in enumerate(self.xvariables):
            for dataset in args:
                if var in dataset.keys():
                    comps = self.utils.find_comps("tot", dataset, var)
                    if comps is not None:
                        for comp in comps:
                            if isinstance(comp, str):
                                comp_name = comp
                            else:
                                comp_name = comp.name
                            comp_name = comp_name.replace("_" + var, "")
                            comp_name = comp_name.replace("u_rel_", "")
                            if comp_name[0:2] == "u_":
                                comp_name = comp_name[2::]
                            comp_list.append(comp_name)
        comp_list = np.unique(np.array(comp_list))
        return self.propagate_ds_specific(
            comp_list,
            *args,
            store_unc_percent=store_unc_percent,
            expand=expand,
            ds_out_pre=ds_out_pre,
            use_ds_out_pre_unmodified=use_ds_out_pre_unmodified,
            include_corr=include_corr,
        )

    def run(self, *args, expand=False):
        """
        Function to calculate the measurand by running input quantities through measurement function.

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :return: measurand
        :rtype: numpy.ndarray
        """
        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )

        try:
            return np.array(self.meas_function(*input_qty))
        except:
            return np.array(self.meas_function(*input_qty), dtype=object)

    def check_sizes_and_run(self, *args, expand=False, ds_out_pre=None):
        """
        Function to check the sizes of the input quantities and measurand and perform some checks and preprocessing

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :return: None
        :rtype: None
        """
        if self.ydims is None:
            self.ydims = np.empty(self.output_vars, dtype=object)
            for i in range(self.output_vars):
                if ds_out_pre is not None:
                    self.ydims[i] = ds_out_pre[self.yvariable[i]].dims
                else:
                    for dataset in args:
                        try:
                            self.ydims[i] = dataset[self.refxvar].dims
                        except:
                            continue

        if self.sizes_dict is None:
            self.sizes_dict = {}
            for i in range(self.output_vars):
                if ds_out_pre is not None:
                    for idim, dim in enumerate(self.ydims[i]):
                        self.sizes_dict[dim] = ds_out_pre[
                            self.yvariable[i]
                        ].values.shape[idim]
                else:
                    for dataset in args:
                        try:
                            for idim, dim in enumerate(self.ydims[i]):
                                self.sizes_dict[dim] = dataset[
                                    self.refxvar
                                ].values.shape[idim]
                        except:
                            continue

        y = self.run(*args, expand=expand)

        if self.output_vars == 1:
            y = y[None, ...]

        if self.sizes_dict is None:
            self.sizes_dict = {}
            for i in range(self.output_vars):
                for idim, dim in enumerate(self.ydims[i]):
                    try:
                        self.sizes_dict[dim] = y[i].shape[idim]
                    except:
                        raise ValueError()

        for idimr in range(len(self.repeat_dims)):
            if isinstance(self.repeat_dims[idimr], str):
                (
                    self.str_repeat_dims[idimr],
                    self.num_repeat_dims[idimr],
                ) = self.check_and_convert_str_dims(self.repeat_dims[idimr])

            elif isinstance(self.repeat_dims[idimr], (int, np.integer)):
                if self.repeat_dims[idimr] >= 0:
                    (
                        self.str_repeat_dims[idimr],
                        self.num_repeat_dims[idimr],
                    ) = self.check_and_convert_num_dims(self.repeat_dims[idimr])
                else:
                    self.num_repeat_dims[idimr] = self.repeat_dims[idimr]

            else:
                raise ValueError(
                    "punpy.measurment_function: repeat_dims needs to be provided as ints or strings"
                )

        str_repeat_noncorr_dims = [
            str_dim for str_dim in self.str_repeat_dims if str_dim != ""
        ]

        all_corr_dims = []
        for idimc in range(len(self.corr_dims)):
            if isinstance(self.corr_dims[idimc], str):
                if "." in self.corr_dims[idimc]:
                    corr_dims = self.corr_dims[idimc].split(".")
                    str_corr_dims = np.empty_like(corr_dims)
                    num_corr_dims = np.empty_like(corr_dims, dtype=int)
                    if corr_dims[0].isdigit():
                        corrlen = 1
                        for ic in range(len(corr_dims)):
                            (
                                str_corr_dims[ic],
                                num_corr_dims[ic],
                            ) = self.check_and_convert_num_dims(int(corr_dims[ic]))
                            all_corr_dims.append(str_corr_dims[ic])
                            corrlen *= y[0].shape[num_corr_dims[ic]]
                        self.num_corr_dims[idimc] = copy.copy(self.corr_dims[idimc])
                        self.str_corr_dims[idimc] = ".".join(str_corr_dims)
                        self.sizes_dict[self.str_corr_dims[idimc]] = corrlen

                    else:
                        corrlen = 1
                        for ic in range(len(corr_dims)):
                            (
                                str_corr_dims[ic],
                                num_corr_dims[ic],
                            ) = self.check_and_convert_str_dims(corr_dims[ic])
                            all_corr_dims.append(str_corr_dims[ic])
                            corrlen *= y[0].shape[num_corr_dims[ic]]
                        self.str_corr_dims[idimc] = copy.copy(self.corr_dims[idimc])
                        self.num_corr_dims[idimc] = ".".join(
                            [str(cdim) for cdim in num_corr_dims]
                        )
                        self.sizes_dict[self.str_corr_dims[idimc]] = corrlen

                elif not self.corr_dims[idimc] in self.ydims[0]:
                    raise ValueError(
                        "punpy.measurement_function: The corr_dim (%s) is not in the measurand dimensions (%s)."
                        % (self.corr_dims[idimc], self.ydims)
                    )

                else:
                    (
                        self.str_corr_dims[idimc],
                        self.num_corr_dims[idimc],
                    ) = self.check_and_convert_str_dims(self.corr_dims[idimc])
                    all_corr_dims.append(self.corr_dims[idimc])

            elif isinstance(self.corr_dims[idimc], (int, np.integer)):
                if self.corr_dims[idimc] >= 0:
                    (
                        self.str_corr_dims[idimc],
                        self.num_corr_dims[idimc],
                    ) = self.check_and_convert_num_dims(self.corr_dims[idimc])
                    all_corr_dims.append(self.ydims[0][self.corr_dims[idimc]])
                else:
                    self.num_corr_dims[idimc] = self.corr_dims[idimc]
                    self.str_corr_dims = []
                    all_corr_dims = self.ydims[0]
            else:
                raise ValueError(
                    "punpy.measurment_function: corr_dims needs to be provided as ints or strings"
                )

        for i in range(self.output_vars):
            for idim, dim in enumerate(self.ydims[i]):
                if dim not in all_corr_dims:
                    str_repeat_noncorr_dims.append(dim)

        self.str_repeat_noncorr_dims = np.array(str_repeat_noncorr_dims).flatten()

        for i in range(self.output_vars):
            # set sizes dict for combined shape of error correlation dict
            key = ""
            val = 1
            for idim, dim in enumerate(self.ydims[i]):
                if dim not in self.str_repeat_noncorr_dims:
                    key += "." + dim
                    val *= self.sizes_dict[dim]
            self.sizes_dict[key[1::]] = val

        if (not expand) and (self.param_fixed is None):
            self.param_fixed = [False] * len(self.xvariables)
            for iv, var in enumerate(self.xvariables):
                found = False
                for dataset in args:
                    if hasattr(dataset, "variables"):
                        if var in dataset.variables:
                            if all(
                                [
                                    self.str_repeat_dims[i] in dataset[var].dims
                                    for i in range(len(self.str_repeat_dims))
                                ]
                            ):
                                found = True

                if not found:
                    self.param_fixed[iv] = True
                    if self.prop.verbose:
                        print(
                            "Variable %s not found in repeat_dims. setting param_fixed to True"
                            % (var)
                        )

        self.utils.ydims = self.ydims
        self.utils.str_repeat_noncorr_dims = self.str_repeat_noncorr_dims
        self.utils.str_repeat_dims = self.str_repeat_dims

        return y

    def check_and_convert_str_dims(self, dim):
        for i in range(self.output_vars):
            if not dim in self.ydims[i]:
                raise ValueError(
                    "punpy.measurement_function: The repeat_dim (%s) is not in the measurand dimensions (%s)."
                    % (dim, self.ydims)
                )
        if not np.all(
            [ydims.index(dim) == self.ydims[0].index(dim) for ydims in self.ydims]
        ):
            warnings.warn(
                "punpy.measurement_function: The repeat_dim (%s) cannot be used because it is not at the same index for every ydims of every measurand (%s)."
                % (dim, self.ydims)
            )
        return copy.copy(dim), copy.copy(self.ydims[0].index(dim))

    def check_and_convert_num_dims(self, dim):
        if dim >= 0:
            if not np.all([ydims[dim] == self.ydims[0][dim] for ydims in self.ydims]):
                warnings.warn(
                    "punpy.measurement_function: The repeat_dim (%s) cannot be used because it is not at the same index for every ydims of every measurand (%s)."
                    % (dim, self.ydims)
                )

        return copy.copy(self.ydims[0][dim]), dim

    def propagate_total(self, *args, expand=False, return_corr=True):
        """
        Function to propagate uncertainties for the total uncertainty component.

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param return_corr:  boolean to indicate whether the measurand error-correlation matrices should be returned. Defaults to True.
        :type return_corr: bool (optional)
        :return: uncertainty on measurand for total uncertainty component, error-correlation matrix of measurand for total uncertainty component
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        y = self.check_sizes_and_run(*args, expand=expand)
        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        input_unc = self.utils.get_input_unc(
            "tot",
            args,
            expand=expand,
            sizes_dict=self.sizes_dict,
            ydims=self.ydims,
            corr_dims=self.str_corr_dims,
        )
        input_corr = self.utils.get_input_corr(
            "tot", args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        if self.prop.verbose:
            print(
                "inputs extracted (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )
        if all([iu is None for iu in input_unc]):
            return None, None
        else:
            if self.use_err_corr_dict:
                MC_x = self.prop.generate_MC_sample(
                    input_qty,
                    input_unc,
                    corr_x=input_corr,
                    corr_between=self.corr_between,
                    comp_list=True,
                )
                MC_y = self.prop.run_samples(
                    self.meas_function, MC_x, output_vars=self.output_vars
                )
                return self.prop.process_samples(
                    MC_x,
                    MC_y,
                    return_corr=return_corr,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                )
            else:
                return self.prop.propagate_standard(
                    self.meas_function,
                    input_qty,
                    input_unc,
                    input_corr,
                    param_fixed=self.param_fixed,
                    corr_between=self.corr_between,
                    return_corr=return_corr,
                    return_samples=False,
                    repeat_dims=self.num_repeat_dims,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                )

    def propagate_random(self, *args, expand=False):
        """
        Function to propagate uncertainties for the random uncertainty component.

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :return: uncertainty on measurand for random uncertainty component
        :rtype: numpy.ndarray
        """
        y = self.check_sizes_and_run(*args, expand=expand)
        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        input_unc = self.utils.get_input_unc( # this should always be absolute
            "rand",
            args,
            expand=expand,
            sizes_dict=self.sizes_dict,
            ydims=self.ydims,
            corr_dims=self.str_corr_dims,
        )

        if all([iu is None for iu in input_unc]):
            return None

        else:
            return self.prop.propagate_random(
                self.meas_function,
                input_qty,
                input_unc,
                param_fixed=self.param_fixed,
                corr_between=self.corr_between,
                return_corr=False,
                return_samples=False,
                repeat_dims=self.num_repeat_dims,
                corr_dims=self.num_corr_dims,
                output_vars=self.output_vars,
            )

    def propagate_systematic(self, *args, expand=False):
        """
        Function to propagate uncertainties for the systemtic uncertainty component.

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :return: uncertainty on measurand for systematic uncertainty component
        :rtype: numpy.ndarray
        """
        y = self.check_sizes_and_run(*args, expand=expand)
        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        input_unc = self.utils.get_input_unc(
            "syst",
            args,
            expand=expand,
            sizes_dict=self.sizes_dict,
            ydims=self.ydims,
            corr_dims=self.str_corr_dims,
        )
        if all([iu is None for iu in input_unc]):
            return None
        else:
            return self.prop.propagate_systematic(
                self.meas_function,
                input_qty,
                input_unc,
                param_fixed=self.param_fixed,
                corr_between=self.corr_between,
                return_corr=False,
                return_samples=False,
                repeat_dims=self.num_repeat_dims,
                corr_dims=self.num_corr_dims,
                output_vars=self.output_vars,
            )

    def propagate_structured(self, *args, expand=False, return_corr=True):
        """
        Function to propagate uncertainties for the structured uncertainty component.

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param return_corr:  boolean to indicate whether the measurand error-correlation matrices should be returned. Defaults to True.
        :type return_corr: bool (optional)
        :return: uncertainty on measurand for structured uncertainty component, error-correlation matrix of measurand for structured uncertainty component
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        y = self.check_sizes_and_run(*args, expand=expand)
        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        input_unc = self.utils.get_input_unc(
            "stru",
            args,
            expand=expand,
            sizes_dict=self.sizes_dict,
            ydims=self.ydims,
            corr_dims=self.str_corr_dims,
        )
        input_corr = self.utils.get_input_corr(
            "stru", args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )

        if all([iu is None for iu in input_unc]):
            return None, None
        else:
            if self.use_err_corr_dict:
                MC_x = self.prop.generate_MC_sample(
                    input_qty,
                    input_unc,
                    corr_x=input_corr,
                    corr_between=self.corr_between,
                    comp_list=True,
                )
                MC_y = self.prop.run_samples(
                    self.meas_function, MC_x, output_vars=self.output_vars
                )
                return self.prop.process_samples(
                    MC_x,
                    MC_y,
                    return_corr=return_corr,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                )
            else:
                return self.prop.propagate_standard(
                    self.meas_function,
                    input_qty,
                    input_unc,
                    input_corr,
                    param_fixed=self.param_fixed,
                    corr_between=self.corr_between,
                    return_corr=return_corr,
                    return_samples=False,
                    repeat_dims=self.num_repeat_dims,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                )

    def propagate_specific(self, form, *args, expand=False, return_corr=False):
        """
        Function to propagate uncertainties for a specific uncertainty component.

        :param form: name or type of uncertainty component
        :type form: str
        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param expand: boolean to indicate whether the input quantities should be expanded/broadcasted to the shape of the measurand. Defaults to False.
        :type expand: bool (optional)
        :param return_corr:  boolean to indicate whether the measurand error-correlation matrices should be returned. Defaults to True.
        :type return_corr: bool (optional)
        :return: uncertainty on measurand for specific uncertainty component, error-correlation matrix of measurand for specific uncertainty component
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        y = self.check_sizes_and_run(*args, expand=expand)

        input_qty = self.utils.get_input_qty(
            args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        input_unc = self.utils.get_input_unc(
            form,
            args,
            expand=expand,
            sizes_dict=self.sizes_dict,
            ydims=self.ydims,
            corr_dims=self.str_corr_dims,
        )
        input_corr = self.utils.get_input_corr(
            form, args, expand=expand, sizes_dict=self.sizes_dict, ydims=self.ydims
        )
        if all([iu is None for iu in input_unc]):
            return None, None
        else:
            if self.use_err_corr_dict:
                MC_x = self.prop.generate_MC_sample(
                    input_qty,
                    input_unc,
                    corr_x=input_corr,
                    corr_between=self.corr_between,
                )
                MC_y = self.prop.run_samples(
                    self.meas_function, MC_x, output_vars=self.output_vars
                )

                return self.prop.process_samples(
                    MC_x,
                    MC_y,
                    return_corr=return_corr,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                )
            else:
                return self.prop.propagate_standard(
                    self.meas_function,
                    input_qty,
                    input_unc,
                    input_corr,
                    param_fixed=self.param_fixed,
                    corr_between=self.corr_between,
                    return_corr=return_corr,
                    return_samples=False,
                    repeat_dims=self.num_repeat_dims,
                    corr_dims=self.num_corr_dims,
                    output_vars=self.output_vars,
                    allow_some_nans=self.allow_some_nans,
                )
