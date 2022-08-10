"""Use Monte Carlo to propagate uncertainties"""

import copy
import time
from abc import ABC,abstractmethod

import numpy as np
import obsarray

from punpy import MCPropagation
from punpy.digital_effects_table.digital_effects_table_templates import (
    DigitalEffectsTableTemplates,)
from punpy.digital_effects_table.measurement_function_utils import\
    MeasurementFunctionUtils

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
        yvariable=None,
        yunit="",
        ydims=None,
        corr_between=None,
        param_fixed=None,
        output_vars=1,
        repeat_dims=-99,
        corr_axis=-99,
        auto_simplify_corr=False,
        refxvar=None,
    ):
        """
        Initialise MeasurementFunction
        """
        if prop is None:
            self.prop = MCPropagation(100)

        else:
            self.prop = prop

        self.xvariables = None
        if self.get_argument_names() is None:
            self.xvariables = xvariables
        else:
            self.xvariables = self.get_argument_names()
            if xvariables is not None:
                if xvariables==self.xvariables:
                    raise ValueError("punpy.MeasurementFunction: when specifying both xvariables and get_argument_names, they need to be the same!")

        if self.xvariables is None:
            raise ValueError("punpy.MeasurementFunction: You need to specify xvariables as keyword when initialising MeasurementFunction object, or add get_argument_names() as a function of the class.")

        if yvariable is None:
            self.yvariable="measurand"
        else:
            self.yvariable = yvariable

        self.templ = DigitalEffectsTableTemplates(self.yvariable, yunit)
        self.ydims = ydims

        self.corr_between = corr_between
        self.output_vars = output_vars

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

        self.param_fixed = param_fixed
        if self.param_fixed is None:
            self.param_fixed = [False] * len(self.xvariables)

        self.corr_axis = corr_axis
        self.auto_simplify_corr = auto_simplify_corr

        self.utils = MeasurementFunctionUtils(self.xvariables,self.str_repeat_dims,self.prop.verbose,self.templ)


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


    def setup(self, *args, **kwargs):
        """
        This function is to provide a setup stage that canbe run before propagating uncertainties.
        This allows to set up additional class attributes etc, without needing to edit the initialiser (which is quite specific to this class).
        This function can optionally be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def propagate_ds(self, *args, store_unc_percent=False, expand=False):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the combined random,
        systematic and structured uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_unc_percent: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        # first calculate the measurand and propagate the uncertainties
        self.check_sizes(*args)
        y = self.run(*args, expand=expand)

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

        u_stru_y, corr_stru_y = self.propagate_structured(*args, expand=expand)

        dim_sizes = {}
        for id, dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]

        repeat_dim_err_corrs = self.utils.find_repeat_dim_corr(
            "str", *args, store_unc_percent=store_unc_percent
        )

        template = self.templ.make_template_main(
            self.ydims,
            dim_sizes,
            store_unc_percent=store_unc_percent,
            str_repeat_dims=self.str_repeat_dims,
            repeat_dim_err_corrs=repeat_dim_err_corrs,
        )
        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y

        if store_unc_percent:
            ucomp_ran = "u_rel_ran_" + self.yvariable
            ucomp_sys = "u_rel_sys_" + self.yvariable
            ucomp_str = "u_rel_str_" + self.yvariable
        else:
            ucomp_ran = "u_ran_" + self.yvariable
            ucomp_sys = "u_sys_" + self.yvariable
            ucomp_str = "u_str_" + self.yvariable

        if u_rand_y is None:
            ds_out = self.templ.remove_unc_component(ds_out, self.yvariable, ucomp_ran)
        else:
            if store_unc_percent:
                ds_out[ucomp_ran].values = u_rand_y / y * 100
            else:
                ds_out[ucomp_ran].values = u_rand_y

        if u_syst_y is None:
            ds_out = self.templ.remove_unc_component(ds_out, self.yvariable, ucomp_sys)
        else:
            if store_unc_percent:
                ds_out[ucomp_sys].values = u_syst_y / y * 100
            else:
                ds_out[ucomp_sys].values = u_syst_y

        if u_stru_y is None:
            ds_out = self.templ.remove_unc_component(
                ds_out,
                self.yvariable,
                ucomp_str,
                err_corr_comp="err_corr_str_" + self.yvariable,
            )
        else:
            if store_unc_percent:
                ds_out[ucomp_str].values = u_stru_y / y * 100
            else:
                ds_out[ucomp_str].values = u_stru_y

            ds_out["err_corr_str_" + self.yvariable].values = corr_stru_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_total(self, *args, store_unc_percent=False, expand=False):
        """
        Function to propagate the total uncertainties present in the digital effects
        tables in the input arguments, through the measurement function to produce
        an output digital effects table with the total uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_unc_percent: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_unc_percent: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        y = self.run(*args, expand=expand)
        u_tot_y, corr_tot_y = self.propagate_total(*args, expand=expand)

        dim_sizes = {}
        for id, dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]

        repeat_dim_err_corrs = self.utils.find_repeat_dim_corr(
            "tot", *args, store_unc_percent=store_unc_percent
        )

        template = self.templ.make_template_tot(
            self.ydims,
            dim_sizes,
            store_unc_percent=store_unc_percent,
            str_repeat_dims=self.str_repeat_dims,
            repeat_dim_err_corrs=repeat_dim_err_corrs,
        )

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y

        if store_unc_percent:
            ucomp = "u_rel_tot_" + self.yvariable
        else:
            ucomp = "u_tot_" + self.yvariable

        if u_tot_y is None:
            ds_out = self.templ.remove_unc_component(
                ds_out,
                self.yvariable,
                ucomp,
                err_corr_comp="err_corr_tot_" + self.yvariable,
            )
        else:
            if store_unc_percent:
                ds_out[ucomp].values = u_tot_y / y * 100
            else:
                ds_out[ucomp].values = u_tot_y
            ds_out["err_corr_tot_" + self.yvariable].values = corr_tot_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_specific(
        self, comp_list, *args, store_unc_percent=False, expand=False
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

        # first calculate the measurand and propagate the uncertainties
        self.check_sizes(*args)
        y = self.run(*args, expand=expand)

        dim_sizes = {}
        for id, dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]

        repeat_dim_err_corrs = [
            self.utils.find_repeat_dim_corr(form, *args, store_unc_percent=store_unc_percent)
            for form in comp_list
        ]

        template = self.templ.make_template_specific(
            comp_list,
            self.ydims,
            dim_sizes,
            store_unc_percent=store_unc_percent,
            str_repeat_dims=self.str_repeat_dims,
            repeat_dim_err_corrs=repeat_dim_err_corrs,
        )

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)
        ds_out[self.yvariable].values = y

        for comp in comp_list:
            err_corr_comp = None
            if comp == "random":
                u_comp_y = self.propagate_random(*args, expand=expand)

            elif comp == "systematic":
                u_comp_y = self.propagate_systematic(*args, expand=expand)

            else:
                u_comp_y, corr_comp_y = self.propagate_specific(
                    comp, *args, return_corr=True, expand=expand
                )
                if corr_comp_y is not None:
                    ds_out[
                        "err_corr_" + comp + "_" + self.yvariable
                    ].values = corr_comp_y
                else:
                    err_corr_comp = "err_corr_" + comp + "_" + self.yvariable

            if u_comp_y is None:
                if store_unc_percent:
                    ds_out = self.templ.remove_unc_component(
                        ds_out,
                        self.yvariable,
                        "u_rel_" + comp + "_" + self.yvariable,
                        err_corr_comp=err_corr_comp,
                    )
                else:
                    ds_out = self.templ.remove_unc_component(
                        ds_out,
                        self.yvariable,
                        "u_" + comp + "_" + self.yvariable,
                        err_corr_comp=err_corr_comp,
                    )
            else:
                if store_unc_percent:
                    ds_out["u_rel_" + comp + "_" + self.yvariable].values = u_comp_y / y * 100
                else:
                    ds_out["u_" + comp + "_" + self.yvariable].values = u_comp_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds_specific (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_all(self, *args, store_unc_percent=False, expand=False):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the combined random,
        systematic and structured uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """

        comp_list = []
        for iv, var in enumerate(self.xvariables):
            for dataset in args:
                if var in dataset.keys():
                    comps = self.utils.find_comps(
                        "tot", dataset, var
                    )
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
            comp_list, *args, store_unc_percent=store_unc_percent, expand=expand
        )

    def run(self, *args, expand=False):
        """

        :param args:
        :type args:
        :param expand:
        :type expand:
        :return:
        :rtype:
        """
        input_qty = self.utils.get_input_qty(args, expand=expand)
        return self.meas_function(*input_qty)

    def check_sizes(self, *args):
        """

        :param args:
        :type args:
        :return:
        :rtype:
        """
        if self.ydims is None:
            for dataset in args:
                try:
                    self.ydims = dataset[self.refxvar].dims
                except:
                    continue

        for i in range(len(self.repeat_dims)):
            if isinstance(self.repeat_dims[i], str):
                if not self.repeat_dims[i] in self.ydims:
                    raise ValueError(
                        "punpy.measurement_function: The repeat_dim (%s) is not in the measurand dimensions (%s)."
                        % (self.repeat_dims[i], self.ydims)
                    )
                self.str_repeat_dims[i] = copy.copy(self.repeat_dims[i])
                self.num_repeat_dims[i] = copy.copy(self.ydims.index(self.repeat_dims[i]))

            elif isinstance(self.repeat_dims[i], (int, np.integer)):
                self.num_repeat_dims[i] = self.repeat_dims[i]
                if self.repeat_dims[i] < 0:
                    self.str_repeat_dims[i] = None
                else:
                    self.str_repeat_dims[i] = copy.copy(self.ydims[self.repeat_dims[i]])
            else:
                raise ValueError(
                    "punpy.measurment_function: repeat_dims needs to be provided as ints or strings"
                )

        self.utils.str_repeat_dims = self.str_repeat_dims

        for iv, var in enumerate(self.xvariables):
            found = False
            for dataset in args:
                if var in dataset.keys():
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

    def propagate_total(self, *args, expand=False):
        self.check_sizes(*args)
        input_qty = self.utils.get_input_qty(args, expand=expand)
        input_unc = self.utils.get_input_unc("tot", args, expand=expand)
        input_corr = self.utils.get_input_corr("tot", args)
        if self.prop.verbose:
            print(
                "inputs extracted (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )
        if all([iu is None for iu in input_unc]):
            return None, None
        else:
            return self.prop.propagate_standard(
                self.meas_function,
                input_qty,
                input_unc,
                input_corr,
                param_fixed=self.param_fixed,
                corr_between=self.corr_between,
                return_corr=True,
                return_samples=False,
                repeat_dims=self.num_repeat_dims,
                corr_axis=self.corr_axis,
                output_vars=self.output_vars,
            )

    def propagate_random(self, *args, expand=False):
        self.check_sizes(*args)
        input_qty = self.utils.get_input_qty(args, expand=expand)
        input_unc = self.utils.get_input_unc("rand", args, expand=expand)
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
                corr_axis=self.corr_axis,
                output_vars=self.output_vars,
            )

    def propagate_systematic(self, *args, expand=False):
        self.check_sizes(*args)
        input_qty = self.utils.get_input_qty(args, expand=expand)
        input_unc = self.utils.get_input_unc("syst", args, expand=expand)
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
                corr_axis=self.corr_axis,
                output_vars=self.output_vars,
            )

    def propagate_structured(self, *args, expand=False):
        self.check_sizes(*args)
        input_qty = self.utils.get_input_qty(args, expand=expand)
        input_unc = self.utils.get_input_unc("stru", args, expand=expand)
        input_corr = self.utils.get_input_corr("stru", args)
        if all([iu is None for iu in input_unc]):
            return None, None
        else:
            return self.prop.propagate_standard(
                self.meas_function,
                input_qty,
                input_unc,
                input_corr,
                param_fixed=self.param_fixed,
                corr_between=self.corr_between,
                return_corr=True,
                return_samples=False,
                repeat_dims=self.num_repeat_dims,
                corr_axis=self.corr_axis,
                output_vars=self.output_vars,
            )

    def propagate_specific(self, form, *args, expand=False, return_corr=False):
        input_qty = self.utils.get_input_qty(args, expand=expand)
        input_unc = self.utils.get_input_unc(form, args, expand=expand)
        input_corr = self.utils.get_input_corr(form, args)
        print(input_qty,input_unc)
        if all([iu is None for iu in input_unc]):
            return None, None
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
                corr_axis=self.corr_axis,
                output_vars=self.output_vars,
            )
