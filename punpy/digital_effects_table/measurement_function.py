"""Use Monte Carlo to propagate uncertainties"""

import time
from abc import ABC,abstractmethod

import numpy as np
import obsarray
import xarray as xr
from punpy.digital_effects_table.digital_effects_table_templates import (
    DigitalEffectsTableTemplates,)

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MeasurementFunction(ABC):
    def __init__(
        self,
        prop,
        xvariables,
        yvariable,
        yunit,
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
        self.prop = prop

        self.xvariables = xvariables
        self.yvariable = yvariable
        self.templ = DigitalEffectsTableTemplates(
            self.yvariable, yunit
        )
        self.ydims = ydims

        self.corr_between = corr_between
        self.output_vars = output_vars

        if refxvar is None:
            self.refxvar = xvariables[0]
        elif isinstance(refxvar, int):
            self.refxvar = xvariables[refxvar]
        else:
            self.refxvar = refxvar

        if isinstance(repeat_dims, int) or isinstance(repeat_dims, str):
            repeat_dims = [repeat_dims]
        self.repeat_dims = np.array(repeat_dims)
        self.num_repeat_dims = np.empty_like(self.repeat_dims, dtype=int)
        self.str_repeat_dims = np.empty_like(self.repeat_dims, dtype='<U30')

        self.param_fixed = param_fixed
        if self.param_fixed is None:
            self.param_fixed = [False] * len(self.xvariables)

        self.corr_axis = corr_axis
        self.auto_simplify_corr = auto_simplify_corr

    @abstractmethod
    def meas_function(self,*args, **kwargs):
        """
        meas_function is the measurement function itself, to be used in the uncertainty propagation.
        This function must be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def setup(self,*args, **kwargs):
        """
        meas_function is the measurement function itself, to be used in the uncertainty propagation.
        This function must be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def propagate_ds(self, *args, store_rel_unc=False):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the combined random,
        systematic and structured uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_rel_unc: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_rel_unc: bool (optional)
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
        y = self.run(*args)

        u_rand_y = self.propagate_random(*args)
        if self.prop.verbose:
            print(
                "propagate_random done (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        u_syst_y = self.propagate_systematic(*args)
        if self.prop.verbose:
            print(
                "propagate systematic done (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        u_stru_y, corr_stru_y = self.propagate_structured(*args)

        dim_sizes = {}
        for id,dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]

        repeat_dim_err_corrs=self.find_repeat_dim_corr("str",*args,store_rel_unc=store_rel_unc)

        template = self.templ.make_template_main(
            self.ydims, dim_sizes, store_rel_unc=store_rel_unc, str_repeat_dims=self.str_repeat_dims,repeat_dim_err_corrs=repeat_dim_err_corrs
        )
        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y

        if store_rel_unc:
            ds_out["u_rel_ran_" + self.yvariable].values = u_rand_y / y
            ds_out["u_rel_sys_" + self.yvariable].values = u_syst_y / y
            ds_out["u_rel_str_" + self.yvariable].values = u_stru_y / y

            ds_out["u_rel_ran_" + self.yvariable].attrs["units"] = "%"
            ds_out["u_rel_sys_" + self.yvariable].attrs["units"] = "%"
            ds_out["u_rel_str_" + self.yvariable].attrs["units"] = "%"
        else:
            ds_out["u_ran_" + self.yvariable].values = u_rand_y
            ds_out["u_sys_" + self.yvariable].values = u_syst_y
            ds_out["u_str_" + self.yvariable].values = u_stru_y

        ds_out["err_corr_str_" + self.yvariable].values = corr_stru_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_total(self, *args, store_rel_unc=False):
        """
        Function to propagate the total uncertainties present in the digital effects
        tables in the input arguments, through the measurement function to produce
        an output digital effects table with the total uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_rel_unc: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_rel_unc: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        y = self.run(*args)
        u_tot_y, corr_tot_y = self.propagate_total(*args)

        dim_sizes = {}
        for id,dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]


        repeat_dim_err_corrs=self.find_repeat_dim_corr("tot",*args,store_rel_unc=store_rel_unc)

        template = self.templ.make_template_tot(
            self.ydims, dim_sizes, store_rel_unc=store_rel_unc, str_repeat_dims=self.str_repeat_dims,repeat_dim_err_corrs=repeat_dim_err_corrs
        )

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y

        if store_rel_unc:
            ds_out["u_rel_tot_" + self.yvariable].values = u_tot_y / y
            ds_out["u_rel_tot_" + self.yvariable].attrs["units"] = "%"
        else:
            ds_out["u_tot_" + self.yvariable].values = u_tot_y

        ds_out["err_corr_tot_" + self.yvariable].values = corr_tot_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds_total (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out

    def propagate_ds_specific(self, comp_list, *args, store_rel_unc=False):
        """
        Function to propagate the uncertainties on the input quantities present in the
        digital effects tables provided as the input arguments, through the measurement
        function to produce an output digital effects table with the uncertainties of specific
        components listed in comp_list.

        :param comp_list: list of uncertainty contributions to propagate
        :rtype comp_list: list of strings or string
        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :param store_rel_unc: Boolean defining whether relative uncertainties should be returned or not. Default to True (relative uncertaintie returned)
        :type store_rel_unc: bool (optional)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print(
                "starting propagate_ds_specific (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        if isinstance(comp_list,str):
            comp_list=[comp_list]

        # first calculate the measurand and propagate the uncertainties
        self.check_sizes(*args)
        y = self.run(*args)

        dim_sizes = {}
        for id,dim in enumerate(self.ydims):
            dim_sizes[dim] = y.shape[id]

        repeat_dim_err_corrs=[self.find_repeat_dim_corr(form,*args,store_rel_unc=store_rel_unc) for form in comp_list]

        template = self.templ.make_template_specific(
            comp_list, self.ydims, dim_sizes, store_rel_unc=store_rel_unc, str_repeat_dims=self.str_repeat_dims,repeat_dim_err_corrs=repeat_dim_err_corrs
        )

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)
        ds_out[self.yvariable].values = y

        for comp in comp_list:
            if comp == "random":
                u_comp_y = self.propagate_random(*args)

            elif comp == "systematic":
                u_comp_y = self.propagate_systematic(*args)

            else:
                u_comp_y, corr_comp_y = self.propagate_specific(
                    comp, *args, return_corr=True
                )
                ds_out["err_corr_" + comp + "_" + self.yvariable].values = corr_comp_y

            if store_rel_unc:
                ds_out["u_rel_" + comp + "_" + self.yvariable].values = u_comp_y / y
                ds_out["u_rel_" + comp + "_" + self.yvariable].attrs["units"] = "%"

            else:
                ds_out["u_" + comp + "_" + self.yvariable].values = u_comp_y

        if self.prop.verbose:
            print(
                "finishing propagate_ds_specific (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

        return ds_out


    def propagate_ds_all(self, *args, store_rel_unc=False):
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

        comp_list=[]
        for iv, var in enumerate(self.xvariables):
            for dataset in args:
                if var in dataset.keys():
                    comps=self.find_comps("tot",dataset,var,store_rel_unc=store_rel_unc)
                    if comps is not None:
                        for comp in comps:
                            if isinstance(comp,str):
                                comp_name=comp
                            else:
                                comp_name=comp.name
                            comp_name=comp_name.replace("_"+var,"")
                            comp_name=comp_name.replace("u_rel_","")
                            if comp_name[0:2]=="u_":
                                comp_name=comp_name[2::]
                            comp_list.append(comp_name)
        comp_list=np.unique(np.array(comp_list))
        return self.propagate_ds_specific(comp_list, *args, store_rel_unc=store_rel_unc)

    def run(self, *args, expand=True):
        """

        :param args:
        :type args:
        :param expand:
        :type expand:
        :return:
        :rtype:
        """
        input_qty = self.get_input_qty(args, expand=expand)
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
                    self.ydims= dataset[self.refxvar].dims
                except:
                    continue

        for i in range(len(self.repeat_dims)):
            if isinstance(self.repeat_dims[i], str):
                if not self.repeat_dims[i] in self.ydims:
                    raise ValueError("punpy.measurement_function: The repeat_dim (%s) is not in the measurand dimensions (%s)."%(self.repeat_dims[i],self.ydims))
                self.str_repeat_dims[i] = self.repeat_dims[i]
                self.num_repeat_dims[i] = self.ydims.index(self.repeat_dims[i])

            elif isinstance(self.repeat_dims[i], (int, np.integer)):
                self.num_repeat_dims[i] = self.repeat_dims[i]
                if self.repeat_dims[i] < 0:
                    self.str_repeat_dims[i] = None
                else:
                    self.str_repeat_dims[i] = self.ydims[self.repeat_dims[i]]
            else:
                raise ValueError(
                    "punpy.measurment_function: repeat_dims needs to be provided as ints or strings"
                )

        for iv, var in enumerate(self.xvariables):
            found = False
            for dataset in args:
                if var in dataset.keys():
                    if all([self.str_repeat_dims[i] in dataset[var].dims for i in range(len(self.str_repeat_dims))]):
                        found = True

            if not found:
                self.param_fixed[iv] = True
                if self.prop.verbose:
                    print(
                        "Variable %s not found in repeat_dims. setting param_fixed to True"
                        % (var)
                    )


    def find_repeat_dim_corr(self,form,*args,store_rel_unc=False):
        """

        :param form:
        :type form:
        :param args:
        :type args:
        :param store_rel_unc:
        :type store_rel_unc:
        :return:
        :rtype:
        """
        repeat_dims_errcorrs={}
        for repeat_dim in self.str_repeat_dims:
            repeat_dims_errcorrs[repeat_dim]={"dim": repeat_dim, "form": None, "params": [], "units": []}
            for iv, var in enumerate(self.xvariables):
                for dataset in args:
                    if var in dataset.keys():
                        comps=self.find_comps(form,dataset,var,store_rel_unc=store_rel_unc)
                        if comps is None:
                            continue
                        elif (repeat_dim in dataset[var].dims):
                            idim=dataset[var].dims.index(repeat_dim)
                            for comp in comps:
                                self.check_repeat_err_corr_same(repeat_dims_errcorrs[repeat_dim],dataset[comp],idim)
                                repeat_dims_errcorrs[repeat_dim]["form"]=dataset[comp].attrs["err_corr_%s_form"%(idim+1)]
                                repeat_dims_errcorrs[repeat_dim]["params"]=dataset[comp].attrs["err_corr_%s_params"%(idim+1)]
                                repeat_dims_errcorrs[repeat_dim]["units"]=dataset[comp].attrs["err_corr_%s_units"%(idim+1)]
                        else:
                            self.check_repeat_err_corr_same(repeat_dims_errcorrs[repeat_dim],"systematic")
                            repeat_dims_errcorrs[repeat_dim]["form"]="systematic"

        return repeat_dims_errcorrs

    def check_repeat_err_corr_same(self,repeat_dims_errcorr,uvar,idim=None):
        if repeat_dims_errcorr["form"] is None:
            pass

        elif isinstance(uvar,str):
            if repeat_dims_errcorr["form"]==uvar:
                pass
            else:
                raise ValueError("punpy.measurement_function: Not all included uncertainty contributions have the same error correlation along the repeat_dims. Either don't use repeat_dims or use a different method, where components are propagated seperately.")
        else:
            if (repeat_dims_errcorr["form"]==uvar.attrs["err_corr_%s_form"%(idim+1)]) and (repeat_dims_errcorr["params"]==uvar.attrs["err_corr_%s_params"%(idim+1)]) and (repeat_dims_errcorr["units"]==uvar.attrs["err_corr_%s_units"%(idim+1)]):
                pass
            else:
                raise ValueError("punpy.measurement_function: Not all included uncertainty contributions have the same error correlation along the repeat_dims. Either don't use repeat_dims or use a different method, where components are propagated seperately.")

    def find_comps(self,form,dataset,var,store_rel_unc=False):
        comps=dataset.unc[var].comps
        if comps is None:
            pass
        elif form=="tot":
            pass
        elif form=="ran" or form=="random":
            comps=dataset.unc[var].random_comps
        elif form=="sys" or form=="systematic":
            comps=dataset.unc[var].systematic_comps
        elif form=="str":
            comps=dataset.unc[var].structured_comps
        else:
            compname=self.templ.make_ucomp_name(form,store_rel_unc=store_rel_unc,var=var)
            if compname in comps:
                comps=[compname]
            else:
                comps=[]
        return comps

    def propagate_total(self, *args, expand=False):
        self.check_sizes(*args)
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc("tot", args, expand=expand)
        input_corr = self.get_input_corr("tot", args)
        if self.prop.verbose:
            print(
                "inputs extracted (%s s since creation of prop object)"
                % (time.time() - self.prop.starttime)
            )

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
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc("rand", args, expand=expand)
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
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc("syst", args, expand=expand)
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
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc("stru", args, expand=expand)
        input_corr = self.get_input_corr("stru", args)
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
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc(form, args, expand=expand)
        input_corr = self.get_input_corr(form, args)
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

    def get_input_qty(self, *args, expand=True):
        if len(self.xvariables) == 0:
            raise ValueError("Variables have not been specified.")
        else:
            inputs = np.empty(len(self.xvariables), dtype=object)
            for iv, var in enumerate(self.xvariables):
                found = False
                for dataset in args[0]:
                    try:
                        inputs[iv] = dataset[var].values
                        found = True
                    except:
                        continue
                if not found:
                    raise ValueError(
                        "Variable %s not found in provided datasets." % (var)
                    )

            if expand:
                datashape = inputs[0].shape
                for i in range(len(inputs)):
                    # if not self.param_fixed:
                    if len(inputs[i].shape) < len(datashape):
                        if inputs[i].shape[0] == datashape[1]:
                            inputs[i] = np.tile(inputs[i], (datashape[0], 1))
                        elif inputs[i].shape[0] == datashape[0]:
                            inputs[i] = np.tile(inputs[i], (datashape[1], 1)).T

            return inputs

    def get_input_unc(self, form, *args, expand=True):
        inputs_unc = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            inputs_unc[iv] = None
            for dataset in args[0]:
                if var in dataset.keys():
                    inputs_unc[iv] = self.calculate_unc(form, dataset, var)
                    if inputs_unc[iv] is not None:
                        # this if else is to be removed when relative uncertainty implemented in obsarray
                        if "pressure" in dataset.keys():
                            inputs_unc[iv] = inputs_unc[iv]
                        else:
                            inputs_unc[iv] = inputs_unc[iv] * dataset[var].values

        if inputs_unc[iv] is None:
            if self.prop.verbose:
                print(
                    "%s uncertainty for variable %s not found in provided datasets. Zero uncertainty assumed."
                    % (form, var)
                )
        return inputs_unc

    def calculate_unc(self, form, ds, var):
        if form == "tot":
            data = ds.unc[var].total_unc()
        elif form == "rand":
            data = ds.unc[var].random_unc()
        elif form == "syst":
            data = ds.unc[var].systematic_unc()
        elif form == "stru":
            data = ds.unc[var].structured_unc()
        else:
            try:
                uvar = "%s_%s" % (form, var)
                data = ds[uvar]
            except:
                try:
                    keys = np.array(list(dataset.keys()))
                    uvar = keys[np.where("_%s_%s" % (form, var) in keys)]
                    data = ds[uvar]
                except:
                    data = None
        if isinstance(data, xr.DataArray):
            data = data.values
        return data

    def get_input_corr(self, form, *args):
        inputs_corr = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            inputs_corr[iv] = None
            for dataset in args[0]:
                if var in dataset.keys():
                    inputs_corr[iv] = self.calculate_corr(form, dataset, var)
            if inputs_corr[iv] is None:
                if self.prop.verbose:
                    print(
                        "%s error-correlation for variable %s not found in provided datasets."
                        % (form, var)
                    )

        return inputs_corr

    def calculate_corr(self, form, ds, var):
        sli = list([slice(None)] * ds[var].ndim)
        var_dims = ds[var].dims
        for i in range(len(sli)):
            if var_dims[i] in self.str_repeat_dims:
                sli[i] = 0
        dsu = ds.unc[var][tuple(sli)]

        if form == "tot":
            return dsu.total_err_corr_matrix().values
        elif form == "stru":
            return dsu.structured_err_corr_matrix().values
        elif form == "rand":
            return "rand"
        elif form == "syst":
            corrlen = len(dsu.values.ravel())
            return "syst"
        else:
            try:
                uvar = "%s_%s" % (form, var)
                return dsu[uvar].err_corr_matrix().values
            except:
                try:
                    keys = np.array(list(dataset.keys()))
                    uvar = keys[np.where("_%s_%s" % (form, var) in keys)]
                    return dsu[uvar].err_corr_matrix().values
                except:
                    return None
