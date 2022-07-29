"""Use Monte Carlo to propagate uncertainties"""

import time
from abc import ABC,abstractmethod

import numpy as np
import obsarray
import xarray as xr
from punpy.digital_effects_table.digital_effects_table_templates import\
    DigitalEffectsTableTemplates

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
        corr_between=None,
        param_fixed=None,
        output_vars=1,
        repeat_dims=-99,
        corr_axis=-99,
        auto_simplify_corr=False,
    ):
        """
        Initialise MeasurementFunction
        """
        self.prop = prop

        self.xvariables = xvariables
        self.yvariable = yvariable
        self.templ=DigitalEffectsTableTemplates(yvariable,yunit,repeat_dims=repeat_dims)

        self.corr_between = corr_between
        self.param_fixed = param_fixed
        self.output_vars = output_vars
        if isinstance(repeat_dims, int):
            repeat_dims = [repeat_dims]
        self.repeat_dims = repeat_dims
        self.corr_axis = corr_axis
        self.auto_simplify_corr = auto_simplify_corr

    @abstractmethod
    def meas_function(self):
        """
        meas_function is the measurement function itself, to be used in the uncertainty propagation.
        This function must be overwritten by the user when creating their MeasurementFunction subclass.
        """
        pass

    def propagate_ds(self, *args):
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
        if self.prop.verbose:
            print("starting propagate_ds (%s s since creation of prop object)"%(time.time()-self.prop.starttime))

        #first calculate the measurand and propagate the uncertainties
        y = self.run(*args)

        u_rand_y = self.propagate_random(*args)
        if self.prop.verbose:
            print("propagate_random done (%s s since creation of prop object)"%(time.time()-self.prop.starttime))

        u_syst_y = self.propagate_systematic(*args)
        if self.prop.verbose:
            print("propagate systematic done (%s s since creation of prop object)"%(time.time()-self.prop.starttime))

        u_stru_y, corr_stru_y = self.propagate_structured(*args)

        xvar_ref=args[0][self.xvariables[0]]
        ucomp=xvar_ref.unc_comps
        if not isinstance(ucomp, str):
            ucomp=ucomp[0]
        u_xvar_ref=args[0][ucomp]
        dims = xvar_ref.dims

        template,dim_sizes=self.templ.make_template_all(dims,u_xvar_ref)

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y
        ds_out["u_ran_"+ self.yvariable].values = u_rand_y
        ds_out["u_sys_"+ self.yvariable].values = u_syst_y
        ds_out["u_str_"+ self.yvariable].values = u_stru_y
        ds_out["err_corr_str_"+ self.yvariable].values = corr_stru_y

        if self.prop.verbose:
            print("finishing propagate_ds (%s s since creation of prop object)"%(time.time()-self.prop.starttime))

        return ds_out

    def propagate_ds_total(self, *args):
        """
        Function to propagate the total uncertainties present in the digital effects
        tables in the input arguments, through the measurement function to produce
        an output digital effects table with the total uncertainties on the measurand

        :param args: One or multiple digital effects tables with input quantities, defined with obsarray
        :type args: obsarray dataset(s)
        :return: digital effects table with uncertainties on measurand
        :rtype: obsarray dataset
        """
        if self.prop.verbose:
            print("starting propagate_ds_total (%s s since creation of prop object)"%(time.time()-self.prop.starttime))
        y = self.run(*args)
        u_tot_y, corr_tot_y = self.propagate_total(*args)

        xvar_ref=args[0][self.xvariables[0]]
        ucomp=xvar_ref.unc_comps
        if not isinstance(ucomp, str):
            ucomp=ucomp[0]
        u_xvar_ref=args[0][ucomp]
        dims = xvar_ref.dims

        template, dim_sizes=self.templ.make_template_tot(dims,u_xvar_ref)

        # create dataset template
        ds_out = obsarray.create_ds(template, dim_sizes)

        ds_out[self.yvariable].values = y
        ds_out["u_tot_"+ self.yvariable].values = u_tot_y
        ds_out["err_corr_tot_"+ self.yvariable].values = corr_tot_y
        if self.prop.verbose:
            print("finishing propagate_ds_total (%s s since creation of prop object)"%(time.time()-self.prop.starttime))
        return ds_out

    def run(self, *args, expand=True):
        input_qty = self.get_input_qty(args, expand=expand)
        if self.repeat_dims is None:
            return self.meas_function(*input_qty)
        else:
            return self.meas_function(*input_qty)

    def propagate_total(self, *args, expand=True):
        input_qty = self.get_input_qty(args, expand=expand)
        input_unc = self.get_input_unc("tot", args, expand=expand)
        input_corr = self.get_input_corr("tot", args)
        if self.prop.verbose:
            print("inputs extracted (%s s since creation of prop object)"%(time.time()-self.prop.starttime))

        return self.prop.propagate_standard(
            self.meas_function,
            input_qty,
            input_unc,
            input_corr,
            param_fixed=self.param_fixed,
            corr_between=self.corr_between,
            return_corr=True,
            return_samples=False,
            repeat_dims=self.repeat_dims,
            corr_axis=self.corr_axis,
            output_vars=self.output_vars,
        )

    def propagate_random(self, *args, expand=True):
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
            repeat_dims=self.repeat_dims,
            corr_axis=self.corr_axis,
            output_vars=self.output_vars,
        )

    def propagate_systematic(self, *args, expand=True):
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
            repeat_dims=self.repeat_dims,
            corr_axis=self.corr_axis,
            output_vars=self.output_vars,
        )

    def propagate_structured(self, *args, expand=True):
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
            repeat_dims=self.repeat_dims,
            corr_axis=self.corr_axis,
            output_vars=self.output_vars,
        )

    def propagate_specific(self, form, *args, expand=True, return_corr=False):
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
            repeat_dims=self.repeat_dims,
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
                    if len(inputs[i].shape) < len(datashape):
                        if inputs[i].shape[0] == datashape[1]:
                            inputs[i] = np.tile(inputs[i], (datashape[0], 1))
                        elif inputs[i].shape[0] == datashape[0]:
                            inputs[i] = np.tile(inputs[i], (datashape[1], 1)).T

            return inputs

    def get_input_unc(self, form, *args, expand=True):
        inputs_unc = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            if form == "tot":
                inputs_unc[iv] = self.get_input_unc_total(var, *args)
            elif form == "rand":
                inputs_unc[iv] = self.get_input_unc_random(var, *args)
            elif form == "syst":
                inputs_unc[iv] = self.get_input_unc_systematic(var, *args)
            elif form == "stru":
                inputs_unc[iv] = self.get_input_unc_structured(var, *args)
            else:
                inputs_unc[iv] = self.get_input_unc_specific(form, var, *args)

        return inputs_unc

    def get_input_unc_total(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            if var in dataset.keys():
                data = dataset.unc[var].total_unc()
                if isinstance(data, xr.DataArray):
                    inputs_unc = data.values
        return inputs_unc

    def get_input_unc_random(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            if var in dataset.keys():
                data = dataset.unc[var].random_unc()
                if isinstance(data, xr.DataArray):
                    inputs_unc = data.values
        return inputs_unc

    def get_input_unc_systematic(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            if var in dataset.keys():
                data = dataset.unc[var].systematic_unc()
                if isinstance(data, xr.DataArray):
                    inputs_unc = data.values
        return inputs_unc

    def get_input_unc_structured(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            if var in dataset.keys():
                data = dataset.unc[var].structured_unc()
                if isinstance(data, xr.DataArray):
                    inputs_unc = data.values
        return inputs_unc

    def get_input_unc_specific(self, form, var, *args):
        found = False
        for dataset in args[0]:
            if var in dataset.keys():
                try:
                    uvar = "%s_%s" % (form, var)
                    inputs_unc = dataset[uvar].values
                    found = True
                except:
                    keys = np.array(list(dataset.keys()))
                    print(keys)
                    uvar = keys[np.where("_%s_%s" % (form, var) in keys)]
                    inputs_unc = dataset[uvar].values
                    found = True
        if not found:
            inputs_unc = None
            print(
                "%s uncertainty for variable %s (%s) not found in provided datasets. Zero uncertainty assumed."
                % (form, var, uvar)
            )
        return inputs_unc

    def get_input_corr(self, form, *args):
        inputs_corr = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            found = False
            for dataset in args[0]:
                if var in dataset.keys():
                    inputs_corr[iv] = self.calculate_corr(
                        form, dataset, var
                    )
                    found = True
            if not found:
                inputs_corr[iv] = None
                print("%s error-correlation for variable %s not found in provided datasets."
                    % (form,var)
                )

        return inputs_corr

    def calculate_corr(self, form, ds, var):
        sli = list([slice(None)] * ds[var].ndim)
        for repeat_dim in self.repeat_dims:
            if repeat_dim>=0:
                sli[repeat_dim]=0
        dsu = ds.unc[var][tuple(sli)]

        if form=="tot":
            return dsu.total_err_corr_matrix().values
        elif form=="stru":
            return dsu.structured_err_corr_matrix().values
        elif form=="rand":
            corrlen=len(dsu.values.ravel())
            return np.eye(corrlen)
        elif form=="syst":
            corrlen=len(dsu.values.ravel())
            return np.ones((corrlen.corrlen))
        else:
            return dsu[form].err_corr_matrix().values



