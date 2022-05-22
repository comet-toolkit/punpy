"""Use Monte Carlo to propagate uncertainties"""

from abc import ABC,abstractmethod

import numpy as np
import xarray as xr

import punpy.fiduceo.fiduceo_correlations as fc
from punpy.lpu.lpu_propagation import LPUPropagation
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
        variables,
        proptype,
        corr_between=None,
        param_fixed=None,
        output_vars=1,
        repeat_dims=-99,
        corr_axis=-99,
        steps=10000,
        parallel_cores=0,
        dtype=None,
        Jx_diag=False,
        step=None,
    ):
        """
        Initialise FiduceoMeasurementFunction
        """
        # self.measurandstring = measurandstring
        self.variables = variables
        if proptype.lower() == "mc":
            self.prop = MCPropagation(steps, parallel_cores, dtype)
        elif proptype.lower() == "lpu":
            self.prop = LPUPropagation(parallel_cores, Jx_diag, step)
        else:
            raise ValueError("punpy: %s (second argument of MeasurementFunction class initialiser) is not a valid uncertainty propagation method. Pleasee use MC or LPU.")
        self.corr_between = corr_between
        self.param_fixed = param_fixed
        self.output_vars = output_vars
        self.repeat_dims = repeat_dims
        self.corr_axis = corr_axis

    @abstractmethod
    def meas_function(self):
        pass

    def propagate_ds(self, measurandstring, *args):
        y = self.run(*args)
        u_rand_y = self.propagate_random(*args)
        u_syst_y = self.propagate_systematic(*args)

        dims = args[0][self.variables[0]].dims
        coords = args[0][self.variables[0]].coords

        ds_out = xr.Dataset(
            {measurandstring: (dims, y)},
            coords=coords,
            attrs={},
        )
        ds_out.unc[measurandstring]["u_ran_" + measurandstring] = (
            dims,
            u_rand_y,
            {
                "err_corr": [
                    {
                        "dim": dim,
                        "form": "random",
                        "params": [],
                    }
                    for dim in dims
                ]
            },
        )
        ds_out.unc[measurandstring]["u_sys_" + measurandstring] = (
            dims,
            u_syst_y,
            {
                "err_corr": [
                    {
                        "dim": dim,
                        "form": "systematic",
                        "params": [],
                    }
                    for dim in dims
                ]
            },
        )

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
        return self.prop.propagate_standard(
            self.meas_function,
            input_qty,
            input_unc,
            input_corr,
            param_fixed=self.param_fixed,
            corr_between=self.corr_between,
            return_corr=False,
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

    def propagate_specific(self, form, *args, expand=True):
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
            return_corr=False,
            return_samples=False,
            repeat_dims=self.repeat_dims,
            corr_axis=self.corr_axis,
            output_vars=self.output_vars,
        )

    def get_input_qty(self, *args, expand=True):
        if len(self.variables) == 0:
            raise ValueError("Variables have not been specified.")
        else:
            inputs = np.empty(len(self.variables), dtype=object)
            for iv, var in enumerate(self.variables):
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
        inputs_unc = np.empty(len(self.variables), dtype=object)
        for iv, var in enumerate(self.variables):
            if form == "tot":
                inputs_unc[iv] = self.get_input_unc_total(var, *args)
            elif form == "rand":
                inputs_unc[iv] = self.get_input_unc_random(var, *args)
            elif form == "syst":
                inputs_unc[iv] = self.get_input_unc_systematic(var, *args)
            else:
                inputs_unc[iv] = self.get_input_unc_specific(form, var, *args)

        # if expand:
        #     datashape = inputs_unc[0].shape
        #     for i in range(len(inputs_unc)):
        #         if inputs_unc[i] is not None:
        #             if len(inputs_unc[i].shape) < len(datashape):
        #                 inputs_unc[i] = np.tile(inputs_unc[i], (datashape[1], 1)).T
        return inputs_unc

    def get_input_unc_total(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            data = dataset.unc[var].total
            if not isinstance(data, float):
                inputs_unc = data.values
        return inputs_unc

    def get_input_unc_random(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            data = dataset.unc[var].random
            if not isinstance(data, float):
                inputs_unc = data.values
        return inputs_unc

    def get_input_unc_systematic(self, var, *args):
        inputs_unc = None
        for dataset in args[0]:
            data = dataset.unc[var].systematic
            if not isinstance(data, float):
                inputs_unc = data.values
        return inputs_unc

    def get_input_unc_specific(self, form, var, *args):
        uvar = "u_%s_%s" % (form, var)
        found = False
        for dataset in args[0]:
            try:
                inputs_unc = dataset[uvar].values
                if "rel_" in form:
                    inputs_unc *= dataset[var].values
                found = True
            except:
                continue
        if not found:
            inputs_unc = None
            print(
                "%s uncertainty for variable %s (%s) not found in provided datasets. Zero uncertainty assumed."
                % (form, var, uvar)
            )
        return inputs_unc

    def get_input_corr(self, form, *args):
        inputs_corr = np.empty(len(self.variables), dtype=object)
        for iv, var in enumerate(self.variables):
            found = False
            for dataset in args[0]:
                try:
                    inputs_corr[iv] = fc.calculate_corr(
                        var, form, dataset, self.repeat_dims
                    )
                    found = True
                except:
                    continue
            if not found:
                raise ValueError(
                    "Correlation for variable %s not found in provided datasets."
                    % (var)
                )

        return inputs_corr
