"""Use Monte Carlo to propagate uncertainties"""

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class MeasurementFunctionUtils():
    def __init__(self,xvariables,str_repeat_dims,verbose,templ):
        self.xvariables=xvariables
        self.verbose=verbose
        self.str_repeat_dims=str_repeat_dims
        self.templ=templ

    def find_repeat_dim_corr(self, form, *args, store_unc_percent=False):
        """

        :param form:
        :type form:
        :param args:
        :type args:
        :param store_unc_percent:
        :type store_unc_percent:
        :return:
        :rtype:
        """
        repeat_dims_errcorrs = {}
        for repeat_dim in self.str_repeat_dims:
            repeat_dims_errcorrs[repeat_dim] = {
                "dim": repeat_dim,
                "form": None,
                "params": [],
                "units": [],
            }
            for iv, var in enumerate(self.xvariables):
                for dataset in args:
                    if var in dataset.keys():
                        comps = self.find_comps(
                            form, dataset, var, store_unc_percent=store_unc_percent
                        )
                        if comps is None:
                            continue
                        elif repeat_dim in dataset[var].dims:
                            idim = dataset[var].dims.index(repeat_dim)
                            for comp in comps:
                                self.check_repeat_err_corr_same(
                                    repeat_dims_errcorrs[repeat_dim],
                                    dataset[comp],
                                    idim,
                                )
                                repeat_dims_errcorrs[repeat_dim]["form"] = dataset[
                                    comp
                                ].attrs["err_corr_%s_form" % (idim + 1)]
                                repeat_dims_errcorrs[repeat_dim]["params"] = dataset[
                                    comp
                                ].attrs["err_corr_%s_params" % (idim + 1)]
                                repeat_dims_errcorrs[repeat_dim]["units"] = dataset[
                                    comp
                                ].attrs["err_corr_%s_units" % (idim + 1)]
                        else:
                            self.check_repeat_err_corr_same(
                                repeat_dims_errcorrs[repeat_dim], "systematic"
                            )
                            repeat_dims_errcorrs[repeat_dim]["form"] = "systematic"

        return repeat_dims_errcorrs

    def check_repeat_err_corr_same(self, repeat_dims_errcorr, uvar, idim=None):
        if repeat_dims_errcorr["form"] is None:
            pass

        elif isinstance(uvar, str):
            if repeat_dims_errcorr["form"] == uvar:
                pass
            else:
                raise ValueError(
                    "punpy.measurement_function: Not all included uncertainty contributions have the same error correlation along the repeat_dims. Either don't use repeat_dims or use a different method, where components are propagated seperately."
                )
        else:
            if (
                (
                    repeat_dims_errcorr["form"]
                    == uvar.attrs["err_corr_%s_form" % (idim + 1)]
                )
                and (
                    repeat_dims_errcorr["params"]
                    == uvar.attrs["err_corr_%s_params" % (idim + 1)]
                )
                and (
                    repeat_dims_errcorr["units"]
                    == uvar.attrs["err_corr_%s_units" % (idim + 1)]
                )
            ):
                pass
            else:
                raise ValueError(
                    "punpy.measurement_function: Not all included uncertainty contributions have the same error correlation along the repeat_dims. Either don't use repeat_dims or use a different method, where components are propagated seperately."
                )

    def find_comps(self, form, dataset, var, store_unc_percent=False):
        comps = dataset.unc[var].comps
        if comps is None:
            pass
        elif form == "tot":
            pass
        elif form == "ran" or form == "random":
            comps = dataset.unc[var].random_comps
        elif form == "sys" or form == "systematic":
            comps = dataset.unc[var].systematic_comps
        elif form == "str":
            comps = dataset.unc[var].structured_comps
        else:
            compname = self.templ.make_ucomp_name(
                form, store_unc_percent=store_unc_percent, var=var
            )
            if compname in comps:
                comps = [compname]
            else:
                comps = []
        return comps

    def get_input_qty(self, *args, expand=False):
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

    def get_input_unc(self, form, *args, expand=False):
        inputs_unc = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            inputs_unc[iv] = None
            for dataset in args[0]:
                if var in dataset.keys():
                    inputs_unc[iv] = self.calculate_unc(form, dataset, var)
                    if inputs_unc[iv] is not None:
                        # this if else is to be removed when relative uncertainty implemented in obsarray
                        inputs_unc[iv] = inputs_unc[iv]

        if inputs_unc[iv] is None:
            if self.verbose:
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
                if data.attrs["units"]=="%":
                    return data.values/100*ds[var].values
            except:
                keys = np.array(list(ds.keys()))
                uvar_ids = [
                    ("_%s_%s" % (form, var) in key) and (key[0] == "u") for key in keys
                ]
                uvar = keys[uvar_ids]
                if len(uvar) > 0:
                    data = ds[uvar[0]]
                    if data.attrs["units"]=="%":
                        return data.values/100*ds[var].values
                    else:
                        return data.values
                else:
                    data = None

        if data is not None:
            return data.values

    def get_input_corr(self, form, *args):
        inputs_corr = np.empty(len(self.xvariables), dtype=object)
        for iv, var in enumerate(self.xvariables):
            inputs_corr[iv] = None
            for dataset in args[0]:
                if var in dataset.keys():
                    inputs_corr[iv] = self.calculate_corr(form, dataset, var)
            if inputs_corr[iv] is None:
                if self.verbose:
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
            return "syst"
        else:
            try:
                uvar = "%s_%s" % (form, var)
                data = ds[uvar]
            except:
                keys = np.array(list(ds.keys()))
                uvar_ids = [
                    ("_%s_%s" % (form, var) in key) and (key[0] == "u") for key in keys
                ]
                uvar = keys[uvar_ids]
                if len(uvar) > 0:
                    data = ds[uvar[0]]
                else:
                    data = None

            try:
                uvar = "%s_%s" % (form, var)
                return dsu[uvar].err_corr_matrix().values
            except:
                keys = np.array(list(ds.keys()))
                uvar_ids = [
                    ("_%s_%s" % (form, var) in key) and (key[0] == "u") for key in keys
                ]
                uvar = keys[uvar_ids]
                if len(uvar) > 0:
                    return dsu[uvar[0]].err_corr_matrix().values
                else:
                    return None
