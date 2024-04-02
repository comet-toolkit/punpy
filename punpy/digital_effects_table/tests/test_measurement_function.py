"""
Tests for mc propagation class
"""

import os.path
import unittest

import numpy as np
import numpy.testing as npt
import xarray as xr

np.random.seed(2222)

from punpy import MeasurementFunction, MCPropagation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "28/07/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class HypernetsMF(MeasurementFunction):
    def setup(self, min_value):
        self.min_value = min_value

    def meas_function(self, digital_number, gains, dark_signal, non_linear, int_time):
        DN = digital_number - dark_signal
        DN[DN == 0] = self.min_value
        corrected_DN = DN / (
            non_linear[0]
            + non_linear[1] * DN
            + non_linear[2] * DN**2
            + non_linear[3] * DN**3
            + non_linear[4] * DN**4
            + non_linear[5] * DN**5
            + non_linear[6] * DN**6
            + non_linear[7] * DN**7
        )
        if gains.ndim == 1:
            return gains[:, None] * corrected_DN / int_time * 1000
        else:
            return gains * corrected_DN / int_time * 1000

    def get_argument_names(self):
        return [
            "digital_number",
            "gains",
            "dark_signal",
            "non_linearity_coefficients",
            "integration_time",
        ]


dir_path = os.path.dirname(os.path.realpath(__file__))
calib_data = xr.open_dataset(os.path.join(dir_path, "det_hypernets_cal.nc"))
L0data = xr.open_dataset(os.path.join(dir_path, "det_hypernets_l0.nc"))
L1data = xr.open_dataset(os.path.join(dir_path, "test_l1.nc"))


# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw(MeasurementFunction):
    @staticmethod
    def meas_function(pres, temp, n):
        return (n * temp * 8.134) / pres


# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw_R(MeasurementFunction):
    @staticmethod
    def meas_function(pres, temp, n, R):
        return (n * temp * R) / pres


# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw_2out(MeasurementFunction):
    def meas_function(self, pres, temp, n):
        return (n * temp * 8.134) / pres, pres / temp


# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw_2out_diffdim(MeasurementFunction):
    def meas_function(self, pres, temp, n):
        return (n * temp * 8.134) / pres, np.mean(pres / temp, axis=0)


dir_path = os.path.dirname(os.path.realpath(__file__))
ds = xr.open_dataset(os.path.join(dir_path, "digital_effects_table_gaslaw_example.nc"))

volume = np.ones(ds["temperature"].values.shape) * 0.9533
u_tot_volume = (
    np.ones(ds["temperature"].values.shape)
    * 0.9533
    * ((1 / 10000) ** 2 + (1 / 293) ** 2 + (0.4 / 293) ** 2 + (1 / 40) ** 2) ** 0.5
)

u_ran_volume = (
    np.ones(ds["temperature"].values.shape)
    * 0.9533
    * ((1 / 293) ** 2 + (1 / 40) ** 2) ** 0.5
)
u_sys_volume = np.ones(ds["temperature"].values.shape) * 0.9533 * (0.4 / 293)
u_str_volume = np.ones(ds["temperature"].values.shape) * 0.9533 * (1 / 10000)


class TestMeasurementFunction(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_gaslaw(self):

        prop = MCPropagation(1000, dtype="float32", verbose=False, parallel_cores=4)

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_(
            ds_y_tot["u_tot_volume"].attrs["err_corr_1_params"][0]
            in list(ds_y_tot.variables)
        )

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.1)

        prop = MCPropagation(1000, dtype="float32", verbose=False, parallel_cores=1)

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            use_err_corr_dict=True,
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        prop = MCPropagation(3000, dtype="float32", verbose=False)
        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            repeat_dims=[0, 2],
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_(
            ds_y["u_str_volume"].attrs["err_corr_3_params"][0] in list(ds_y.variables)
        )

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)
        npt.assert_allclose(ds_y["u_ran_volume"].values, u_ran_volume, rtol=0.08)
        npt.assert_allclose(ds_y["u_sys_volume"].values, u_sys_volume, rtol=0.08)
        npt.assert_allclose(ds_y["u_str_volume"].values, u_str_volume, rtol=0.08)
        npt.assert_allclose(ds_y["u_str_volume"].values, u_str_volume, rtol=0.08)
        npt.assert_allclose(
            ds_y.unc["volume"].total_unc().values, u_tot_volume, rtol=0.12
        )

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            uncxvariables=["pressure"],
            yvariable="volume",
            yunit="m^3",
            repeat_dims=[0, 2],
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)

    def test_gaslaw_scalar(self):
        prop = MCPropagation(100, dtype="float32", verbose=True, parallel_cores=1)

        gl = IdealGasLaw_R(
            prop,
            ["pressure", "temperature", "n_moles", "R"],
            yvariable="volume",
            yunit="m^3",
        )
        ds_y_tot = gl.propagate_ds_total(ds)
        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.03)

    def test_gaslaw_2out(self):
        prop = MCPropagation(1000, dtype="float32", verbose=True)

        gl = IdealGasLaw_2out(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable=["volume", "P/T"],
            yunit=["m^3", "Pa/K"],
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)
        npt.assert_equal(ds_y_tot["err_corr_tot_volume"].values.shape, (3600, 3600))
        npt.assert_equal(ds_y_tot["err_corr_tot_P/T"].values.shape, (3600, 3600))

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.1)

    def test_gaslaw_2out_diffdim(self):
        prop = MCPropagation(100, dtype="float32", verbose=False, parallel_cores=1)

        gl = IdealGasLaw_2out_diffdim(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable=["volume", "P/T"],
            yunit=["m^3", "Pa/K"],
            ydims=[["x", "y", "time"], ["y", "time"]],
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)
        npt.assert_equal(ds_y["err_corr_str_volume"].values.shape, (3600, 3600))
        npt.assert_equal(ds_y["err_corr_str_P/T"].values.shape, (180, 180))

        gl = IdealGasLaw_2out_diffdim(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable=["volume", "P/T"],
            yunit=["m^3", "Pa/K"],
            ydims=[["x", "y", "time"], ["y", "time"]],
            corr_dims=["x", None],
            separate_corr_dims=True,
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)
        npt.assert_equal(ds_y["err_corr_str_volume_x"].values.shape, (20, 20))

        gl = IdealGasLaw_2out_diffdim(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable=["volume", "P/T"],
            yunit=["m^3", "Pa/K"],
            ydims=[["x", "y", "time"], ["y", "time"]],
            corr_dims=["x"],
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)
        npt.assert_equal(ds_y["err_corr_str_volume_x"].values.shape, (20, 20))

        gl = IdealGasLaw_2out_diffdim(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable=["volume", "P/T"],
            yunit=["m^3", "Pa/K"],
            ydims=[["x", "y", "time"], ["y", "time"]],
            corr_dims=["x", "y.time"],
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)
        npt.assert_equal(ds_y_tot["err_corr_tot_volume_x"].values.shape, (20, 20))
        npt.assert_equal(
            ds_y_tot["err_corr_tot_volume_y.time"].values.shape, (180, 180)
        )
        npt.assert_equal(ds_y_tot["err_corr_tot_P/T_y.time"].values.shape, (180, 180))

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.4)

    def test_gaslaw_errcorrdict(self):
        prop = MCPropagation(1000, dtype="float32", verbose=False)

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            use_err_corr_dict=True,
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.1)

        prop = MCPropagation(3000, dtype="float32", verbose=False)
        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            repeat_dims=[0, 2],
        )
        ds_y = gl.propagate_ds(ds)

        npt.assert_allclose(ds_y["volume"].values, volume, rtol=0.002)
        npt.assert_allclose(ds_y["u_ran_volume"].values, u_ran_volume, rtol=0.06)
        npt.assert_allclose(ds_y["u_sys_volume"].values, u_sys_volume, rtol=0.06)
        npt.assert_allclose(ds_y["u_str_volume"].values, u_str_volume, rtol=0.07)
        npt.assert_allclose(
            ds_y.unc["volume"].total_unc().values, u_tot_volume, rtol=0.06
        )

    def test_gaslaw_corrdim(self):
        prop = MCPropagation(1000, dtype="float32", verbose=False)

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            corr_dims=["0.1", 2],
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.1)

        gl = IdealGasLaw(
            prop,
            ["pressure", "temperature", "n_moles"],
            yvariable="volume",
            yunit="m^3",
            corr_dims=["x.y", "time"],
        )
        ds_y_tot = gl.propagate_ds_total(ds)

        npt.assert_allclose(ds_y_tot["volume"].values, volume, rtol=0.002)

        npt.assert_allclose(ds_y_tot["u_tot_volume"].values, u_tot_volume, rtol=0.1)

    def test_hypernets(self):
        prop = MCPropagation(3000, dtype="float32", parallel_cores=1, verbose=False)

        hmf = HypernetsMF(
            prop=prop,
            xvariables=[
                "digital_number",
                "gains",
                "dark_signal",
                "non_linearity_coefficients",
                "integration_time",
            ],
            yvariable="irradiance",
            yunit="W m^-2",
            corr_between=None,
            param_fixed=None,
        )

        hmf.setup(0.1)
        y = hmf.run(calib_data, L0data.variables, L0data)
        u_y_rand = hmf.propagate_random(L0data, calib_data)
        print(u_y_rand)
        # print(u_y_rand,L0data)
        mask = np.where(
            (
                (L1data["wavelength"].values < 1350)
                | (L1data["wavelength"].values > 1450)
            )
        )

        u_y_syst_indep = hmf.propagate_specific("systematic_indep", L0data, calib_data)
        u_y_syst_corr = hmf.propagate_specific(
            "u_rel_systematic_corr_rad_irr", L0data, calib_data
        )

        u_y_syst = (u_y_syst_indep**2 + u_y_syst_corr**2) ** 0.5
        u_y_tot = (u_y_syst_indep**2 + u_y_syst_corr**2 + u_y_rand**2) ** 0.5

        ds_tot = hmf.propagate_ds_total(L0data, calib_data, store_unc_percent=True)
        mask = np.where(
            (np.isfinite(u_y_tot / y) & np.isfinite(ds_tot["u_rel_tot_irradiance"]))
        )[0]
        npt.assert_allclose(
            ds_tot["u_rel_tot_irradiance"][mask],
            u_y_tot[mask] / y[mask] * 100,
            rtol=0.05,
            atol=0.05,
        )

    def test_hypernets_expand(self):
        prop = MCPropagation(3000, dtype="float32", parallel_cores=1, verbose=False)

        hmf = HypernetsMF(
            prop=prop,
            xvariables=[
                "digital_number",
                "gains",
                "dark_signal",
                "non_linearity_coefficients",
                "integration_time",
            ],
            uncxvariables=["gains"],
            yvariable="irradiance",
            yunit="W m^-2",
            corr_between=None,
            param_fixed=[False, False, False, True, False],
            broadcast_correlation="rand",
        )

        hmf.setup(0.1)

        ds_tot = hmf.propagate_ds_total(
            L0data, calib_data, store_unc_percent=True, expand=True
        )
        npt.assert_allclose(
            ds_tot["err_corr_tot_irradiance"].values[:9, :9], np.eye(9), atol=0.07
        )

        ds_tot = hmf.propagate_ds_total(
            L0data, calib_data, store_unc_percent=True, expand=False
        )
        npt.assert_allclose(
            ds_tot["err_corr_tot_irradiance"].values[:9, :9], np.ones((9, 9)), atol=0.07
        )

    def test_hypernets_repeat_dim(self):
        prop = MCPropagation(3000, dtype="float32", parallel_cores=0, verbose=False)

        hmf = HypernetsMF(
            prop=prop,
            yvariable="irradiance",
            yunit="W m^-2",
            corr_between=None,
            repeat_dims="scan",
            corr_dims=-99,
        )
        hmf.setup(0.1)
        y = hmf.run(calib_data, L0data)
        u_y_rand = hmf.propagate_random(L0data, calib_data)
        # print(list(L1data.variables))
        mask = np.where(
            (
                (L1data["wavelength"].values < 1350)
                | (L1data["wavelength"].values > 1450)
            )
        )

        npt.assert_allclose(L1data["irradiance"].values, y, rtol=0.03)

        npt.assert_allclose(
            L1data["u_rel_random_irradiance"].values[mask][
                np.where(np.isfinite(u_y_rand[mask]))
            ],
            (u_y_rand[mask] / y[mask] * 100)[np.where(np.isfinite(u_y_rand[mask]))],
            rtol=0.03,
            atol=0.2,
        )

        ds_all = hmf.propagate_ds_all(L0data, calib_data, store_unc_percent=True)
        ds_main = hmf.propagate_ds(L0data, calib_data, store_unc_percent=True)
        ds_spec = hmf.propagate_ds_specific(
            ["random", "systematic_indep", "systematic_corr_rad_irr"],
            L0data,
            calib_data,
            store_unc_percent=True,
        )

        ds_main.to_netcdf("propagate_ds_example.nc")

        u_y_syst_indep = hmf.propagate_specific(
            "u_rel_systematic_indep", L0data, calib_data
        )
        u_y_syst_corr = hmf.propagate_specific(
            "u_rel_systematic_corr_rad_irr", L0data, calib_data, return_corr=False
        )

        u_y_syst = (u_y_syst_indep**2 + u_y_syst_corr**2) ** 0.5
        u_y_tot = (u_y_syst_indep**2 + u_y_syst_corr**2 + u_y_rand**2) ** 0.5

        npt.assert_allclose(
            L1data["u_rel_systematic_indep_irradiance"].values[mask],
            u_y_syst_indep[mask] / y[mask] * 100,
            rtol=0.03,
            atol=0.2,
        )

        npt.assert_allclose(
            L1data["u_rel_systematic_corr_rad_irr_irradiance"].values[mask],
            u_y_syst_corr[mask] / y[mask] * 100,
            rtol=0.03,
            atol=0.3,
        )

        npt.assert_allclose(
            L1data["u_rel_systematic_indep_irradiance"].values[mask],
            ds_spec["u_rel_systematic_indep_irradiance"].values[mask],
            rtol=0.03,
            atol=0.2,
        )

        npt.assert_allclose(
            L1data["u_rel_systematic_corr_rad_irr_irradiance"].values[mask],
            ds_spec["u_rel_systematic_corr_rad_irr_irradiance"].values[mask],
            rtol=0.03,
            atol=0.3,
        )

        npt.assert_allclose(
            L1data["u_rel_systematic_indep_irradiance"].values[mask],
            ds_all["u_rel_systematic_indep_irradiance"].values[mask],
            rtol=0.03,
            atol=0.2,
        )

        npt.assert_allclose(
            L1data["u_rel_systematic_corr_rad_irr_irradiance"].values[mask],
            ds_all["u_rel_systematic_corr_rad_irr_irradiance"].values[mask],
            rtol=0.03,
            atol=0.3,
        )
        # plt.plot(
        #     L1data["wavelength"][mask],
        #     ds_main["u_rel_str_irradiance"][mask] - (u_y_syst[mask] / y[mask]),
        #     "r-",
        # )
        # plt.show()

        npt.assert_allclose(
            ds_main["u_rel_ran_irradiance"][mask],
            u_y_rand[mask] / y[mask] * 100,
            rtol=0.03,
            atol=0.2,
        )
        npt.assert_allclose(
            ds_main["u_rel_str_irradiance"][mask],
            u_y_syst[mask] / y[mask] * 100,
            rtol=0.03,
            atol=0.2,
        )


if __name__ == "__main__":
    unittest.main()
