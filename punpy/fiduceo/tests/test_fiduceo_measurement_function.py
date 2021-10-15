"""
Tests for mc propagation class
"""
import os.path
import unittest

import xarray as xr

from punpy.fiduceo.fiduceo_measurement_function import FiduceoMeasurementFunction

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "28/07/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class HypernetsMF(FiduceoMeasurementFunction):
    def function(self, digital_number, gains, dark_signal, non_linear, int_time):
        DN = digital_number - dark_signal
        DN[DN == 0] = 1
        corrected_DN = DN / (
            non_linear[0]
            + non_linear[1] * DN
            + non_linear[2] * DN ** 2
            + non_linear[3] * DN ** 3
            + non_linear[4] * DN ** 4
            + non_linear[5] * DN ** 5
            + non_linear[6] * DN ** 6
            + non_linear[7] * DN ** 7
        )
        return gains * corrected_DN / int_time * 1000


dir_path = os.path.dirname(os.path.realpath(__file__))
calib_data = xr.open_dataset(os.path.join(dir_path, "test_cal.nc"))
L0data = xr.open_dataset(os.path.join(dir_path, "test_l0.nc"))
L1data = xr.open_dataset(os.path.join(dir_path, "test_l1.nc"))


class TestFiduceoMeasurementFunction(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_end_to_end(self):
        pass
        # hmf = HypernetsMF(
        #     [
        #         "digital_number",
        #         "gains",
        #         "dark_signal",
        #         "non_linearity_coefficients",
        #         "integration_time",
        #     ],
        #     corr_between=None,
        #     param_fixed=[False, False, False, True, False],
        #     output_vars=1,
        #     repeat_dims=1,
        #     corr_axis=-99,
        #     mc=True,
        #     steps=10000,
        #     parallel_cores=0,
        #     dtype=None,
        # )
        # y = hmf.run(calib_data, L0data)
        # u_y = hmf.propagate_u("rel_random", L0data, calib_data)
        # # print(list(L1data.variables))
        # mask = np.where(
        #     (L1data["wavelength"].values > 1350) & (L1data["wavelength"].values < 1400)
        # )
        #
        # npt.assert_allclose(L1data["irradiance"].values, y, rtol=0.03)
        # npt.assert_allclose(
        #     L1data["u_rel_random_irradiance"].values[not mask],
        #     u_y[not mask] / y[not mask],
        #     rtol=0.03,
        # )


if __name__ == "__main__":
    unittest.main()
