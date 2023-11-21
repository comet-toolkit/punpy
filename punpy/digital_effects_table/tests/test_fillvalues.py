"""
Tests for mc propagation class
"""
import os.path
import unittest

import numpy as np
import numpy.testing as npt
import xarray as xr
import obsarray

from punpy import MeasurementFunction, MCPropagation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "28/07/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

dir_path = os.path.dirname(os.path.realpath(__file__))
ds_gaslaw = xr.open_dataset(os.path.join(dir_path, "digital_effects_table_gaslaw_example.nc"))

# define dim_size_dict to specify size of arrays
dim_sizes = {
    "x": 20,
    "y": 30,
    "time": 6
}

# define ds variables
template = {
    "temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "unc_comps": ["u_ran_temperature"]
        },
    },
    "u_ran_temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "err_corr": [
              {
                  "dim": "x",
                  "form": "random",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "y",
                  "form": "random",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "time",
                  "form": "random",
                  "params": [],
                  "units": []
              }
          ]
        },
    }}

template_encoding = {
    "temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "unc_comps": ["u_ran_temperature"]
        },
        "encoding": {"dtype": np.uint16, "scale_factor": 0.01},
    },
    "u_ran_temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "err_corr": [
              {
                  "dim": "x",
                  "form": "random",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "y",
                  "form": "random",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "time",
                  "form": "random",
                  "params": [],
                  "units": []
              }
          ]
        },
    }}

class TestFillValue(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_to_netcdf_encoding_fixed(self):
        # create dataset template
        ds = obsarray.create_ds(template_encoding, dim_sizes)

        # populate with example data
        ds["temperature"].values = 293 * np.ones((20, 30, 6))
        ds["temperature"].values[0, 0, 0] = np.nan
        ds["temperature"].values[0, 0, 1] = 0
        path = os.path.join(dir_path, "test_encoding.nc")
        ds.to_netcdf(path)

        ds_load = xr.open_dataset(path)
        assert (np.isnan(ds_load.temperature.values[0, 0, 0]))
        assert (ds_load.temperature.values[0, 0, 1] == 0)
        # os.remove(path)

    def test_to_netcdf_encoding_popencoding(self):
        # create dataset template
        ds = obsarray.create_ds(template_encoding, dim_sizes)
        # populate with example data
        ds["temperature"].values = 293 * np.ones((20, 30, 6))
        ds["temperature"].values[0, 0, 0] = np.nan
        ds["temperature"].values[0, 0, 1] = 0.0
        ds["temperature"].encoding.pop("_FillValue")
        path = os.path.join(dir_path, "test_encoding.nc")
        ds.to_netcdf(path)

        ds_load = xr.open_dataset(path)
        print(ds_load.temperature.values[0, 0], ds_load.temperature.dtype, ds_load.temperature.attrs)
        assert (ds_load.temperature.values[0, 0, 0] == 0)  # here nans are replaces by 0, which is wrong fillvalue (i.e. encoding fillvalue should not be removed)
        assert (ds_load.temperature.values[0, 0, 1] == 0)
        # os.remove(path)

    def test_to_netcdf_popattr(self):
        # create dataset template
        ds = obsarray.create_ds(template, dim_sizes)

        # populate with example data
        ds["temperature"].values = 293 * np.ones((20, 30, 6))
        ds["temperature"].values[0, 0, 0] = np.nan
        ds["temperature"].values[0, 0, 1] = 0
        ds["temperature"].attrs.pop("_FillValue")
        path = os.path.join(dir_path, "test_encoding.nc")
        ds.to_netcdf(path)

        ds_load = xr.open_dataset(path)
        assert (np.isnan(ds_load.temperature.values[0, 0, 0]))
        assert (ds_load.temperature.values[0, 0, 1] == 0)
        # os.remove(path)

    def test_to_netcdf(self):
        # create dataset template
        ds = obsarray.create_ds(template, dim_sizes)

        # populate with example data
        ds["temperature"].values = 293 * np.ones((20, 30, 6))
        ds["temperature"].values[0, 0, 0] = np.nan
        ds["temperature"].values[0, 0, 1] = 0
        path = os.path.join(dir_path, "test_encoding.nc")
        ds.to_netcdf(path)

        ds_load = xr.open_dataset(path)
        assert (np.isnan(ds_load.temperature.values[0, 0, 0]))
        assert (ds_load.temperature.values[0, 0, 1] == 0)
        # os.remove(path)


if __name__ == "__main__":
    unittest.main()
