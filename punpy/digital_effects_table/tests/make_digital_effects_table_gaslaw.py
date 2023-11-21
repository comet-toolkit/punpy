import numpy as np
import obsarray
import os

# define ds variables
template = {
    "temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "unc_comps": ["u_ran_temperature","u_sys_temperature"]
        }
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
    },
    "u_sys_temperature": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "K",
            "err_corr": [
              {
                  "dim": "x",
                  "form": "systematic",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "y",
                  "form": "systematic",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "time",
                  "form": "systematic",
                  "params": [],
                  "units": []
              }
          ]
        }
    },
    "pressure": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "Pa",
            "unc_comps": ["u_str_pressure"]
        }
    },
    "u_str_pressure": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "Pa",
            "err_corr": [
              {
                  "dim": "x",
                  "form": "random",
                  "params": [],
                  "units": []
              },
              {
                  "dim": "y",
                  "form": "err_corr_matrix",
                  "params": "err_corr_str_pressure_y",
                  "units": []
              },
              {
                  "dim": "time",
                  "form": "systematic",
                  "params": [],
                  "units": []
              }
          ]
        },
    },
    "err_corr_str_pressure_y": {
        "dtype": np.float32,
        "dim": ["y", "y"],
        "attributes": {"units": ""},
    },
    "n_moles": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "",
            "unc_comps": ["u_ran_n_moles"]
        }
    },
    "u_ran_n_moles": {
        "dtype": np.float32,
        "dim": ["x", "y", "time"],
        "attributes": {
            "units": "",
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
    },
}

# define dim_size_dict to specify size of arrays
dim_sizes = {
    "x": 20,
    "y": 30,
    "time": 6
}

# create dataset template
ds = obsarray.create_ds(template, dim_sizes)

# populate with example data
ds["temperature"].values = 293*np.ones((20,30,6))
ds["u_ran_temperature"].values = 1*np.ones((20,30,6))
ds["u_sys_temperature"].values = 0.4*np.ones((20,30,6))
ds["pressure"].values = 10**5*np.ones((20,30,6))
ds["u_str_pressure"].values = 10*np.ones((20,30,6))
ds["err_corr_str_pressure_y"].values = 0.5*np.ones((30,30))+0.5*np.eye(30)
ds["n_moles"].values = 40*np.ones((20,30,6))
ds["u_ran_n_moles"].values = 1*np.ones((20,30,6))

# store example file
dir_path = os.path.dirname(os.path.realpath(__file__))
ds.to_netcdf(os.path.join(dir_path, "digital_effects_table_gaslaw_example.nc"))