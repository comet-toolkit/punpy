import obsarray
import xarray as xr
from variables import L0_IRR_VARIABLES, CAL_VARIABLES
import numpy as np

ds_in = xr.open_dataset("test_l0.nc")  # read digital effects table
ds_cal = xr.open_dataset("test_cal.nc")  # read digital effects table

#relative uncertainties should be in percent
for var in ds_in.variables:
    if "u_rel_" in var:
        ds_in[var].values=ds_in[var].values*100
        #ds_in[var].attrs["units"]=="%"

for var in ds_cal.variables:
    if "u_rel_" in var and "units" in ds_cal[var].attrs and ds_cal[var].attrs["units"]=="-":
        ds_cal[var].values=ds_cal[var].values*100
        ds_cal[var].attrs["units"]=="%"

# define dim_size_dict to specify size of arrays
dim_sizes = {
    "wavelength": len(ds_in.wavelength),
    "scan": len(ds_in.scan),
}

dim_sizes_cal = {
    "wavelength": len(ds_cal.wavelength),
    "calibrationdates": 1,
    "nonlinearcoef": 13,
    "nonlineardates": 1,
    "wavcoef": 5,
    "wavdates": 1,
}

# create dataset template
ds_out = obsarray.create_ds(L0_IRR_VARIABLES, dim_sizes)
ds_cal_out = obsarray.create_ds(CAL_VARIABLES, dim_sizes_cal)

print(ds_out.digital_number)

for key in ds_out.keys():
    ds_out[key].values = ds_in[key].values

ds_out.assign_coords(wavelength=ds_in.wavelength)
ds_out.assign_coords(scan=ds_in.scan)

for key in ds_cal_out.keys():
    ds_cal_out[key].values = ds_cal[key].values.reshape(ds_cal_out[key].shape)

# store example file
ds_out.to_netcdf("det_hypernets_l0.nc")
ds_cal_out.to_netcdf("det_hypernets_cal.nc")
