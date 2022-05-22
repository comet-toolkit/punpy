
import xarray as xr

from punpy import MeasurementFunction

ds = xr.open_dataset("digital_effects_table_gaslaw_example.nc")  # read digital effects table

# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw(MeasurementFunction):
    def meas_function(self, pres, temp):
        return (temp * 8.134)/pres


# create class object and pass all optional keywords for punpy
gl = IdealGasLaw(["pressure", "temperature"], "MC", steps=100000)

# propagate the uncertainties on the input quantities in ds to measurand uncertainties in ds_y
ds_y = gl.propagate_ds("Volume", ds)
print(ds_y)