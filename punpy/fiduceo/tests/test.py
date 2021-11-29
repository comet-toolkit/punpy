# Define your measurement function inside a subclass of MeasurementFunction
class GasLaw(MeasurementFunction):
    def function(self, pres, temp):
        return pres/(temp*8.134)

# create class object and pass all optional keywords for punpy
gl = GasLaw(["pressure","temperature"],steps=100000)  

# propagate the uncertainties on the input quantities in ds to measurand uncertainties in ds_y
ds_y=gl.propagate_ds("V/n",ds) 

