from punpy.fiduceo.correlation_forms import bell_relative,triangular_relative
from punpy.fiduceo.measurement_function import MeasurementFunction
from punpy.lpu.lpu_propagation import LPUPropagation
from punpy.mc.mc_propagation import MCPropagation
from ._version import get_versions

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/03/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

__version__ = get_versions()["version"]
del get_versions
