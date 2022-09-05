"""Use Monte Carlo to propagate uncertainties"""

import numpy as np

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "30/03/2019"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def select_repeated_x(x, u_x, param_fixed, i, repeat_dims, repeat_shape):
    """
    Select one (index i) of multiple repeated entries and return the input quantities and uncertainties for that entry.

    :param x: list of input quantities (usually numpy arrays)
    :type x: list[array]
    :param u_x: list of uncertainties/covariances on input quantities (usually numpy arrays)
    :type u_x: list[array]
    :param param_fixed: when repeat_dims>=0, set to true or false to indicate for each input quantity whether it has repeated measurements that should be split (param_fixed=False) or whether the input is fixed (param fixed=True), defaults to None (no inputs fixed).
    :type param_fixed: list of bools, optional
    :param i: index of the repeated measurement
    :type i: int
    :param repeat_dims: dimension along which the measurements are repeated
    :type repeat_dims: int
    :param repeat_shape: shape of measurements along which to select repeats
    :type repeat_shape: tuple
    :return: list of input quantities, list of uncertainties for single measurement
    :rtype: list[array]. list[array]
    """
    xb = np.zeros(len(x), dtype=object)
    u_xb = np.zeros(len(u_x), dtype=object)
    for j in range(len(x)):
        selected = False
        if param_fixed is not None:
            if param_fixed[j] == True:
                xb[j] = x[j]
                u_xb[j] = u_x[j]
                selected = True
        if not selected:
            index = list(np.ndindex(repeat_shape))[i]
            xb[j] = x[j]
            u_xb[j] = u_x[j]
            for idim in range(len(repeat_dims)):
                repeat_axis = repeat_dims[idim]
                ii = index[idim]
                if len(xb[j].shape) > repeat_axis:
                    sli = tuple(
                        [
                            ii if (idim == repeat_axis) else slice(None)
                            for idim in range(xb[j].ndim)
                        ]
                    )
                    xb[j] = xb[j][sli]
                    u_xb[j] = u_xb[j][sli]
                elif len(xb[j]) > 1:
                    try:
                        xb[j] = xb[j][ii]
                        u_xb[j] = u_xb[j][ii]
                    except:
                        pass

    return xb, u_xb
