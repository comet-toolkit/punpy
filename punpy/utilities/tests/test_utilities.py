"""
Tests for mc propagation class
"""

import unittest

import numpy as np
import numpy.testing as npt

import punpy.utilities.utilities as util

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def function(x1, x2):
    return x1 ** 2 - 10 * x2 + 30.0


x1 = np.ones(200) * 10
x2 = np.ones(200) * 30
x1err = np.ones(200)
x2err = 2 * np.ones(200)

xs = [x1, x2]
xerrs = [x1err, x2err]

# below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_uncorr = 802 ** 0.5 * np.ones(200)
yerr_corr = 2 ** 0.5 * np.ones(200)


def functionb(x1, x2):
    return 2 * x1 - x2


x1b = np.ones((20, 30)) * 50
x2b = np.ones((20, 30)) * 30
x1errb = np.ones((20, 30))
x2errb = 2 * np.ones((20, 30))

xsb = np.array([x1b, x2b])
xerrsb = np.array([x1errb, x2errb])

yerr_uncorrb = 8 ** 0.5 * np.ones((20, 30))
yerr_corrb = np.zeros((20, 30))


def functionc(x1, x2, x3):
    return x1 + 4 * x2 - 2 * x3


x1c = np.ones(200) * 10
x2c = np.ones(200) * 10
x3c = np.ones(200) * 10

x1errc = 12 * np.ones(200)
x2errc = 5 * np.ones(200)
x3errc = np.zeros(200)

xsc = np.array([x1c, x2c, x3c])
xerrsc = np.array([x1errc, x2errc, x3errc])
corr_c = np.array([[1, 0.9999999, 0], [0.99999999, 1.0, 0], [0.0, 0.0, 1.0]])
yerr_uncorrc = 544 ** 0.5 * np.ones(200)
yerr_corrc = 1024 ** 0.5 * np.ones(200)


def functiond(x1, x2):
    return 2 * x1 - x2, 2 * x1 + x2


x1d = np.ones((20, 3, 4)) * 50
x2d = np.ones((20, 3, 4)) * 30
x1errd = np.ones((20, 3, 4))
x2errd = 2 * np.ones((20, 3, 4))

xsd = [x1d, x2d]
xerrsd = [x1errd, x2errd]
corr_d = np.ones(
    (2, 2)
)  # np.array([[1,0.9999999,0.9999999],[0.99999999,1.,0.99999999],[0.9999999,0.9999999,1.]])

yerr_uncorrd = [
    np.array(8 ** 0.5 * np.ones((20, 3, 4))),
    np.array(8 ** 0.5 * np.ones((20, 3, 4))),
]
yerr_corrd = [np.zeros((20, 3, 4)), 16 ** 0.5 * np.ones((20, 3, 4))]


class TestUtilities(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_select_repeated_x(self):
        xsb2 = np.array([x1b, 1.0])
        xerrsb2 = np.array([x1errb, 2.0])

        out = util.select_repeated_x(xsb2, xerrsb2, [False, True], 0, [0], x1b.shape)
        npt.assert_allclose(out[0][0], x1b[0], atol=0.06)
        npt.assert_allclose(out[0][1], 1, atol=0.06)
        npt.assert_allclose(out[1][0], x1errb[0], atol=0.06)
        npt.assert_allclose(out[1][1], 2.0, atol=0.06)


if __name__ == "__main__":
    unittest.main()
