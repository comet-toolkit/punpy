"""
Tests for mc propagation class
"""

import unittest

import comet_maths as cm
import numpy as np
import numpy.testing as npt
from punpy.mc.mc_propagation import MCPropagation

np.random.seed(12434)

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def function(x1, x2):
    return x1**2 - 10 * x2 + 30.0


x1 = np.ones(200) * 10
x2 = np.ones(200) * 30
x1err = np.ones(200)
x2err = 2 * np.ones(200)

xs = [x1, x2]
xerrs = [x1err, x2err]

# below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_uncorr = 802**0.5 * np.ones(200)
yerr_corr = 2**0.5 * np.ones(200)


def functionb(x1, x2):
    return 2 * x1 - x2


def functionb_fail(x1, x2):
    zero_or_one = np.random.choice([0, 1], p=[0.1, 0.9])
    with np.errstate(divide="raise"):
        try:
            return 2 * x1 - x2 / zero_or_one
        except:
            return np.nan


def functionb_failnan(x1, x2):
    zero_or_one = np.random.choice([0, 1], p=[0.1, 0.9])
    return 2 * x1 - x2 / zero_or_one


x1b = np.ones((20, 30)) * 50
x2b = np.ones((20, 30)) * 30
x1errb = np.ones((20, 30))
x2errb = 2 * np.ones((20, 30))

xsb = np.array([x1b, x2b])
xerrsb = np.array([x1errb, x2errb])

yerr_uncorrb = 8**0.5 * np.ones((20, 30))
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
yerr_uncorrc = 544**0.5 * np.ones(200)
yerr_corrc = 1024**0.5 * np.ones(200)


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
    np.array(8**0.5 * np.ones((20, 3, 4))),
    np.array(8**0.5 * np.ones((20, 3, 4))),
]
yerr_corrd = [np.zeros((20, 3, 4)), 16**0.5 * np.ones((20, 3, 4))]


def functione(x1, x2):
    return 2 * x1 - x2, np.mean(2 * x1 + x2, axis=0)


class TestMCPropagation(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_propagate_random_1D(self):
        prop = MCPropagation(0, parallel_cores=0)
        uf, ucorr = prop.propagate_random(function, xs, xerrs, return_corr=True)
        assert uf is None
        assert ucorr is None

        prop = MCPropagation(40000, parallel_cores=0)
        uf, ucorr = prop.propagate_random(function, xs, xerrs, return_corr=True)
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.06)
        npt.assert_allclose(uf, yerr_uncorr, rtol=0.06)

        ucov = cm.convert_corr_to_cov(ucorr, uf)
        ucorr2 = cm.convert_cov_to_corr(ucov, uf)
        npt.assert_allclose(ucorr, ucorr2, atol=0.01)

        uf = prop.propagate_random(function, xs, xerrs, corr_between=np.ones((2, 2)))
        npt.assert_allclose(uf, yerr_corr, atol=0.1)

        uf, ucorr, yvalues, xvalues = prop.propagate_random(
            function, xs, xerrs, return_corr=True, return_samples=True
        )

        print(xvalues)

        npt.assert_allclose(uf, yerr_uncorr, rtol=0.06)

        uf, yvalues, xvalues = prop.propagate_random(
            function, xs, xerrs, corr_between=np.ones((2, 2)), return_samples=True
        )
        npt.assert_allclose(uf, yerr_corr, atol=0.1)

        uf, yvalues, xvalues = prop.propagate_standard(
            function,
            xs,
            xerrs,
            corr_x=["rand", "rand"],
            corr_between=np.ones((2, 2)),
            return_samples=True,
        )
        npt.assert_allclose(uf, yerr_corr, atol=0.1)

        uf, yvalues, xvalues = prop.propagate_random(
            function,
            xs,
            xerrs,
            corr_x=["rand", None],
            corr_between=np.ones((2, 2)),
            return_samples=True,
        )
        npt.assert_allclose(uf, yerr_corr, atol=0.1)

        uf, yvalues, xvalues = prop.propagate_random(
            function,
            xs,
            xerrs,
            corr_x=[np.diag(np.ones(xs[0].shape)), None],
            corr_between=np.ones((2, 2)),
            return_samples=True,
        )
        npt.assert_allclose(uf, yerr_corr, atol=0.1)

    def test_propagate_random_2D(self):
        prop = MCPropagation(20000, parallel_cores=4)
        ufb, ucorrb = prop.propagate_random(
            functionb, xsb, xerrsb, return_corr=True, repeat_dims=1
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=2, dtype=np.float32)

        ufb, ucorrb = prop.propagate_random(
            functionb, xsb, xerrsb, corr_x=[None, np.eye(len(xsb[0]))], return_corr=True
        )

        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        ufb, ucorrb = prop.propagate_random(
            functionb,
            xsb,
            xerrsb,
            corr_x=[None, np.eye(len(xsb[0] * xsb[1]))],
            return_corr=True,
        )

        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        ufb, ucorrb = prop.propagate_random(
            functionb, xsb, xerrsb, return_corr=True, corr_dims=1
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        ufb = prop.propagate_random(
            functionb, xsb, xerrsb, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.06)

        ufb, yvaluesb, xvaluesb = prop.propagate_random(
            functionb,
            xsb,
            xerrsb,
            corr_between=np.ones((2, 2)),
            return_samples=True,
            repeat_dims=1,
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.07)

    def test_propagate_random_1D_3var(self):
        prop = MCPropagation(20000, parallel_cores=1)
        ufc, ucorrc = prop.propagate_random(functionc, xsc, xerrsc, return_corr=True)
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.06)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.06)

        ufc = prop.propagate_random(functionc, xsc, xerrsc, corr_between=corr_c)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.06)

        ufc, yvaluesc, xvaluesc = prop.propagate_random(
            functionc, xsc, xerrsc, corr_between=corr_c, return_samples=True
        )
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.06)

    def test_propagate_random_3D_2out(self):
        prop = MCPropagation(20000, parallel_cores=0, verbose=True)
        ufd, ucorrd, corr_out = prop.propagate_random(
            functiond, xsd, xerrsd, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        ufd, ucorrd, corr_out, yvalues, xvalues = prop.propagate_random(
            functiond,
            xsd,
            xerrsd,
            return_corr=True,
            output_vars=2,
            return_samples=True,
            repeat_dims=[2, 1],
        )
        npt.assert_allclose(ucorrd[1], np.eye(len(ucorrd[1])), atol=0.06)

        ufd = prop.propagate_random(
            functiond, xsd, xerrsd, corr_between=corr_d, output_vars=2
        )
        npt.assert_allclose(ufd[0], yerr_corrd[0], atol=0.06)
        npt.assert_allclose(ufd[1], yerr_corrd[1], rtol=0.06)

        ufd, yvaluesd, xvaluesd = prop.propagate_random(
            functiond,
            xsd,
            xerrsd,
            corr_between=corr_d,
            return_samples=True,
            output_vars=2,
        )
        npt.assert_allclose(ufd[0], yerr_corrd[0], atol=0.1)
        npt.assert_allclose(ufd[1], yerr_corrd[1], atol=0.1)

        ufd = prop.propagate_random(
            functiond,
            xsd,
            xerrsd,
            corr_between=corr_d,
            return_samples=False,
            output_vars=2,
            repeat_dims=0,
        )
        npt.assert_allclose(ufd[0], yerr_corrd[0], atol=0.1)
        npt.assert_allclose(ufd[1], yerr_corrd[1], atol=0.1)

    def test_propagate_systematic_2D_corrdict(self):
        prop = MCPropagation(20000)

        corrb = [
            {"0": np.eye(len(xerrb[:, 0].ravel())), "1": np.eye(len(xerrb[0].ravel()))}
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=corrb,
            return_corr=True,
        )

        ufb2, ucorrb2 = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=[{"0": np.eye(len(xerrsb[0])), "1": "syst"}, "syst"],
            return_corr=True,
        )

        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)
        npt.assert_allclose(ufb2, yerr_uncorrb, rtol=0.06)

    def test_propagate_systematic_1D(self):
        prop = MCPropagation(30000)

        corr = (np.ones_like(x1err) + np.eye(len(x1err))) / 2
        uf, ucorr = prop.propagate_systematic(
            function,
            xs,
            [x1err, None],
            corr_x=[corr, None],
            return_corr=True,
            fixed_corr_var=True,
        )
        npt.assert_allclose(ucorr, corr, atol=0.0001)

        uf, ucorr = prop.propagate_systematic(function, xs, xerrs, return_corr=True)
        npt.assert_allclose(ucorr, np.ones_like(ucorr), atol=0.06)
        npt.assert_allclose(uf, yerr_uncorr, rtol=0.06)

        uf = prop.propagate_systematic(
            function, xs, xerrs, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(uf, yerr_corr, rtol=0.06)

    def test_propagate_systematic_2D(self):
        prop = MCPropagation(30000)
        ufb, ucorrb = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        ufb, ucorrb = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True, repeat_dims=1
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.06)

        ufb, ucorrb = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True, corr_dims=0
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.06)

        ufb = prop.propagate_systematic(
            functionb, xsb, xerrsb, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.03)

    def test_propagate_systematic_1D_3var(self):
        prop = MCPropagation(30000)
        ufc, ucorrc = prop.propagate_systematic(
            functionc, xsc, xerrsc, return_corr=True
        )
        npt.assert_allclose(ucorrc, np.ones_like(ucorrc), atol=0.06)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.06)

        ufc = prop.propagate_systematic(functionc, xsc, xerrsc, corr_between=corr_c)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.06)

        xsc2 = [x1c, 10.0, x3c]
        xerrsc2 = [x1errc, 5.0, x3errc]

        ufc, ucorrc = prop.propagate_systematic(
            functionc, xsc2, xerrsc2, return_corr=True, param_fixed=[False, True, False]
        )
        npt.assert_allclose(ucorrc, np.ones_like(ucorrc), atol=0.06)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.06)

        xsc2 = [x1c, 10.0, x3c]
        xerrsc2 = [x1errc, 5.0, x3errc]

        ufc = prop.propagate_systematic(functionc, xsc2, xerrsc2)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.06)

    def test_propagate_systematic_3D_2out(self):
        prop = MCPropagation(30000, parallel_cores=2)
        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=0, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)

        prop = MCPropagation(30000)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=1, output_vars=2
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.06)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=2, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)

    def test_propagate_cov_1D(self):
        prop = MCPropagation(25000)

        cov = [
            cm.convert_corr_to_cov(np.eye(len(xerr.ravel())), xerr) for xerr in xerrs
        ]
        uf, ucorr = prop.propagate_cov(function, xs, cov, return_corr=True)
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.06)
        npt.assert_allclose(uf, yerr_uncorr, rtol=0.06)

        cov = [
            cm.convert_corr_to_cov(
                np.ones((len(xerr.ravel()), len(xerr.ravel()))) + np.eye(len(xerr)),
                xerr,
            )
            for xerr in xerrs
        ]
        uf, ucorr = prop.propagate_cov(function, xs, cov, return_corr=True)
        npt.assert_allclose(uf, yerr_uncorr * 2**0.5, rtol=0.06)

        cov = [
            cm.convert_corr_to_cov(np.eye(len(xerr.ravel())), xerr) for xerr in xerrs
        ]
        uf, ucorr = prop.propagate_cov(
            function, xs, cov, return_corr=True, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.06)
        npt.assert_allclose(uf, yerr_corr, atol=0.15)

    def test_propagate_cov_2D(self):
        prop = MCPropagation(20000)

        covb = [
            cm.convert_corr_to_cov(np.eye(len(xerrb.ravel())), xerrb)
            for xerrb in xerrsb
        ]

        ufb, ucorrb = prop.propagate_cov(functionb, xsb, covb, return_corr=True)
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        # covb = [
        #     cm.convert_corr_to_cov(np.eye(len(xerrb[:, 0].ravel())), xerrb[:, 0])
        #     for xerrb in xerrsb
        # ]
        #
        # ufb, ucorrb = prop.propagate_cov(
        #     functionb, xsb, covb, return_corr=True, repeat_dims=1
        # )
        # npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        # npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        xsb2 = [50, xsb[1]]
        covb = [
            1,
            cm.convert_corr_to_cov(
                np.ones((len(xerrsb[1].ravel()), len(xerrsb[1].ravel()))),
                xerrsb[1],
            ),
        ]

        ufb, ucorrb = prop.propagate_cov(
            functionb,
            xsb2,
            covb,
            return_corr=True,
            param_fixed=[True, False],
        )
        npt.assert_allclose(ucorrb, np.ones((len(ucorrb), len(ucorrb))), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        covb = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrb.ravel()), len(xerrb.ravel())))
                + np.eye(len(xerrb.ravel())),
                xerrb,
            )
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_cov(functionb, xsb, covb, return_corr=True)
        npt.assert_allclose(ufb, yerr_uncorrb * 2**0.5, rtol=0.06)

        covb = [
            cm.convert_corr_to_cov(np.eye(len(xerrb.ravel())), xerrb)
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_cov(
            functionb, xsb, covb, return_corr=True, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_corrb, atol=0.1)

    def test_propagate_cov_1D_3var(self):
        prop = MCPropagation(20000)
        covc = [
            cm.convert_corr_to_cov(np.eye(len(xerrc.ravel())), xerrc)
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(functionc, xsc, covc, return_corr=True)
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.06)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.06)

        covc = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrc.ravel()), len(xerrc.ravel()))) + np.eye(len(xerrc)),
                xerrc,
            )
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(functionc, xsc, covc, return_corr=True)
        npt.assert_allclose(ufc, yerr_uncorrc * 2**0.5, rtol=0.06)

        covc = [
            cm.convert_corr_to_cov(np.eye(len(xerrc.ravel())), xerrc)
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(
            functionc, xsc, covc, return_corr=True, corr_between=corr_c
        )
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.06)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.06)

    def test_propagate_cov_3D_2out(self):
        prop = MCPropagation(20000)

        covd = [
            cm.convert_corr_to_cov(np.eye(len(xerrd.ravel())), xerrd)
            for xerrd in xerrsd
        ]
        ufd, ucorrd, corr_out = prop.propagate_cov(
            functiond, xsd, covd, return_corr=True, corr_dims=0, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        covd = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrd.ravel()), len(xerrd.ravel()))), xerrd
            )
            for xerrd in xerrsd
        ]
        ufd, ucorrd, corr_out = prop.propagate_cov(
            functiond, xsd, covd, return_corr=True, corr_dims=0, output_vars=2
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

    def test_propagate_syst_corr_2D(self):
        prop = MCPropagation(20000)

        corrb = [np.eye(len(xerrb[0].ravel())) for xerrb in xerrsb]
        ufb, ucorrb = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=corrb,
            repeat_dims=0,
            corr_dims=1,
            return_corr=True,
        )
        ufb, ucorrb2 = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=["rand", "rand"],
            repeat_dims=0,
            corr_dims=1,
            return_corr=True,
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ucorrb2, np.eye(len(ucorrb2)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        corrb = [
            np.ones((len(xerrb[0].ravel()), len(xerrb[0].ravel()))) for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=corrb,
            repeat_dims=0,
            corr_dims=1,
            return_corr=True,
        )
        ufb, ucorrb2 = prop.propagate_systematic(
            functionb,
            xsb,
            xerrsb,
            corr_x=["syst", None],
            repeat_dims=0,
            corr_dims=1,
            return_corr=True,
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.06)
        npt.assert_allclose(ucorrb2, np.ones_like(ucorrb2), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

    def test_separate_corr_dims(self):
        prop = MCPropagation(20000, parallel_cores=1)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functione,
            xsd,
            xerrsd,
            return_corr=True,
            corr_dims=[-99, 1],
            separate_corr_dims=True,
            output_vars=2,
            param_fixed=[False, True],
        )
        npt.assert_allclose(ucorrd[0], np.ones((240, 240)), atol=0.06)
        npt.assert_allclose(ucorrd[1], np.ones((4, 4)), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=1)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functione,
            xsd,
            xerrsd,
            return_corr=True,
            corr_dims=[None, 1],
            separate_corr_dims=True,
            output_vars=2,
            param_fixed=[False, True],
        )
        npt.assert_equal(ucorrd[0], [None])
        npt.assert_allclose(ucorrd[1], np.ones((4, 4)), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)

    def test_propagate_syst_corr_3D_2out(self):
        prop = MCPropagation(20000, parallel_cores=1)

        corrd = [np.eye(len(xerrd[:, 0, 0].ravel())) for xerrd in xerrsd]
        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond,
            xsd,
            xerrsd,
            corr_x=corrd,
            return_corr=True,
            corr_dims=0,
            repeat_dims=[1, 2],
            output_vars=2,
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=1)

        corrd = [np.eye(len(xerrd[:, 0, 0].ravel())) for xerrd in xerrsd]
        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond,
            xsd,
            xerrsd,
            corr_x=corrd,
            return_corr=True,
            corr_dims=0,
            repeat_dims=1,
            output_vars=2,
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        xsd2 = [x1d, 1.0]
        xerrsd2 = [x1errd, 2.0]

        prop = MCPropagation(20000, parallel_cores=1)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functione,
            xsd2,
            xerrsd2,
            return_corr=True,
            corr_dims=0,
            output_vars=2,
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=3)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functione,
            xsd2,
            xerrsd2,
            return_corr=True,
            corr_dims=0,
            output_vars=2,
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functione,
            xsd2,
            xerrsd2,
            return_corr=True,
            corr_dims=0,
            output_vars=2,
            repeat_dims=1,
            param_fixed=[False, True],
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=1, dtype=np.float32)

        corrd = [
            np.ones((len(xerrd[:, 0, 0].ravel()), len(xerrd[:, 0, 0].ravel())))
            for xerrd in xerrsd
        ]
        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond,
            xsd,
            xerrsd,
            corr_x=corrd,
            return_corr=True,
            corr_dims=0,
            repeat_dims=[1, 2],
            output_vars=2,
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        print(ufd[0][0, 0, 0].dtype)
        self.assertIsInstance(ufd[0][0, 0, 0], np.float32)

        ufd, ucorrd = prop.propagate_systematic(
            functionb,
            xsd,
            xerrsd,
            corr_x=corrd,
            return_corr=True,
            corr_dims=0,
            repeat_dims=[1, 2],
            output_vars=1,
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.01)

    def test_separate_processing(self):
        prop = MCPropagation(20000, parallel_cores=0)

        corrd = [np.eye(len(xerrd.ravel())) for xerrd in xerrsd]

        MC_x = prop.generate_MC_sample(xsd, xerrsd, corrd)
        MC_y1 = prop.run_samples(functiond, MC_x, output_vars=2, start=0, end=10000)
        MC_y2 = prop.run_samples(functiond, MC_x, output_vars=2, start=10000, end=20000)
        MC_y = prop.combine_samples([MC_y1, MC_y2])

        ufd, ucorrd, corr_out = prop.process_samples(
            MC_x, MC_y, return_corr=True, corr_dims=0, output_vars=2
        )

        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=0)

        corrd = [np.eye(len(xerrd[:, 0, 0].ravel())) for xerrd in xerrsd]

        MC_x = prop.generate_MC_sample(xsd, xerrsd, corr_x=["rand", "rand"])
        MC_y1 = prop.run_samples(functiond, MC_x, output_vars=2, start=0, end=10000)
        MC_y2 = prop.run_samples(functiond, MC_x, output_vars=2, start=10000, end=20000)
        MC_y = prop.combine_samples([MC_y1, MC_y2])

        ufd, ucorrd, corr_out = prop.process_samples(
            MC_x, MC_y, return_corr=True, corr_dims=0, output_vars=2
        )

        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

        prop = MCPropagation(20000, parallel_cores=4)

        corrd = [np.eye(len(xerrd[:, 0, 0].ravel())) for xerrd in xerrsd]

        MC_x = prop.generate_MC_sample(xsd, xerrsd, corr_x=["rand", "rand"])
        MC_y1 = prop.run_samples(functiond, MC_x, output_vars=2, start=0, end=10000)
        MC_y2 = prop.run_samples(functiond, MC_x, output_vars=2, start=10000, end=20000)
        MC_y = prop.combine_samples([MC_y1, MC_y2])

        ufd, ucorrd, corr_out = prop.process_samples(
            MC_x,
            MC_y,
            return_corr=True,
            corr_dims=[0, 0],
            output_vars=2,
            separate_corr_dims=True,
        )

        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

    def test_separate_corrdims(self):

        prop = MCPropagation(30000)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond,
            xsd,
            xerrsd,
            return_corr=True,
            corr_dims=[0, 1],
            output_vars=2,
            separate_corr_dims=True,
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.06)
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.06)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond,
            xsd,
            xerrsd,
            return_corr=True,
            corr_dims=[1, 1],
            output_vars=2,
            separate_corr_dims=True,
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.06)

        prop = MCPropagation(20000, parallel_cores=4)

        corrd = [np.eye(len(xerrd[:, 0, 0].ravel())) for xerrd in xerrsd]

        MC_x = prop.generate_MC_sample(xsd, xerrsd, corr_x=["rand", "rand"])
        MC_y1 = prop.run_samples(functiond, MC_x, output_vars=2, start=0, end=10000)
        MC_y2 = prop.run_samples(functiond, MC_x, output_vars=2, start=10000, end=20000)
        MC_y = prop.combine_samples([MC_y1, MC_y2])

        ufd, ucorrd, corr_out = prop.process_samples(
            MC_x,
            MC_y,
            return_corr=True,
            corr_dims=[0, 0],
            output_vars=2,
            separate_corr_dims=True,
        )

        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.06)
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], rtol=0.06)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], rtol=0.06)

    def test_failed_processing(self):
        prop = MCPropagation(10000, parallel_cores=4, verbose=True)
        ufb, ucorrb = prop.propagate_random(
            functionb_fail, xsb, xerrsb, return_corr=True
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

        prop = MCPropagation(10000, parallel_cores=4, verbose=True)
        ufb, ucorrb = prop.propagate_random(
            functionb_failnan, xsb, xerrsb, return_corr=True
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.06)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.06)

    def test__perform_checks(self):
        prop = MCPropagation(20000)
        corrc = [
            np.ones((len(xerrc[0].ravel()), len(xerrc[0].ravel()))) for xerrc in xerrsc
        ]
        corrd = [
            np.ones((len(xerrd[0].ravel()), len(xerrd[0].ravel()))) for xerrd in xerrsd
        ]

        out = prop._perform_checks(
            functionc,
            xsc,
            xerrsc,
            corr_x=corrc,
            repeat_dims=0,
            corr_dims=1,
            separate_corr_dims=False,
            output_vars=1,
            fixed_corr_var=None,
            param_fixed=None,
            refyvar=0,
        )

        out = prop._perform_checks(
            functiond,
            xsd,
            xerrsd,
            corr_x=corrd,
            repeat_dims=[0, 1],
            corr_dims=2,
            separate_corr_dims=False,
            output_vars=2,
            fixed_corr_var=None,
            param_fixed=[True, False, False],
            refyvar=0,
        )
        try:
            out = prop._perform_checks(
                functionc,
                xsc,
                xerrsc,
                corr_x=corrc,
                repeat_dims=0,
                corr_dims=0,
                separate_corr_dims=False,
                output_vars=1,
                fixed_corr_var=None,
                param_fixed=None,
                refyvar=0,
            )
            out = prop._perform_checks(
                functiond,
                xsd,
                xerrsd,
                corr_x=corrd,
                repeat_dims=[0, 1],
                corr_dims=1,
                separate_corr_dims=False,
                output_vars=2,
                fixed_corr_var=None,
                param_fixed=[True, False, False],
                refyvar=0,
            )
        except:
            print("done")


if __name__ == "__main__":
    unittest.main()
