"""
Tests for mc propagation class
"""

import unittest

import comet_maths as cm
import numpy as np
import numpy.testing as npt
from punpy.lpu.lpu_propagation import LPUPropagation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def function(x1, x2):
    return x1**2 - 10 * x2


def Jac_function(x1, x2):
    Jac_x1 = np.diag(2 * x1)
    Jac_x2 = np.diag(-10 * np.ones_like(x2))
    Jac = np.concatenate((Jac_x1, Jac_x2)).T
    return Jac


x1 = np.ones(200) * 10
x2 = np.ones(200) * 30
x1err = np.ones(200)
x2err = 2 * np.ones(200)

xs = np.array([x1, x2])
xerrs = np.array([x1err, x2err])

# below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_uncorr = 802**0.5 * np.ones(200)
yerr_corr = 2**0.5 * np.ones(200)

# below, the higher order Taylor expansion terms have not been taken into account
yerr_uncorr_1order = 800**0.5 * np.ones(200)
yerr_corr_1order = 0 * np.ones(200)


def functionb(x1, x2):
    return 2 * x1 - x2


x1b = np.ones((20, 3)) * 50
x2b = np.ones((20, 3)) * 30
x1errb = np.ones((20, 3))
x2errb = 2 * np.ones((20, 3))

xsb = np.array([x1b, x2b])
xerrsb = np.array([x1errb, x2errb])

yerr_uncorrb = 8**0.5 * np.ones((20, 3))
yerr_corrb = np.zeros((20, 3))


def functionc(x1, x2, x3):
    return x1 + 4 * x2 - 2 * x3


def Jac_functionc(x1, x2, x3):
    Jac_x1 = np.diag(np.ones_like(x2))
    Jac_x2 = np.diag(4 * np.ones_like(x2))
    Jac_x3 = np.diag(-2 * np.ones_like(x2))
    Jac = np.concatenate((Jac_x1, Jac_x2, Jac_x3)).T
    return Jac


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


x1d = np.ones((5, 3, 2)) * 50
x2d = np.ones((5, 3, 2)) * 30
x1errd = np.ones((5, 3, 2))
x2errd = 2 * np.ones((5, 3, 2))

xsd = [x1d, x2d]
xerrsd = [x1errd, x2errd]
corr_d = np.ones(
    (2, 2)
)  # np.array([[1,0.9999999,0.9999999],[0.99999999,1.,0.99999999],[0.9999999,0.9999999,1.]])

yerr_uncorrd = [8**0.5 * np.ones((5, 3, 2)), 8**0.5 * np.ones((5, 3, 2))]
yerr_corrd = [np.zeros((5, 3, 2)), 16**0.5 * np.ones((5, 3, 2))]


x1e = np.array(10.0)
x2e = np.array(30.0)
x1erre = np.array(1.0)
x2erre = np.array(2.0)

xse = np.array([x1e, x2e])
xerrse = np.array([x1erre, x2erre])

# below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_uncorre = 802**0.5 * np.ones(1)
yerr_corre = 2**0.5 * np.ones(1)

# below, the higher order Taylor expansion terms have not been taken into account
yerr_uncorr_1ordere = 800**0.5 * np.ones(1)
yerr_corr_1ordere = 0 * np.ones(1)


class TestLPUPropagation(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_propagate_random_1D(self):
        prop = LPUPropagation()
        Jx = Jac_function(*xs)
        uf, ucorr = prop.propagate_random(function, xs, xerrs, return_corr=True, Jx=Jx)
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1order, rtol=0.01)

        uf, ucorr = prop.propagate_random(
            function, xs, xerrs, corr_x=["rand", None], return_corr=True, Jx=Jx
        )
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1order, rtol=0.01)

        uf, ucorr = prop.propagate_standard(
            function, xs, xerrs, ["rand", "rand"], return_corr=True, Jx=Jx
        )
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1order, rtol=0.01)

        ucov = cm.convert_corr_to_cov(ucorr, uf)
        ucorr2 = cm.convert_cov_to_corr(ucov, uf)
        npt.assert_allclose(ucorr, ucorr2, atol=0.01)

        uf = prop.propagate_random(
            function, xs, xerrs, corr_between=np.ones((2, 2)), Jx=Jx
        )
        npt.assert_allclose(uf, yerr_corr_1order, atol=0.01)

        uf, ucorr = prop.propagate_random(function, xs, xerrs, return_corr=True, Jx=Jx)
        npt.assert_allclose(uf, yerr_uncorr_1order, rtol=0.01)

        uf = prop.propagate_random(
            function, xs, xerrs, corr_between=np.ones((2, 2)), Jx=Jx
        )
        npt.assert_allclose(uf, yerr_corr_1order, atol=0.01)

    def test_propagate_random_0D(self):
        prop = LPUPropagation()
        uf, ucorr = prop.propagate_random(function, xse, xerrse, return_corr=True)
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1ordere, rtol=0.01)

        uf, ucorr = prop.propagate_random(
            function, xse, xerrse, corr_x=["rand", None], return_corr=True
        )
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1ordere, rtol=0.01)

        uf, ucorr = prop.propagate_standard(
            function, xse, xerrse, ["rand", "rand"], return_corr=True
        )
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1ordere, rtol=0.01)

        ucov = cm.convert_corr_to_cov(ucorr, uf)
        ucorr2 = cm.convert_cov_to_corr(ucov, uf)
        npt.assert_allclose(ucorr, ucorr2, atol=0.01)

        uf = prop.propagate_random(function, xse, xerrse, corr_between=np.ones((2, 2)))
        npt.assert_allclose(uf, yerr_corr_1ordere, atol=0.01)

        uf, ucorr = prop.propagate_random(function, xse, xerrse, return_corr=True)
        npt.assert_allclose(uf, yerr_uncorr_1ordere, rtol=0.01)

        uf = prop.propagate_random(function, xse, xerrse, corr_between=np.ones((2, 2)))
        npt.assert_allclose(uf, yerr_corr_1ordere, atol=0.01)

    def test_propagate_random_2D(self):
        prop = LPUPropagation(parallel_cores=2)
        ufb, ucorrb, jac_x = prop.propagate_random(
            functionb,
            xsb,
            xerrsb,
            return_corr=True,
            repeat_dims=1,
            return_Jacobian=True,
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.01)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.01)

        ufb = prop.propagate_random(
            functionb, xsb, xerrsb, corr_between=np.ones((2, 2))
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.03)

        ufb = prop.propagate_random(
            functionb,
            xsb,
            xerrsb,
            corr_between=np.ones((2, 2)),
            repeat_dims=1,
            Jx=jac_x,
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.03)

    def test_propagate_random_1D_3var(self):
        prop = LPUPropagation()
        Jx = Jac_functionc(*xsc)

        ufc, ucorrc = prop.propagate_random(
            functionc, xsc, xerrsc, return_corr=True, Jx=Jx
        )
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.01)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.01)

        ufc = prop.propagate_random(functionc, xsc, xerrsc, corr_between=corr_c, Jx=Jx)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.01)

        ufc = prop.propagate_random(functionc, xsc, xerrsc, corr_between=corr_c, Jx=Jx)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.01)

    def test_propagate_random_3D_2out(self):
        prop = LPUPropagation()
        ufd, ucorrd, corr_out = prop.propagate_random(
            functiond, xsd, xerrsd, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.01)
        npt.assert_allclose(ufd, yerr_uncorrd, rtol=0.01)

        ufd, ucorrd, corr_out = prop.propagate_random(
            functiond, xsd, xerrsd, return_corr=True, output_vars=2, repeat_dims=[1, 2]
        )
        npt.assert_allclose(ufd[0], yerr_uncorrd[0], atol=0.01)
        npt.assert_allclose(ufd[1], yerr_uncorrd[1], atol=0.01)
        npt.assert_allclose(corr_out, np.eye(len(corr_out)), atol=0.01)
        npt.assert_allclose(ucorrd[1], np.eye(len(ucorrd[1])), atol=0.01)

        ufd, ucorrd, corr_out = prop.propagate_random(
            functiond, xsd, xerrsd, corr_between=corr_d, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ufd[0], yerr_corrd[0], atol=0.01)
        npt.assert_allclose(ufd[1], yerr_corrd[1], rtol=0.01)

    def test_propagate_systematic_1D(self):
        prop = LPUPropagation()
        Jx = Jac_function(*xs)
        corr = (np.ones_like(x1err) + np.eye(len(x1err))) / 2
        uf, ucorr = prop.propagate_systematic(
            function,
            xs,
            [x1err, None],
            corr_x=[corr, None],
            return_corr=True,
            fixed_corr_var=True,
            Jx=Jx,
        )
        npt.assert_allclose(ucorr, corr, atol=0.0001)

        uf, ucorr = prop.propagate_systematic(
            function, xs, xerrs, return_corr=True, Jx=Jx
        )
        npt.assert_allclose(ucorr, np.ones_like(ucorr), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1order, atol=0.01)

        uf = prop.propagate_systematic(
            function, xs, xerrs, corr_between=np.ones((2, 2)), Jx=Jx
        )
        npt.assert_allclose(uf, yerr_corr_1order, atol=0.01)

    def test_propagate_systematic_2D(self):
        prop = LPUPropagation()
        ufb, ucorrb, jac_x = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True, return_Jacobian=True
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.01)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.01)

        ufb, ucorrb = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True, repeat_dims=1
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.01)

        ufb, ucorrb = prop.propagate_systematic(
            functionb, xsb, xerrsb, return_corr=True, corr_dims=0
        )
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.01)

        ufb = prop.propagate_systematic(
            functionb, xsb, xerrsb, corr_between=np.ones((2, 2)), Jx=jac_x
        )
        npt.assert_allclose(ufb, yerr_corrb, atol=0.03)

    def test_propagate_systematic_1D_3var(self):
        prop = LPUPropagation()
        Jx = Jac_functionc(*xsc)

        ufc, ucorrc = prop.propagate_systematic(
            functionc, xsc, xerrsc, return_corr=True, Jx=Jx
        )
        npt.assert_allclose(ucorrc, np.ones_like(ucorrc), atol=0.01)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.01)

        ufc = prop.propagate_systematic(
            functionc, xsc, xerrsc, corr_between=corr_c, Jx=Jx
        )
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.01)

    def test_propagate_systematic_3D_2out(self):
        prop = LPUPropagation()
        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=0, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.01)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=1, output_vars=2
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.01)

        ufd, ucorrd, corr_out = prop.propagate_systematic(
            functiond, xsd, xerrsd, return_corr=True, corr_dims=2, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.ones_like(ucorrd[0]), atol=0.01)

    def test_propagate_cov_1D(self):
        prop = LPUPropagation()

        cov = [
            cm.convert_corr_to_cov(np.eye(len(xerr.ravel())), xerr) for xerr in xerrs
        ]

        Jx = Jac_function(*xs)
        uf, ucorr = prop.propagate_cov(function, xs, cov, return_corr=True, Jx=Jx)
        npt.assert_allclose(ucorr, np.eye(len(ucorr)), atol=0.01)
        npt.assert_allclose(uf, yerr_uncorr_1order, rtol=0.01)

        cov = [
            cm.convert_corr_to_cov(
                np.ones((len(xerr.ravel()), len(xerr.ravel()))), xerr
            )
            for xerr in xerrs
        ]
        uf, ucorr = prop.propagate_cov(function, xs, cov, return_corr=True, Jx=Jx)
        npt.assert_allclose(uf, yerr_uncorr_1order, atol=0.01)
        npt.assert_allclose(ucorr, np.ones((len(ucorr), len(ucorr))), atol=0.01)

        cov = [cm.convert_corr_to_cov(np.eye(len(xerr)), xerr) for xerr in xerrs]
        uf, ucorr = prop.propagate_cov(
            function, xs, cov, return_corr=True, corr_between=np.ones((2, 2)), Jx=Jx
        )
        # ucorr is full of nans because the uncertainties are 0
        npt.assert_allclose(uf, yerr_corr_1order, atol=0.01)

    def test_propagate_cov_2D(self):
        prop = LPUPropagation(parallel_cores=2)

        covb = [
            cm.convert_corr_to_cov(np.eye(len(xerrb.ravel())), xerrb)
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_cov(
            functionb, xsb, covb, return_corr=True, Jx_diag=True
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.01)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.01)

        covb = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrb.ravel()), len(xerrb.ravel())))
                + np.eye(len(xerrb.ravel())),
                xerrb,
            )
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_cov(
            functionb, xsb, covb, return_corr=True, Jx_diag=True
        )
        npt.assert_allclose(ufb, yerr_uncorrb * 2**0.5, rtol=0.01)

        covb = [
            cm.convert_corr_to_cov(np.eye(len(xerrb.ravel())), xerrb)
            for xerrb in xerrsb
        ]
        ufb, ucorrb = prop.propagate_cov(
            functionb,
            xsb,
            covb,
            return_corr=True,
            corr_between=np.ones((2, 2)),
            Jx_diag=True,
        )
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.01)
        npt.assert_allclose(ufb, yerr_corrb, atol=0.03)

    def test_propagate_cov_1D_3var(self):
        prop = LPUPropagation()
        Jx = Jac_functionc(*xsc)

        covc = [
            cm.convert_corr_to_cov(np.eye(len(xerrc.ravel())), xerrc)
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(functionc, xsc, covc, return_corr=True, Jx=Jx)
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.01)
        npt.assert_allclose(ufc, yerr_uncorrc, rtol=0.01)

        covc = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrc.ravel()), len(xerrc.ravel()))) + np.eye(len(xerrc)),
                xerrc,
            )
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(functionc, xsc, covc, return_corr=True, Jx=Jx)
        npt.assert_allclose(ufc, yerr_uncorrc * 2**0.5, rtol=0.01)

        covc = [
            cm.convert_corr_to_cov(np.eye(len(xerrc.ravel())), xerrc)
            for xerrc in xerrsc
        ]
        ufc, ucorrc = prop.propagate_cov(
            functionc, xsc, covc, return_corr=True, corr_between=corr_c, Jx=Jx
        )
        npt.assert_allclose(ucorrc, np.eye(len(ucorrc)), atol=0.01)
        npt.assert_allclose(ufc, yerr_corrc, rtol=0.01)

    def test_propagate_cov_3D_2out(self):
        prop = LPUPropagation()

        covd = [
            cm.convert_corr_to_cov(np.eye(len(xerrd.ravel())), xerrd)
            for xerrd in xerrsd
        ]
        ufd, ucorrd, corr_out = prop.propagate_cov(
            functiond, xsd, covd, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.01)
        npt.assert_allclose(ufd, yerr_uncorrd, rtol=0.01)

        covd = [
            cm.convert_corr_to_cov(
                np.ones((len(xerrd.ravel()), len(xerrd.ravel()))), xerrd
            )
            for xerrd in xerrsd
        ]
        ufd, ucorrd, corr_out = prop.propagate_cov(
            functiond, xsd, covd, return_corr=True, output_vars=2
        )
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.01)
        npt.assert_allclose(ufd, yerr_uncorrd, rtol=0.01)

    def test_propagate_syst_corr_2D(self):
        prop = LPUPropagation()

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
        npt.assert_allclose(ucorrb, np.eye(len(ucorrb)), atol=0.01)
        npt.assert_allclose(ucorrb2, np.eye(len(ucorrb2)), atol=0.01)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.01)

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
        npt.assert_allclose(ucorrb, np.ones_like(ucorrb), atol=0.01)
        npt.assert_allclose(ucorrb2, np.ones_like(ucorrb2), atol=0.01)
        npt.assert_allclose(ufb, yerr_uncorrb, rtol=0.01)

    def test_propagate_syst_corr_3D_2out(self):
        prop = LPUPropagation()

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
        npt.assert_allclose(ucorrd[0], np.eye(len(ucorrd[0])), atol=0.01)
        npt.assert_allclose(ufd, yerr_uncorrd, rtol=0.01)

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
        npt.assert_allclose(ucorrd[1], np.ones_like(ucorrd[1]), atol=0.01)
        npt.assert_allclose(ufd, yerr_uncorrd, rtol=0.01)

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

    def test__perform_checks(self):
        prop = LPUPropagation()
        corrc = [
            np.ones((len(xerrc[0].ravel()), len(xerrc[0].ravel()))) for xerrc in xerrsc
        ]
        corrd = [
            np.ones((len(xerrd[0].flatten()), len(xerrd[0].flatten())))
            for xerrd in xerrsd
        ]

        out = prop._perform_checks(
            functionc,
            xsc,
            xerrsc,
            corr_x=corrc,
            repeat_dims=0,
            corr_dims=1,
            output_vars=1,
            fixed_corr_var=None,
            Jx_diag=None,
            param_fixed=None,
        )

        out = prop._perform_checks(
            functiond,
            xsd,
            xerrsd,
            corr_x=corrd,
            repeat_dims=[0, 1],
            corr_dims=2,
            output_vars=2,
            fixed_corr_var=None,
            Jx_diag=None,
            param_fixed=[True, False, False],
        )
        try:
            out = prop._perform_checks(
                functionc,
                xsc,
                xerrsc,
                corr_x=corrc,
                repeat_dims=0,
                corr_dims=0,
                output_vars=1,
                fixed_corr_var=None,
                Jx_diag=None,
                param_fixed=None,
            )
            out = prop._perform_checks(
                functiond,
                xsd,
                xerrsd,
                corr_x=corrd,
                repeat_dims=[0, 1],
                corr_dims=1,
                output_vars=2,
                fixed_corr_var=None,
                Jx_diag=None,
                param_fixed=[True, False, False],
            )
        except:
            print("done")


if __name__ == "__main__":
    unittest.main()
