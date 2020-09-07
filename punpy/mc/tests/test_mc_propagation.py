"""
Tests for mc propagation class
"""

import unittest
import numpy as np
import numpy.testing as npt
import punpy
from punpy.mc.mc_propagation import MCPropagation

'''___Authorship___'''
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__version__ = punpy.__version__
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

def function(x1,x2):
    return x1**2 - 10*x2

x1=np.ones(200)*10
x2=np.ones(200)*30
x1err=np.ones(200)
x2err=2*np.ones(200)

xs=np.array([x1,x2])
xerrs=np.array([x1err,x2err])

#below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_uncorr=802**0.5*np.ones(200)
yerr_corr=2**0.5*np.ones(200)

def functionb(x1,x2):
    return 2* x1 - x2

x1b=np.ones((20,3))*50
x2b=np.ones((20,3))*30
x1errb=np.ones((20,3))
x2errb=2*np.ones((20,3))

xsb=np.array([x1b,x2b])
xerrsb=np.array([x1errb,x2errb])

yerr_uncorrb=8**0.5*np.ones((20,3))
yerr_corrb=np.zeros((20,3))

def functionc(x1,x2,x3):
    return x1  +4*x2 -2*x3

x1c=np.ones(200)*10
x2c=np.ones(200)*10
x3c=np.ones(200)*10

x1errc=12*np.ones(200)
x2errc=5*np.ones(200)
x3errc=np.zeros(200)

xsc=np.array([x1c,x2c,x3c])
xerrsc=np.array([x1errc,x2errc,x3errc])
corr_c=np.array([[1,0.9999999,0],[0.99999999,1.,0],[0.,0.,1.]])
yerr_uncorrc=544**0.5*np.ones(200)
yerr_corrc=1024**0.5*np.ones(200)

def functiond(x1,x2):
    return 2* x1 - x2, 2*x1+x2

x1d=np.ones((20,3,4))*50
x2d=np.ones((20,3,4))*30
x1errd=np.ones((20,3,4))
x2errd=2*np.ones((20,3,4))

xsd=np.array([x1d,x2d])
xerrsd=np.array([x1errd,x2errd])
corr_d=np.ones((2,2))#np.array([[1,0.9999999,0.9999999],[0.99999999,1.,0.99999999],[0.9999999,0.9999999,1.]])

yerr_uncorrd=[8**0.5*np.ones((20,3,4)),8**0.5*np.ones((20,3,4))]
yerr_corrd=[np.zeros((20,3,4)),16**0.5*np.ones((20,3,4))]


class TestMCPropagation(unittest.TestCase):
    """
    Class for unit tests
    """
    def test_propagate_random(self):
        prop = MCPropagation(40000)

        uf,ucorr = prop.propagate_random(function,xs,xerrs,return_corr=True)
        npt.assert_allclose(ucorr,np.eye(len(ucorr)),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)
        ucov= prop.convert_corr_to_cov(ucorr,uf)
        ucorr2= prop.convert_cov_to_corr(ucov,uf)
        npt.assert_allclose(ucorr,ucorr2,atol=0.01)

        uf = prop.propagate_random(function,xs,xerrs,corr_between=np.ones((2,2)))
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        uf,ucorr,yvalues,xvalues = prop.propagate_random(function,xs,xerrs,return_corr=True,return_samples=True)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf,yvalues,xvalues = prop.propagate_random(function,xs,xerrs,corr_between=np.ones((2,2)),return_samples=True)
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        #b
        ufb,ucorrb = prop.propagate_random(functionb,xsb,xerrsb,return_corr=True)
        npt.assert_allclose(ucorrb,np.eye(len(ucorrb)),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb = prop.propagate_random(functionb,xsb,xerrsb,corr_between=np.ones((2,2)))
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)

        ufb,yvaluesb,xvaluesb = prop.propagate_random(functionb,xsb,xerrsb,corr_between=np.ones((2,2)),
                                                   return_samples=True)
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)

        #c
        ufc,ucorrc = prop.propagate_random(functionc,xsc,xerrsc,return_corr=True)
        npt.assert_allclose(ucorrc,np.eye(len(ucorrc)),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc = prop.propagate_random(functionc,xsc,xerrsc,corr_between=corr_c)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

        ufc,yvaluesc,xvaluesc = prop.propagate_random(functionc,xsc,xerrsc,corr_between=corr_c,return_samples=True)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

        #d
        ufd,ucorrd,corr_out = prop.propagate_random(functiond,xsd,xerrsd,return_corr=True,output_vars=2)
        npt.assert_allclose(ucorrd[0],np.eye(len(ucorrd[0])),atol=0.05)
        npt.assert_allclose(ufd,yerr_uncorrd,rtol=0.05)

        ufd,ucorrd,corr_out,yvalues,xvalues = prop.propagate_random(functiond,xsd,xerrsd,return_corr=True,output_vars=2,return_samples=True)
        npt.assert_allclose(ucorrd[1],np.eye(len(ucorrd[1])),atol=0.05)

        ufd = prop.propagate_random(functiond,xsd,xerrsd,corr_between=corr_d,output_vars=2)
        npt.assert_allclose(ufd[0],yerr_corrd[0],atol=0.05)
        npt.assert_allclose(ufd[1],yerr_corrd[1],rtol=0.05)

        ufd,yvaluesd,xvaluesd = prop.propagate_random(functiond,xsd,xerrsd,corr_between=corr_d,return_samples=True,output_vars=2)
        npt.assert_allclose(ufd,yerr_corrd,atol=0.05)

    def test_propagate_systematic(self):
        prop = MCPropagation(30000,parallel_cores=3)

        uf,ucorr = prop.propagate_systematic(function,xs,xerrs,return_corr=True)
        npt.assert_allclose(ucorr,np.ones_like(ucorr),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf = prop.propagate_random(function,xs,xerrs,
                                   corr_between=np.ones((2,2)))
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        #b
        ufb,ucorrb = prop.propagate_systematic(functionb,xsb,xerrsb,return_corr=True)
        npt.assert_allclose(ucorrb,np.ones_like(ucorrb),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb,ucorrb = prop.propagate_systematic(functionb,xsb,xerrsb,return_corr=True,corr_axis=0)
        npt.assert_allclose(ucorrb,np.ones_like(ucorrb),atol=0.05)

        ufb,ucorrb = prop.propagate_systematic(functionb,xsb,xerrsb,return_corr=True,corr_axis=1)
        npt.assert_allclose(ucorrb,np.ones_like(ucorrb),atol=0.05)

        ufb = prop.propagate_systematic(functionb,xsb,xerrsb,corr_between=np.ones((2,2)))
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)

        #c
        ufc,ucorrc = prop.propagate_systematic(functionc,xsc,xerrsc,return_corr=True)
        npt.assert_allclose(ucorrc,np.ones_like(ucorrc),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc = prop.propagate_random(functionc,xsc,xerrsc,corr_between=corr_c)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

        ufd,ucorrd,corr_out = prop.propagate_systematic(functiond,xsd,xerrsd,return_corr=True,corr_axis=0,
                                                        output_vars=2)
        npt.assert_allclose(ucorrd[0],np.ones_like(ucorrd[0]),atol=0.05)

        ufd,ucorrd,corr_out = prop.propagate_systematic(functiond,xsd,xerrsd,return_corr=True,corr_axis=1,
                                                        output_vars=2)
        npt.assert_allclose(ucorrd[1],np.ones_like(ucorrd[1]),atol=0.05)

        ufd,ucorrd,corr_out = prop.propagate_systematic(functiond,xsd,xerrsd,return_corr=True,corr_axis=2,
                                                        output_vars=2)
        npt.assert_allclose(ucorrd[0],np.ones_like(ucorrd[0]),atol=0.05)


    def test_propagate_both(self):
        prop = MCPropagation(20000,parallel_cores=1)

        uf,ucorr = prop.propagate_both(function,xs,xerrs,
                                       [np.zeros_like(x1err),np.zeros_like(x2err)],return_corr=True)
        npt.assert_allclose(ucorr,np.eye(len(ucorr)),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf,ucorr = prop.propagate_both(function,xs,[np.zeros_like(x1err),np.zeros_like(x2err)],
                                       xerrs,return_corr=True)
        npt.assert_allclose(ucorr,np.ones_like(ucorr),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf = prop.propagate_both(function,xs,xerrs,[np.zeros_like(x1err),np.zeros_like(x2err)],
                                 return_corr=False,corr_between=np.ones((2,2)))
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        #b
        ufb,ucorrb = prop.propagate_both(functionb,xsb,xerrsb,
                                       [np.zeros_like(x1errb),np.zeros_like(x2errb)],return_corr=True)
        npt.assert_allclose(ucorrb,np.eye(len(ucorrb)),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb,ucorrb = prop.propagate_both(functionb,xsb,[np.zeros_like(x1errb),np.zeros_like(x2errb)],
                                       xerrsb,return_corr=True)
        npt.assert_allclose(ucorrb,np.ones_like(ucorrb),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb = prop.propagate_both(functionb,xsb,xerrsb,[np.zeros_like(x1errb),np.zeros_like(x2errb)],
                                 return_corr=False,corr_between=np.ones((2,2)))
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)

        #c
        ufc,ucorrc = prop.propagate_both(functionc,xsc,xerrsc,
                                       [np.zeros_like(x1c),np.zeros_like(x2c),np.zeros_like(x3c)],return_corr=True)
        npt.assert_allclose(ucorrc,np.eye(len(ucorrc)),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc,ucorrc = prop.propagate_both(functionc,xsc,[np.zeros_like(x1c),np.zeros_like(x2c),np.zeros_like(x3c)],
                                       xerrsc,return_corr=True)
        npt.assert_allclose(ucorrc,np.ones_like(ucorrc),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc = prop.propagate_both(functionc,xsc,xerrsc,[np.zeros_like(x1c),np.zeros_like(x2c),np.zeros_like(x3c)],return_corr=False,corr_between=corr_c)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

    def test_propagate_type(self):
        prop = MCPropagation(20000)
        uf,ucorr = prop.propagate_type(function,xs,xerrs,['rand','rand'],return_corr=True)
        npt.assert_allclose(ucorr,np.eye(len(ucorr)),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf,ucorr = prop.propagate_type(function,xs,xerrs,['syst','syst'],return_corr=True)

        npt.assert_allclose(ucorr,np.ones_like(ucorr),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        uf = prop.propagate_type(function,xs,xerrs,['rand','rand'],return_corr=False,corr_between=np.ones((2,2)))
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        #b
        ufb,ucorrb = prop.propagate_type(functionb,xsb,xerrsb,['rand','rand'],return_corr=True)
        npt.assert_allclose(ucorrb,np.eye(len(ucorrb)),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb,ucorrb = prop.propagate_type(functionb,xsb,xerrsb,['syst','syst'],return_corr=True)

        npt.assert_allclose(ucorrb,np.ones_like(ucorrb),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        ufb = prop.propagate_type(functionb,xsb,xerrsb,['rand','rand'],return_corr=False,corr_between=np.ones((2,2)))
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)


        ufc,ucorrc = prop.propagate_type(functionc,xsc,xerrsc,['rand','rand','rand'],return_corr=True)
        npt.assert_allclose(ucorrc,np.eye(len(ucorrc)),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc,ucorrc = prop.propagate_type(functionc,xsc,xerrsc,['syst','syst','syst'],return_corr=True)

        npt.assert_allclose(ucorrc,np.ones_like(ucorrc),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        ufc = prop.propagate_type(functionc,xsc,xerrsc,['rand','rand','rand'],return_corr=False,corr_between=corr_c)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

    def test_propagate_cov(self):
        prop = MCPropagation(20000)

        cov = [MCPropagation.convert_corr_to_cov(np.eye(len(xerr.flatten())),xerr) for xerr in xerrs]
        uf,ucorr = prop.propagate_cov(function,xs,cov,return_corr=True)
        npt.assert_allclose(ucorr,np.eye(len(ucorr)),atol=0.05)
        npt.assert_allclose(uf,yerr_uncorr,rtol=0.05)

        cov = [MCPropagation.convert_corr_to_cov(np.ones((len(xerr.flatten()),len(xerr.flatten())))+np.eye(len(xerr)),xerr) for xerr in xerrs]
        uf,ucorr = prop.propagate_cov(function,xs,cov,return_corr=True)
        npt.assert_allclose(uf,yerr_uncorr*2**0.5,rtol=0.05)

        cov = [MCPropagation.convert_corr_to_cov(np.eye(len(xerr.flatten())),xerr) for xerr in xerrs]
        uf,ucorr = prop.propagate_cov(function,xs,cov,return_corr=True,corr_between=np.ones((2,2)))
        npt.assert_allclose(ucorr,np.eye(len(ucorr)),atol=0.05)
        npt.assert_allclose(uf,yerr_corr,rtol=0.05)

        #b
        covb = [MCPropagation.convert_corr_to_cov(np.eye(len(xerrb.flatten())),xerrb) for xerrb in xerrsb]
        ufb,ucorrb = prop.propagate_cov(functionb,xsb,covb,return_corr=True)
        npt.assert_allclose(ucorrb,np.eye(len(ucorrb)),atol=0.05)
        npt.assert_allclose(ufb,yerr_uncorrb,rtol=0.05)

        covb = [MCPropagation.convert_corr_to_cov(np.ones((len(xerrb.flatten()),len(xerrb.flatten())))+np.eye(len(xerrb.flatten())),xerrb) for xerrb in
               xerrsb]
        ufb,ucorrb = prop.propagate_cov(functionb,xsb,covb,return_corr=True)
        npt.assert_allclose(ufb,yerr_uncorrb*2**0.5,rtol=0.05)

        covb = [MCPropagation.convert_corr_to_cov(np.eye(len(xerrb.flatten())),xerrb) for xerrb in xerrsb]
        ufb,ucorrb = prop.propagate_cov(functionb,xsb,covb,return_corr=True,corr_between=np.ones((2,2)))
        npt.assert_allclose(ucorrb,np.eye(len(ucorrb)),atol=0.05)
        npt.assert_allclose(ufb,yerr_corrb,atol=0.03)

        #c
        covc = [MCPropagation.convert_corr_to_cov(np.eye(len(xerrc.flatten())),xerrc) for xerrc in xerrsc]
        ufc,ucorrc = prop.propagate_cov(functionc,xsc,covc,return_corr=True)
        npt.assert_allclose(ucorrc,np.eye(len(ucorrc)),atol=0.05)
        npt.assert_allclose(ufc,yerr_uncorrc,rtol=0.05)

        covc = [MCPropagation.convert_corr_to_cov(np.ones((len(xerrc.flatten()),len(xerrc.flatten())))+np.eye(len(xerrc)),xerrc) for xerrc in
                xerrsc]
        ufc,ucorrc = prop.propagate_cov(functionc,xsc,covc,return_corr=True)
        npt.assert_allclose(ufc,yerr_uncorrc*2**0.5,rtol=0.05)

        covc = [MCPropagation.convert_corr_to_cov(np.eye(len(xerrc.flatten())),xerrc) for xerrc in xerrsc]
        ufc,ucorrc = prop.propagate_cov(functionc,xsc,covc,return_corr=True,corr_between=corr_c)
        npt.assert_allclose(ucorrc,np.eye(len(ucorrc)),atol=0.05)
        npt.assert_allclose(ufc,yerr_corrc,rtol=0.05)

if __name__ == '__main__':
    unittest.main()
