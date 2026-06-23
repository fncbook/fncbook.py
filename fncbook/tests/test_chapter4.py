import pytest
import fncbook as FNC  # The code to test
from numpy import isclose
import numpy as np


def test_newton():
    for c in [2,4,7.5,11]:
        f = lambda x: np.exp(x) - x - c
        dfdx = lambda x: np.exp(x) - 1
        x = FNC.newton(f,dfdx,1.0)
        r = x[-1]
        assert isclose(f(r), 0)

def test_secant():
     for c in [2,4,7.5,11]:
        f = lambda x: np.exp(x) - x - c
        x = FNC.secant(f, 3, 0.5)
        r = x[-1]
        assert isclose(f(r), 0)

def test_systems():
    def nlfun(x):
        f = np.array([     
            np.exp(x[1]-x[0]) - 2,
            x[0]*x[1] + x[2],
            x[1]*x[2] + x[0]**2 - x[1]
        ])
        return f
    def nljac(x):
        J = np.zeros((3,3))
        J[0] = [-np.exp(x[1]-x[0]), np.exp(x[1]-x[0]), 0]
        J[1] = [x[1], x[0], 1]
        J[2] = [2*x[0], x[2]-1, x[1]]
        return J
    x = FNC.newtonsys(nlfun,nljac,np.array([0.,0,0]))
    assert isclose(nlfun(x[-1]), [0, 0, 0]).all()
    x = FNC.newtonsys(nlfun,nljac,np.array([1.,2,3]))
    assert isclose(nlfun(x[-1]), [0, 0, 0]).all()
    x = FNC.levenberg(nlfun, np.array([10.,-4,-3]))
    assert isclose(nlfun(x[-1]), [0, 0, 0]).all()

def test_nlsq():
    t = np.linspace(-1, 1, 40)
    y = np.sin(np.exp(3 * t))
    def nlfun(x):
        f = x[0] * t**0 + x[1] * t + x[2] * np.sin(t)- y
        return f
    x = FNC.levenberg(nlfun, np.array([1., 1, 1]))
    assert isclose(x[-1], [0.25941633, 2.9946118, -3.417461]).all()

def test_newton_fail():
    f = lambda x: np.exp(x) - x - 2
    dfdx = lambda x: x    # wrong derivative to force failure
    with pytest.warns(UserWarning):
        FNC.newton(f, dfdx, 1.0)
    

def test_secant_fail():
    f = lambda x: np.sqrt(np.abs(x))
    with pytest.warns(UserWarning):
        x = FNC.secant(f, 2, 1)