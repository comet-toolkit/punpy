"""Sample unit tests"""

import pytest


def test_pass():
    assert True, "dummy sample test"


@pytest.mark.slow
def test_slow():
    print("This is a very slow test which will sometimes be skipped (see the readme)")
    assert True
