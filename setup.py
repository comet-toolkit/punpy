import io
import os
import re
from distutils.core import setup

from setuptools import find_packages


# from Cython.Build import cythonize


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())

# extensions = [Extension(
#         name="utilities",
#         sources=["punpy/utilities/utilities.pyx"],
#         include_dirs=[numpy.get_include()],
#         )
#     ]

setup(
    #version=versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    version='0.31',
    name="punpy",
    url="https://github.com/comet-toolkit/punpy",
    license="LGPLv3",
    author="CoMet Toolkit Team",
    author_email="team@comet-toolkit.org",
    description="Propagating UNcertainties in PYthon",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["comet_maths", "numpy", "numdifftools==0.9.39","scipy","xarray","netcdf4"],
    extras_require={"dev": ["pre-commit", "tox", "sphinx", "sphinx_rtd_theme"]},
    # ext_modules=cythonize(extensions),
)
