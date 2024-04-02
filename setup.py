import io
import os
import re
from distutils.core import setup

from setuptools import find_packages

exec(open("punpy/_version.py").read())

# from Cython.Build import cythonize


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


# extensions = [Extension(
#         name="utilities",
#         sources=["punpy/utilities/utilities.pyx"],
#         include_dirs=[numpy.get_include()],
#         )
#     ]

setup(
    version=__version__,
    name="punpy",
    url="https://github.com/comet-toolkit/punpy",
    license="LGPLv3",
    author="CoMet Toolkit Team",
    author_email="team@comet-toolkit.org",
    description="Propagating UNcertainties in PYthon",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "comet_maths>=1.0.0",
        "obsarray>=1.0.0",
        "numpy",
        "scipy",
        "netcdf4",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "tox",
            "sphinx",
            "sphinx_design",
            "sphinx_book_theme",
            "ipython",
            "sphinx_autosummary_accessors",
        ],
        ':python_version >= "3.9"': "xarray>=2023.6.0",
        ':python_version < "3.9"': "xarray==0.19.0",
    },
    # ext_modules=cythonize(extensions),
)
