punpy-npl
=========

Propagating UNcertainties in PYthon

Usage
=====

Virtual environment
-------------------

It's always recommended to make a virtual environment for each of your python
projects. Use your preferred virtual environment manager if you want and
activate it for the rest of these commands. If you're unfamiliar, read
https://realpython.com/python-virtual-environments-a-primer/. You can set one up
using::

    python -m venv venv

and then activate it on Windows by using ``venv/Scripts/activate``. 

Installation
------------

Install your package and its dependancies by using::

    pip install -e .

Development
-----------

For developing the package, you'll want to install the pre-commit hooks as well. Type::

    pre-commit install


Note that from now on when you commit, `black` will check your code for styling
errors. If it finds any it will correct them, but the commit will be aborted.
This is so that you can check its work before you continue. If you're happy,
just commit again. 

Compatibility
-------------

Licence
-------

Authors
-------

`punpy-npl` was written by `Pieter De Vis <pieter.de.vis@npl.co.uk>`_.
