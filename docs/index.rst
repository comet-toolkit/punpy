.. punpy documentation master file, created by
   sphinx-quickstart on Fri Mar 20 17:28:40 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Punpy: Propagating Uncertainties with PYthon
====================================================================

The **punpy** module is a Python software package to propagate random, structured and systematic uncertainties through a given measurement function. 

**punpy** can be used as a standalone tool, where the input uncertainties are inputted manually.
Alternatively, **punpy** can also be used in combination with digital effects tables created with **obsarray**.
This documentation provides general information on how to use the module (with some examples), as well as a detailed API of the included classes and function.

~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   content/getting_started
   content/punpy_standalone
   content/punpy_digital_effects_table
   content/punpy_memory_and_speed
   content/examples
   content/atbd


API Documentation
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 4

   content/API/punpy.mc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
