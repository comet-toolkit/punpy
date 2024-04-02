Punpy: Propagating Uncertainties with PYthon
==================================================

The **punpy** module is a Python software package to propagate random, structured and systematic uncertainties through a given measurement function. 

**punpy** can be used as a standalone tool, where the input uncertainties are inputted manually.
Alternatively, **punpy** can also be used in combination with digital effects tables created with **obsarray**.
This documentation provides general information on how to use the module (with some examples), as well as a detailed API of the included classes and function.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::  Quickstart Guide
        :link: content/getting_started
        :link-type: doc

        New to *punpy*? Check out the quickstart guide for an introduction.

    .. grid-item-card::  User Guide
        :link: content/user_guide
        :link-type: doc

        The user guide provides a documentation and examples how to use **punpy** either standalone or in combination with *obsarray* digital effects tables.

    .. grid-item-card::  ATBD
        :link: content/atbd
        :link-type: doc

        ATBD mathematical description of **punpy** (under development).

    .. grid-item-card::  API Reference
        :link: content/api
        :link-type: doc

        The API Reference contains a description of each of the **punpy** classes and functions.


Acknowledgements
----------------

**punpy** has been developed by `Pieter De Vis <https://github.com/pdevis>`_.

The development has been funded by:

* The UK's Department for Business, Energy and Industrial Strategy's (BEIS) National Measurement System (NMS) programme
* The IDEAS-QA4EO project funded by the European Space Agency.

Project status
--------------

**punpy** is under active development. It is beta software.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   Quickstart <content/getting_started>
   User Guide <content/user_guide>
   ATBD <content/atbd>
   API Reference <content/api>
