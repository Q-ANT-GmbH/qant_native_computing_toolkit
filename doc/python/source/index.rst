Q.ANT native computing toolkit
==============================

This library provides functionality relevant for scientific workloads or neural network inference on Q.ANT native processing units.
It also acts as the user interface for the Q.ANT native computing driver, which is a required dependency.

Installation
************

Install this package from the provided wheel via ``pip``.
To use the Q.ANT native computing toolkit, you need to have the Q.ANT native computing driver installed on your system.
You can install the debian package of the installer from the provided ``.deb`` file via ``apt``. 

API documentation
*****************

The Python API is grouped into five submodules:

- ``qant_native_computing_toolkit.generic``
- ``qant_native_computing_toolkit.native``
- ``qant_native_computing_toolkit.ai``
- ``qant_native_computing_toolkit.info``
- ``qant_native_computing_toolkit.utils``

Generic functionality
---------------------

.. automodule:: qant_native_computing_toolkit.generic
   :members:
   :imported-members:
   :no-index:

Native math functionality
-------------------------

.. automodule:: qant_native_computing_toolkit.native
   :members:
   :imported-members:

AI functionality
----------------

.. automodule:: qant_native_computing_toolkit.ai
   :members:
   :imported-members:
   :no-index:

Information and logging
-----------------------

.. automodule:: qant_native_computing_toolkit.info
   :members:
   :imported-members:
   :no-index:

Helper functions
----------------

.. automodule:: qant_native_computing_toolkit.utils
   :members:
   :imported-members:
   :no-index:

Memory Layout
-------------------------------

The NPU requires all input buffers to be aligned to a page boundary of 4096 bytes.
If the provided data is not page-aligned, the runtime will perform an additional copy, which can negatively impact performance.
To avoid this unnecessary copy, ensure that your data is already aligned before passing it to the NPU.

You can use the helper function ``qant_native_computing_toolkit.utils.align_ndarray_page_boundary()``
to create a page-aligned NumPy array.

