Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As of right now, `deprl` has been tested with python 3.8 - 3.11, Windows, Macos and Ubuntu.

.. _installation:

deprl uses the `poetry <https://python-poetry.org>`_ package and dependency manager. If you are not familiar with poetry, you can simply install `deprl` either from pypi, i.e.

.. code-block:: bash

  pip install deprl

or from source:

.. code-block:: bash

  git clone https://github.com/martius-lab/depRL.git
  cd deprl
  pip install -e .


The default pypi installation of deprl includes GPU support. If CUDA is not installed on your system, try the CPU version explicitly with:

.. code-block:: bash

  pip install torch --index-url https://download.pytorch.org/whl/cpu

If you have any missing requirements, try calling:

.. code-block:: bash

  pip install -r requirements.txt

but that should normally not be necessary.

A poetry installation can be done with:

.. code-block:: bash

  git clone https://github.com/martius-lab/depRL.git
  cd deprl
  poetry install

but requires you to setup poetry first.
