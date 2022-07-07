======
ConQuR
======


.. image:: https://img.shields.io/pypi/v/conqur.svg
        :target: https://pypi.python.org/pypi/conqur

.. image:: https://img.shields.io/travis/Karkarmath/conqur.svg
        :target: https://travis-ci.com/Karkarmath/conqur

.. image:: https://readthedocs.org/projects/conqur/badge/?version=latest
        :target: https://conqur.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Batch effects removal for microbiome data via conditional quantile regression on Python.


* Free software: MIT license
* Documentation: https://conqur.readthedocs.io.


Description
--------

We implement in Python the batch effect removal algorithm described in the article:

* https://www.biorxiv.org/content/10.1101/2021.09.23.461592v1

In this article you can find a description of the main tasks that the algorithm solves and a description of the methods
that are used in this case. The first implementation of the algorithm, made by one of the authors of the article,
was made in the R language. When developing this algorithm, we tried to make it as similar as possible to the
original one, but added some new features, see Features for details. At this link you can find the original implementation
of the algorithm:

* https://github.com/wdl2459/ConQuR

Features
--------

* the values of features, butches and covariates are now arbitrary float;

* the use of logistic regression with an L2 penalty is available;

* the use of arbitrary quantiles in quantile regression is available.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
