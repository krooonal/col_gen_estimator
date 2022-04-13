.. -*- mode: rst -*-

|AppVeyor|_ |Codecov|_ |CircleCI|_

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/6eo2m9ydofn1nvb6?svg=true
.. _AppVeyor: https://ci.appveyor.com/api/projects/status/6eo2m9ydofn1nvb6

.. |Codecov| image:: https://codecov.io/gh/krooonal/col_gen_estimator/branch/master/graph/badge.svg?token=ZR8HME2LGV
.. _Codecov: https://codecov.io/gh/krooonal/col_gen_estimator

.. |CircleCI| image:: https://circleci.com/gh/krooonal/col_gen_estimator/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/krooonal/col_gen_estimator/tree/master


col_gen_estimator - A template for scikit-learn compatible column generation 
based estimators contributions
============================================================

**col_gen_estimator** is a template project for scikit-learn compatible
column generation based estimators.

This project is built using the sklearn template. 

It aids development of estimators that can be used in scikit-learn pipelines
and (hyper)parameter search, while facilitating testing (including some API
compliance), documentation, open source development, packaging, and continuous
integration.

An example extension of a column generation based binary classifier (Boolean 
Decision Rule Generation by S. Dash et. al. 2018) is included. The developer 
needs to extend the master and subproblem classes and implement the required
methods. The coumn generation part is taken care of by the template fit method.

*Thank you for cleanly contributing to the estimator template!*