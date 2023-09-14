.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: create your own column generation based estimator
==================================================

Master Problem
---------
Each column generation based classifier must create a Master Problem class that 
inherits :class:`BaseMasterProblem`. It can be imported as::

    >>> from ._col_gen_classifier import BaseMasterProblem

By default the :class:`BaseMasterProblem` uses the Google OR-Tools
MPSolver with 'glop' to solve the linear programs. We can use a different underlying solver
by providing the 'solver_str' argument to the constructor. A complete list of valid arguments
are given at https://github.com/google/or-tools/blob/v9.4/ortools/linear_solver/linear_solver.h#L262

We can also choose to not use the OR-Tools MPSolver but this is not recommended.

We need to implement the following abstract methods.

* ``generate_mp``: Generates the master problem model (RMP) and initializes the primal
        and dual solutions.
* ``add_column``: Adds the given column to the master problem model.
* ``solve_rmp``: Solves the RMP with given solver params. Returns the dual costs.
* ``solve_ip``: Solves the integer RMP with given solver params.
        Returns false if the problem is not solved.

Subproblem
---------
Each column generation based classifier must create a Master Problem class that 
inherits :class:`BaseSubproblem`. It can be imported as::

    >>> from ._col_gen_classifier import BaseSubproblem

By default the :class:`BaseMasterProblem` uses the Google OR-Tools
MPSolver with 'cbc' to solve the integer linear programs. We can use a different underlying solver
by providing the 'solver_str' argument to the constructor. A complete list of valid arguments
are given at https://github.com/google/or-tools/blob/v9.4/ortools/linear_solver/linear_solver.h#L262

We can also choose to not use the OR-Tools MPSolver by implementing a different way 
to generate columns. This is specifically used for heuristic based column generation.

The :class:`BaseSubproblem` also provides support to solve the subproblems using OR-Tools
CP-SAT solver (with automated parameter tuning). 

We need to implement the following abstract method.
* ``generate_columns``: Generates the new columns to be added to the RMP.


Estimator
---------

The central piece of column generation based classifier is
:class:`ColGenClassifier`. All column generation based estimators in this repository are derived
from this class. In more details, this base class enables to set and get
parameters of the estimator. It can be imported as::

    >>> from ._col_gen_classifier import ColGenClassifier

Once imported, we can create a class which inherate from this base class::

    >>> class MyOwnEstimator(ColGenClassifier):
    ...     pass



.. _mixin: https://en.wikipedia.org/wiki/Mixin

Predictor
---------

Classifier
~~~~~~~~~~

Classifiers implement ``predict``. In addition, they
output the probabilities of the prediction using the ``predict_proba`` method:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``. The output corresponds to the predicted class for each sample;
* ``predict_proba`` will give a 2D matrix where each column corresponds to the
  class and each entry will be the probability of the associated class.

In addition, scikit-learn provides a mixin, i.e.
:class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method
which computes the accuracy score of the predictions. The :class:`ColGenClassifier` already
inherits the :class:`sklearn.base.ClassifierMixin`.

In order to create a column generation based classifier, :class:`MyOwnClassifier` which inherits
from :class:`ColGenClassifier`. 

The ``fit`` method is already implemented in :class:`ColGenClassifier`

We need to implement the ``predict`` and ``predict_proba`` methods for our classifier.

