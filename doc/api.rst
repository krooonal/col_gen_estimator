####################
col_gen_estimator API
####################

This is an example on how to document the API of your own project.

.. currentmodule:: col_gen_estimator

Estimator
=========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BooleanDecisionRuleClassifier
   BooleanDecisionRuleClassifierWithHeuristic
   DTreeClassifier

Master Problems
=========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BDRMasterProblem
   DTreeMasterProblem

Subproblems and heuristics
=========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BDRSubProblem
   BDRHeuristic
   DTreeSubProblem
   DTreeSubProblemSat
   DTreeSubProblemHeuristic
   DTreeSubProblemOld

Other Classifiers
=========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PathGenerator
   Row
   Path
   Node
   Leaf
   Split
   CutGenerator