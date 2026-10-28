"""Prevalence simulation of pooling designs (Phase 2, item 2.4).

Model: each of the n samples is positive independently with probability p (no test errors).

One-stage designs (a t x n 0/1 matrix, rows = pools) are decoded with COMP: a sample is
*flagged* if it is in no negative pool. A flagged sample is *certain* if it is the only flagged
sample in some positive pool (that pool must contain a positive, so it is this sample; this is
the DD rule). Every flagged sample that is not certain is retested individually, as a second
stage. A d-disjunct design certifies every positive whenever there are at most d positives, so
then no retest is needed. With more positives, COMP never misses a positive (flagged ⊇
positives), so after the retests every sample is classified correctly.

Adaptive hierarchical splitting (PoolPy's "Hierarchical") is simulated directly in
``gt.sim.designs.hierarchical_tests``.
"""

from gt.sim.prevalence import SimResult, simulate_hierarchical, simulate_one_stage

__all__ = ["SimResult", "simulate_one_stage", "simulate_hierarchical"]
