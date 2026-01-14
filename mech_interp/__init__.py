"""
Mechanistic Interpretability Experiments

This package contains tools for analyzing the internal decision-making process
of language models at branching points where token choice matters.

Modules:
  - token_impact: Identifies impactful token positions via divergence analysis
  - dla: Direct Logit Attribution - layer-wise contribution decomposition
  - patching: Activation patching for causal intervention
  - gradient: Gradient attribution for sensitivity analysis
  - run_mech_interp: Main runner script for the full pipeline
"""

__version__ = "0.1.0"
