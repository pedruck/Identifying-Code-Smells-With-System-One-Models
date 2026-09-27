"""smellclm — code-smell verification with the official CLM-v0.1-8B verifier.

The package holds everything that does not need a GPU (dataset contract, static
metrics, leakage control, frozen split, conversion to the official CLM typed
decision schema, baselines, metrics and bootstrap) plus thin wrappers around
the official CLM embedding recipe and projection heads for the GPU stages.
"""

__version__ = "0.1.0"
