"""Minimal OpenBCI EEG classification library."""

from .artifact import load_artifact
from .inference import predict_file

__all__ = ["load_artifact", "predict_file"]
