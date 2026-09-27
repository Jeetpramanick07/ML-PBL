"""FastAPI demo backend for the asr-personalization pipeline.

This package exists purely to serve a live PBL review/expo demo over the
local network — it is not a production service (see module docstrings in
main.py for the full scope statement). It wraps the existing research
pipeline (src.models, src.training, src.inference, src.evaluation) rather
than reimplementing any of it.
"""
