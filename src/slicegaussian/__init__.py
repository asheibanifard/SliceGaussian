"""SliceGaussian: frequency-aware Gaussian representation of slice-based volumes."""
from .field import GaussianField, render_planes, render_volume
from .phase1 import P1Config
from .phase2 import P2Config
from .pipeline import Config, ReestimationConfig, fit, load_volume
from .volume import Volume

__version__ = "0.1.0"
__all__ = ["Config", "GaussianField", "P1Config", "P2Config", "ReestimationConfig", "Volume", "fit", "load_volume",
           "render_planes", "render_volume", "__version__"]
