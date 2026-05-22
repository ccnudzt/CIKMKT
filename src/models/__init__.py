"""Model registry and model implementations."""

from src.core.registry import Registry

MODEL_REGISTRY = Registry("model")

# Import modules for side-effect registration.
from src.models import dkt  # noqa: F401
from src.models import gcl_kaga_dkt  # noqa: F401
from src.models import kaga_dkt  # noqa: F401
from src.models import text_gcl_dkt  # noqa: F401
