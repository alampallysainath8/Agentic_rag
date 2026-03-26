
from __future__ import annotations


def get_grok_model(model: str = "grok-3"):
    """
    Build and return a DeepEval GrokModel instance.

    Parameters
    ----------
    model : Grok model variant – "grok-3", "grok-3-mini", etc.

    Returns
    -------
    GrokModel ready to pass to any DeepEval metric's ``model=`` argument.
    """
    try:
        from deepeval.models import GrokModel as _GrokModel  # type: ignore
        return _GrokModel(model=model)
    except ImportError as exc:
        raise ImportError(
            "deepeval is not installed. Run: pip install deepeval"
        ) from exc
