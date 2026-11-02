"""Saliency map methods with a shared batch interface."""

from .base import SaliencyMethod
from .cam import CAM
from .gradcam import GradCAM, GradCAMPlusPlus, GuidedGradCAM, XGradCAM
from .lime import LIME
from .lrp import LRP
from .perturbation_cam import AblationCAM, ScoreCAM
from .rise import RISE

__all__ = [
    "SaliencyMethod",
    "CAM",
    "GradCAM",
    "GradCAMPlusPlus",
    "GuidedGradCAM",
    "XGradCAM",
    "LRP",
    "AblationCAM",
    "ScoreCAM",
    "RISE",
    "LIME",
]
