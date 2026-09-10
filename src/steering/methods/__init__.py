"""Method packages — importing this module registers every Controller and Scorer
with the internal lookup tables (used by the CLI runners) and re-exports each
concrete wrapper for direct use::

    from src.steering import CAASteeringController
    ctrl = CAASteeringController.from_pretrained("user/ace-step-caa-piano", alpha=20)
"""

from .audioldm_caa import AudioLDMCAASteeringController, AudioLDMCaaScorer
from .austeer import AUSteerSteeringController, AuSteerScorer
from .caa import CAASteeringController, CAAScorer
from .concept_slider import ConceptSlidersSteeringController, ConceptSliderTrainScorer
from .freesliders import FreeSlidersSteeringController
from .pci import PCISteeringController
from .sae import LayerSpec, SAEActivationsScorer, SAESteeringController, SAETrainScorer
from .stable_audio_caa import StableAudioCAASteeringController, StableAudioCaaScorer
from .textemb import TextEmbSteeringController
from .tokemb import TokEmbSteeringController, TokEmbScorer

__all__ = [
    "AudioLDMCAASteeringController",
    "AudioLDMCaaScorer",
    "AUSteerSteeringController",
    "AuSteerScorer",
    "CAASteeringController",
    "CAAScorer",
    "ConceptSlidersSteeringController",
    "ConceptSliderTrainScorer",
    "FreeSlidersSteeringController",
    "LayerSpec",
    "PCISteeringController",
    "SAEActivationsScorer",
    "SAESteeringController",
    "SAETrainScorer",
    "StableAudioCAASteeringController",
    "StableAudioCaaScorer",
    "TextEmbSteeringController",
    "TokEmbSteeringController",
    "TokEmbScorer",
]
