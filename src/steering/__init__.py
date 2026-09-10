"""Public API for the audio-interv steering framework.

Two layers:

1. **Core ABCs / runtime** — :class:`Controller`, :class:`Scorer`, the two
   model wrappers (:class:`SteerableACEModel`, :class:`SteerableAudioLDMModel`)
   and the ``with model.steer(controller):`` context.

2. **Concrete method wrappers** — one class per method, imported here so user
   code can do ``from src.steering import CAASteeringController`` instead of going
   through a string lookup. Each Controller has ``from_pretrained(...)`` (Hub
   or local path) and ``set_alpha(...)`` for alpha sweeps::

       from src.steering import SteerableACEModel, CAASteeringController

       model = SteerableACEModel(device="cuda")
       model.pipeline.load()
       ctrl = CAASteeringController.from_pretrained(
           "lukasz-staniszewski/ace-step-caa-piano", alpha=20,
       )
       with model.steer(ctrl):
           audio = model.generate(prompt="instrumental music",
                                  lyrics="[inst]", audio_duration=10.0,
                                  infer_step=30, manual_seed=0,
                                  return_type="audio")

The registry (``register_method`` / ``register_scorer``) remains internal to
the unified CLI runners (``src/steering/run_eval.py`` /
``src/steering/run_compute.py``). User code uses the classes directly.
"""


class PromptRewriteWarning(UserWarning):
    """Emitted by methods that internally rewrite the user's prompt into a
    (neutral, positive, negative) triple before feeding the model
    (``TextEmbSteeringController``, ``TokEmbSteeringController``).

    Suppress in batch contexts via::

        import warnings
        from src.steering import PromptRewriteWarning
        warnings.simplefilter("ignore", PromptRewriteWarning)
    """


from .controller import CFGAwareMixin, Controller, NullController
from .hub import SteeringVectorArtifact, push_sae_to_hub
from .model import SteerableACEModel, SteerableAudioLDMModel, SteerableStableAudioModel
from .scorer import Scorer

# Method wrappers — each method exposes its public Controller (and Scorer
# where applicable). Importing src.steering.methods triggers registration with
# the internal lookup tables; user code can also do
# ``from src.steering import CAASteeringController`` directly.
from .methods import (
    AudioLDMCAASteeringController,
    AudioLDMCaaScorer,
    AUSteerSteeringController,
    AuSteerScorer,
    CAASteeringController,
    CAAScorer,
    ConceptSlidersSteeringController,
    ConceptSliderTrainScorer,
    FreeSlidersSteeringController,
    LayerSpec,
    PCISteeringController,
    SAEActivationsScorer,
    SAESteeringController,
    SAETrainScorer,
    StableAudioCAASteeringController,
    StableAudioCaaScorer,
    TextEmbSteeringController,
    TokEmbSteeringController,
    TokEmbScorer,
)

__all__ = [
    # Core
    "CFGAwareMixin",
    "Controller",
    "NullController",
    "Scorer",
    "SteerableACEModel",
    "SteerableAudioLDMModel",
    "SteerableStableAudioModel",
    "SteeringVectorArtifact",
    "PromptRewriteWarning",
    "push_sae_to_hub",
    # Method wrappers — Controllers
    "AudioLDMCAASteeringController",
    "AUSteerSteeringController",
    "CAASteeringController",
    "ConceptSlidersSteeringController",
    "FreeSlidersSteeringController",
    "PCISteeringController",
    "SAESteeringController",
    "StableAudioCAASteeringController",
    "TextEmbSteeringController",
    "TokEmbSteeringController",
    # Method wrappers — Scorers (only methods with a compute phase)
    "AudioLDMCaaScorer",
    "StableAudioCaaScorer",
    "AuSteerScorer",
    "CAAScorer",
    "ConceptSliderTrainScorer",
    "SAEActivationsScorer",
    "SAETrainScorer",
    "TokEmbScorer",
    # SAE helper
    "LayerSpec",
]
