from nnsight import util
from nnsight.modeling.mixins import RemoteableMixin
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline
import torch
from typing import Any, Dict, List, Optional, Union

class NNSightSimpleACEStep(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.pipeline = SimpleACEStepPipeline(
            *args,
            **kwargs,
        )
        self.pipeline.load()
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)

class NNSightSimpleACEStepModel(RemoteableMixin):
    __methods__ = {"generate": "_generate"}

    def __init__(self, *args, **kwargs) -> None:
        self._model: NNSightSimpleACEStep = None
        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, **kwargs):
        model = NNSightSimpleACEStep(
            repo_id,
            **kwargs,
        )
        return model

    def _load(self, repo_id: str, **kwargs) -> NNSightSimpleACEStep:

        model = NNSightSimpleACEStep(repo_id, **kwargs)

        return model

    def _prepare_input(self, inputs: Union[str, List[str]]) -> Any:
        if isinstance(inputs, str):
            inputs = [inputs]
        return ((inputs,), {}), len(inputs)

    def _batch(self, batched_inputs: Optional[Dict[str, Any]], prepared_inputs: Any) -> torch.Tensor:
        if batched_inputs is None:
            return ((prepared_inputs,), {})
        return (batched_inputs + prepared_inputs,)

    def _execute(self, prepared_inputs: Any, *args, **kwargs):
        return self._model.pipeline.generate(prepared_inputs, *args, **kwargs)

    def _generate(self, prepared_inputs: Any, *args, seed: int = None, **kwargs):
        if self._scanning():
            kwargs["infer_step"] = 1
        generator = torch.Generator()
        if seed is not None:
            if isinstance(prepared_inputs, list):
                generator = [torch.Generator().manual_seed(seed) for _ in range(len(prepared_inputs))]
            else:
                generator = generator.manual_seed(seed)
        output = self._model.pipeline.generate(prepared_inputs, *args, **kwargs)
        output = self._model(output)
        return output 