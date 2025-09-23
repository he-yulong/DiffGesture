# DiffGesture/scripts/utils/pipeline_factory.py
from scripts.utils.pipelines.eval_pipeline import EvalPipeline
from scripts.utils.pipelines.short_pipeline import ShortPipeline
from scripts.utils.pipelines.long_pipeline import LongPipeline


class PipelineFactory:
    """Factory to create pipelines based on mode."""

    @staticmethod
    def create(mode, *args, **kwargs):
        if mode == "eval":
            return EvalPipeline(*args, **kwargs)
        elif mode == "short":
            return ShortPipeline(*args, **kwargs)
        elif mode == "long":
            return LongPipeline(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
