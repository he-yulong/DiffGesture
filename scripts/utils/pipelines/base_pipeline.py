# DiffGesture/scripts/utils/base_pipeline.py

class BasePipeline:
    """Abstract base class for all pipelines."""

    def __init__(self, args, diffusion, lang_model, speaker_model, mean_pose, mean_dir_vec, pose_dim, collate_fn,
                 device):
        self.args = args
        self.diffusion = diffusion
        self.lang_model = lang_model
        self.speaker_model = speaker_model
        self.mean_pose = mean_pose
        self.mean_dir_vec = mean_dir_vec
        self.pose_dim = pose_dim
        self.collate_fn = collate_fn
        self.device = device
        self.val_data_path = "data/ted_expressive_dataset/val"

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
