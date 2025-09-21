# DiffGesture/scripts/data_loader/motion_preprocessor.py
import numpy as np


class MotionProcessorBase:
    def __init__(self, skeletons, mean_pose):
        # Force everything to NumPy with consistent shape
        self.skeletons = np.asarray(skeletons, dtype=np.float32)
        self.mean_pose = np.asarray(mean_pose, dtype=np.float32).reshape(-1, 3)
        self.filtering_message = "PASS"

    def get(self, as_list=True):
        """Run filtering checks and return skeletons + message.

        Args:
            as_list (bool): If True, return skeletons as Python list.
                            If False, return NumPy array.
        """
        if self.skeletons is None or self.skeletons.size == 0:
            return [] if as_list else np.empty((0, *self.mean_pose.shape)), "empty_skeleton"

        # filtering
        if self.check_pose_diff():
            self.skeletons = np.empty((0, *self.mean_pose.shape))
            self.filtering_message = "pose"
        elif self.check_spine_angle():
            self.skeletons = np.empty((0, *self.mean_pose.shape))
            self.filtering_message = "spine angle"
        elif self.check_static_motion():
            self.skeletons = np.empty((0, *self.mean_pose.shape))
            self.filtering_message = "motion"
        # sanity check for NaNs
        if self.skeletons.size > 0:
            assert not np.isnan(self.skeletons).any(), "NaN detected in skeletons"

        return self.skeletons.tolist() if as_list else self.skeletons, self.filtering_message

    def process(self, motion_data):
        raise NotImplementedError("Subclasses should implement this method.")
