# DiffGesture/scripts/data_loader/motion_preprocessor.py
import numpy as np
from .motion_precessor_base import MotionProcessorBase


class MotionPreprocessor(MotionProcessorBase):
    def __init__(self, skeletons, mean_pose):
        super().__init__(skeletons, mean_pose)

    def check_static_motion(self, verbose=False):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print('skip - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print('pass - check_static_motion left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return False

    def check_pose_diff(self, verbose=False):
        diff = np.abs(self.skeletons - self.mean_pose)
        diff = np.mean(diff)

        # th = 0.017
        th = 0.02  # exclude 3594
        if diff < th:
            if verbose:
                print('skip - check_pose_diff {:.5f}'.format(diff))
            return True
        else:
            if verbose:
                print('pass - check_pose_diff {:.5f}'.format(diff))
            return False

    def check_spine_angle(self, verbose=False):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
            # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print('skip - check_spine_angle {:.5f}, {:.5f}'.format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print('pass - check_spine_angle {:.5f}'.format(max(angles)))
            return False
