import numpy as np

from reap.kinematics.angles import angle_at, pose_angles
from reap.pose.mappings import MP_TO_COCO


def test_right_angle():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(angle_at(a, b, c) - 90.0) < 1e-6


def test_straight_leg():
    k = np.zeros((17, 2))
    k[11] = [0, 0]  # hip
    k[13] = [0, 1]  # knee
    k[15] = [0, 2]  # ankle
    angs = pose_angles(k)
    assert abs(angs["left_knee"] - 180.0) < 1e-6


def test_mapping_covers_all_coco():
    assert sorted(MP_TO_COCO.keys()) == list(range(17))
    assert len(set(MP_TO_COCO.values())) == 17
