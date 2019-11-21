from typing import *

import numpy as np

from joint import Joint
from pose import Pose

# head to joint thresholds
# > THS[0] = max acceptable length of the segment head->head
# > THS[1] = max acceptable length of the segment head->neck
# > THS[1] = max acceptable length of the segment head->right_shoulder
# ...
THS = [0, 0.5, 1, 1.5, 2, 1, 1.5, 2, 2, 2.5, 3, 2, 2.5, 3]


def dist(p1, p2):
    # type: (Tuple[int, float, float, float], Tuple[int, float, float, float]) -> float
    """
    Returns the Euclidian distance between quadruples (jtype, x3d, y3d, z3d)
    """
    return np.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)


def coord_to_joint(coord, person_id, conf):
    # type: (Tuple[int, float, float, float], int, float) -> Joint
    """
    Given a `coord`, i.e. a tuple (jtype, x3d, y3d, z3d) and a person ID,
    returns the corresponding `Joint` object
    """
    return Joint(np.array([-1, person_id, coord[0], -1, -1, coord[1], coord[2], coord[3], 0, 0]), confidence=conf)


def coords_to_poses(coords, confidences):
    # type: (List[Tuple[int, float, float, float]], List[float]) -> List[Pose]
    """
    Associate the input joints to form the poses.
I   Input joints are provided as a list of quadruples (jtype, x3d, y3d, z3d).
    :param coords: list of quadruples (jtype, x3d, y3d, z3d)
    :param confidences: list of confidence values (confidences[i] is the confidence value of coords[i])
    :return: list of `Pose` objects
    """

    coords = list(zip(coords, confidences))
    coords.sort(key=lambda x: -x[1])

    # get all the heads
    heads = [c for c in coords if c[0][0] == 0]

    # attach the closest joint to each head
    poses = []
    done = []
    for person_id, head in enumerate(heads):
        pose = [coord_to_joint(head[0], person_id=person_id, conf=head[1])]
        for jtype in range(1, 14):
            try:
                other_joint = min([j for j in coords if j[0][0] == jtype], key=lambda j: dist(head[0], j[0]))
                if dist(head[0], other_joint[0]) < THS[jtype] and not other_joint[0] in done:
                    pose.append(coord_to_joint(other_joint[0], person_id=person_id, conf=other_joint[1]))
                    done.append(other_joint[0])
                else:
                    # holes are represented by a `Joint` with jtype=-1 and 3D-coords=(0,0,0)
                    pose.append(coord_to_joint((-1, 0, 0, 0), person_id=person_id, conf=0))
            except ValueError:
                # if I arrive here it means that I have not found
                # any valid joint of that type for that person
                pose.append(coord_to_joint((-1, 0, 0, 0), person_id=person_id, conf=0))
        poses.append(Pose(pose))

    return poses
