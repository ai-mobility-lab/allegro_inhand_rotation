"""Image and pose conventions shared by the simulator export and its writer."""

import numpy as np

FINGER_NAMES = ["index", "middle", "ring", "thumb"]


def resized_intrinsics(matrix, source_size, target_size):
    """Resize an actual sensor K using PIL's pixel-center convention; sizes are (W,H)."""
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    return dict(w=int(target_size[0]), h=int(target_size[1]),
                fx=float(matrix[0, 0] * sx), fy=float(matrix[1, 1] * sy),
                cx=float((matrix[0, 2] + 0.5) * sx - 0.5),
                cy=float((matrix[1, 2] + 0.5) * sy - 0.5))


# Exact transform used by NeuralFeels Allegro._hora_to_neural:
# T_world_camera = T_world_tip @ inverse(T_camera_tip).
NEURALFEELS_TIP_TO_CAMERA = np.linalg.inv(np.array([
    [0., -1., 0., .000021], [0., 0., 1., -.017545],
    [-1., 0., 0., -.002132], [0., 0., 0., 1.],
]))
NEURALFEELS_CAMERA_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)
