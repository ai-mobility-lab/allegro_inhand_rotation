# DIGIT Taxim calibration data

`isaaclab_contrib`'s `VisuoTactileSensor` renders camera-based tactile RGB images with the Taxim
example-based model (Si & Yuan, 2022, https://arxiv.org/abs/2109.04027). This requires real
photometric calibration data captured from a physical sensor — it cannot be derived from the
DIGIT paper's published spec (size/resolution/sensing field) alone.

Drop the following two files, captured from a real DIGIT sensor, into this directory:

- `bg.jpg` — a reference background image (no contact) from the sensor's camera.
- `polycalib.npz` — polynomial photometric-stereo calibration data, containing three arrays
  (`grad_r`, `grad_g`, `grad_b`) of shape `(num_bins, num_bins, 6)`, mapping a surface-gradient
  bin `(magnitude, direction)` to a quadratic-polynomial RGB response. See the Taxim paper/repo
  (https://github.com/Robo-Touch/Taxim) for the calibration procedure that produces this file
  from a set of calibration images (e.g. ball-indentation images at known depths/positions).

Until these are added, `DIGIT_CFG` in `inhand_rotation/assets/digit_sensor_cfg.py` cannot be used
with `enable_camera_tactile=True` — `GelsightRender.__init__` raises `FileNotFoundError` if either
file is missing (see `isaaclab_contrib/sensors/tacsl_sensor/visuotactile_render.py`).

An optional `real_bg.npy` (real background height data) may also be added; it is not required.
