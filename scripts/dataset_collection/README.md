# Export for unmodified NeuralFeels

Both collectors now write data for the original NeuralFeels `Allegro`,
`TactileDataset`, `DigitSensor`, and `RealsenseSensor` classes. No NeuralFeels code
changes or extra loader fields are required.

## Collection

Use a new directory: existing recordings are not converted, and the writer refuses
nonempty episode directories to avoid mixing old and new frames.

```bash
python scripts/collect_stage1_feelsight_dataset.py --enable_cameras --headless \
  --checkpoint outputs/LeftAllegroHandDigitHora/dextouch_pear/stage1_nn/best.pth \
  --output_dir data/feelsight_sim_legacy --num_episodes 2 --episode_steps 300 \
  --camera_views front-left -- task.env.object.type=dextouch_pear
```

Both stage collectors also support `--task LeftAllegroHandDigitContactHora`.
Use a checkpoint trained for that task and the corresponding stage; the contact
task adds 12 fingertip force values to the 9 object privileged observations.
The collectors retain its contact sensors and reward alongside the dataset cameras.
For example, in the IsaacLab Python environment:

```bash
python scripts/collect_stage1_feelsight_dataset.py --enable_cameras --headless \
  --task LeftAllegroHandDigitContactHora \
  --checkpoint outputs/LeftAllegroHandDigitContactHora/baseline/stage1_nn/best.pth \
  --output_dir data/feelsight_contact --num_episodes 2 --episode_steps 300
```

For stage 2, use `collect_stage2_feelsight_dataset.py` and a `stage2_nn` checkpoint.
Without `--checkpoint`, the default is `outputs/<task>/baseline/stageN_nn/best.pth`.
Without `--object_name`, the object type comes from the task configuration (or a
`task.env.object.type=...` override after `--`).

The collector reads the target NeuralFeels URDF without modifying it. The default
is the sibling checkout's `data/assets/allegro/allegro_digit_left_ball.urdf`.
Use `--neuralfeels_urdf PATH` if that checkout is elsewhere.

## Camera alignment and image encoding

The simulator cameras now use the exact inverse matrix from NeuralFeels'
`Allegro._hora_to_neural`, relative to each DIGIT housing tip:

```text
position (meters):      (-0.002132, 0.000021, 0.017545)
quaternion (w,x,y,z):    (0.5, 0.5, -0.5, -0.5)
IsaacLab convention:    opengl
```

The far clip is 50 mm so the gel is visible from this camera position. The nominal
22 mm distance sets the camera aperture/FOV; actual depth comes from the sensor.
The hand URDF must also match the NeuralFeels kinematic geometry after finger
renaming, including the thumb joint-13 origin and thumb-tip rotation.

The collector checks every camera's world pose against NeuralFeels FK on every
saved frame. A disagreement raises an error, identifying a possible URDF, USD
cache, joint mapping, or camera-offset mismatch. It never hides a mismatch by
reprojecting onto a larger image canvas.

- Joint values are gathered by name: simulator joints `8..11, 4..7, 0..3, 12..15`
  become the saved NeuralFeels joint blocks.
- Simulator ring is saved as NeuralFeels index, and simulator index as NeuralFeels
  ring; middle and thumb names are unchanged.
- RGB is saved directly from the full sensor image, with resizing only. There is
  no extra rotation, adaptive FOV, point splatting, or artificial black padding.
- The Taxim renderer retains its calibrated 640x480 input size. The default saved
  image is 240x320 (W x H); both focal lengths and principal-point coordinates are
  independently scaled using the actual camera K and pixel-center convention.
  This handles the aspect-ratio change consistently with backprojection.
- One shared tactile K is saved unchanged across frames and episodes.
- Depth uses the original loader formula:

  ```text
  negative_camera_z_meters = depth_uint8 / digit_info.depth_scale + digit_info.cam_dist
  ```

  Zero is reserved for no contact. The offset and scale are fitted per episode to
  avoid saturation; `--digit_depth_scale` is the maximum scale in units per meter.
  Quantization error is at most half a depth unit before float rounding.

Tactile RGB remains JPEG. Depth/mask filenames remain `<frame>.jpg` because the
original loader hard-codes that extension, but their payload is lossless PNG;
OpenCV reads by file header. There are no tactile `depth.npz`,
`allegro.camera_poses`, or `digit_info.gel_depth` extensions.

Scene RGB/depth remains **640x480** by default. Scene camera poses use OpenGL axes
(+X right, +Y up, -Z forward), with negative camera-z meters in `depth.npz` and
scale 1. Calibration is taken from the actual camera sensor.

Terminal steps are excluded because IsaacLab has already reset the state when
`step()` returns done. Nonempty shorter episodes are saved. Rendering runs every
physics step to keep images synchronized with poses.

## NeuralFeels settings

Use the original **GT tactile depth mode** to consume the exported measurements.
Learned depth predictions are not ground-truth simulator depth. For the standard
`main=vitac` configuration:

```text
main.sensor0.masks=read
main.sensor0.sim_noise_iters=0
main.sensor1.tactile_depth.mode=gt
main.sensor2.tactile_depth.mode=gt
main.sensor3.tactile_depth.mode=gt
main.sensor4.tactile_depth.mode=gt
```

These are existing configuration options, not changes to NeuralFeels. Record
`front-left` for its default vision sensor `realsense_front_left`.

The original loader's filtering still applies: tactile contact masks below 1% of
image area are ignored, and scene GT masks are ignored unless all three labels
(background, hand, object) occur. The exporter does not fabricate labels or contacts
to bypass these filters. Existing recordings are unchanged; regenerate to use the
matched cameras and direct full-frame image export.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q scripts/dataset_collection/tests/test_neuralfeels_export.py
```

CPU tests exercise the original consumer source without optional SLAM model imports.
Set `NEURALFEELS_ROOT` if the checkout is not a sibling. The simulator smoke test
also loads produced files with the actual unmodified NeuralFeels environment.
