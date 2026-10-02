# DIGIT contact rotation task

`LeftAllegroHandDigitContactHora` inherits `LeftAllegroHandDigitHora`, including
its hand asset, objects, randomization, and grasp cache.

The privileged observation grows from 9 to 21 values. The last 12 values are
world-frame xyz normal contact forces in newtons, ordered by
`env.asset.fingertipLinkNames` (index, middle, ring, thumb). Each sensor is
filtered against the manipulated object: ground and hand contacts are excluded.
These are the normal force vectors reported by IsaacLab, without tangential
friction forces, sampled at the end of the control step.

The 96 policy observations and the 30-by-32 proprioceptive history are unchanged.
Stage 1 uses the forces through the privileged encoder. Stage 2 actions depend
only on proprioception; during adaptation training, the frozen privileged
encoder still needs all 21 values to supply its teacher target. Stage 2 inference
does not need privileged observations. Train a new stage 1 checkpoint for this
task: the original 9-input privileged encoder has a different shape.

The reward adds `fingertipContactRewardScale * number_of_contacting_pads` to the
existing rotation reward and penalties. A pad counts when its filtered force
magnitude exceeds `env.contact.forceThreshold` (default 0.01 N). The default
scale is 0.1, so four touching pads add 0.4 per step. Increasing force on an
already contacting pad gives no extra bonus. The mean count and weighted bonus
are logged as `fingertip_contact_count` and `fingertip_contact_reward`.

Use the existing training scripts with task and output overrides:

```bash
bash scripts/train_s1.sh 0 42 contact_run \
  task=LeftAllegroHandDigitContactHora \
  train.ppo.output_name=LeftAllegroHandDigitContactHora/contact_run

bash scripts/train_s2.sh 0 42 contact_run \
  task=LeftAllegroHandDigitContactHora \
  train.ppo.output_name=LeftAllegroHandDigitContactHora/contact_run \
  checkpoint=outputs/LeftAllegroHandDigitContactHora/contact_run/stage1_nn/best.pth
```

Tune the bonus with `task.env.reward.fingertipContactRewardScale=0.1` and the
contact threshold with `task.env.contact.forceThreshold=0.01`.
