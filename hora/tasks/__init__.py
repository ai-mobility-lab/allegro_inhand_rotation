from hora.tasks.isaaclab.allegro_hand_grasp_env import AllegroHandGraspEnv
from hora.tasks.isaaclab.allegro_hand_hora_env import AllegroHandHoraEnv, build_hora_env_cfg
from hora.tasks.isaaclab.wrapper import HoraDirectEnvWrapper


def AllegroHandHora(config, sim_device, graphics_device_id, headless):
    cfg = build_hora_env_cfg(config, sim_device, graphics_device_id, headless)
    env = AllegroHandHoraEnv(cfg, render_mode=None if headless else "human")
    return HoraDirectEnvWrapper(env, config)


def AllegroHandGrasp(config, sim_device, graphics_device_id, headless):
    cfg = build_hora_env_cfg(config, sim_device, graphics_device_id, headless)
    env = AllegroHandGraspEnv(cfg, render_mode=None if headless else "human")
    return HoraDirectEnvWrapper(env, config)


# Mappings from strings to environments
# Map task names (strings) to their environment factories.
# Multiple task names can point to the same factory when the implementation is identical
# but differs only in configuration (e.g., left vs. right hand or other config parameters).
isaaclab_task_map = {
    "AllegroHandHora": AllegroHandHora,           # hora original
    "AllegroHandGrasp": AllegroHandGrasp,
    "RightAllegroHandHora": AllegroHandHora,      # allgero v4 right hand
    "RightAllegroHandGrasp": AllegroHandGrasp,
    "LeftAllegroHandHora": AllegroHandHora,       # allgero v4 left hand
    "LeftAllegroHandGrasp": AllegroHandGrasp,
    "LeftAllegroHandDigitHora": AllegroHandHora,  # left hand w/ DIGIT tactile fingertips
    "LeftAllegroHandDigitGrasp": AllegroHandGrasp,
}
