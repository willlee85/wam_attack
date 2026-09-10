# Reference snippets for eval_polict_client_openpi.py joint-energy shared-delta mode.

# --- episode loop init ---
shared_joint_delta = None
shared_joint_metrics = None

# --- first frame: reuse delta for episodes 1..N-1 ---
if first:
    observation = TASK_ENV.get_obs()
    first_obs = format_obs(observation, prompt)
    if joint_config is not None and shared_joint_delta is not None:
        first_obs = apply_delta_to_observation(
            {"obs": first_obs}, shared_joint_delta
        )["obs"]

# --- only episode 0 triggers server-side PGD ---
if (
    first
    and joint_config is not None
    and shared_joint_delta is None
):
    request["joint_energy_attack"] = joint_config

# --- after infer: cache delta from episode 0 ---
if first and "joint_energy_attack_delta_by_key" in ret:
    episode_delta = ret["joint_energy_attack_delta_by_key"]
    shared_joint_delta = episode_delta
    shared_joint_metrics = ret["joint_energy_attack_metrics"]
    save_joint_energy_attack_artifacts(
        args["save_root"], st_seed, task_name, 0,
        shared_joint_metrics, shared_joint_delta,
    )
    first_obs = apply_delta_to_observation(
        {"obs": first_obs}, episode_delta
    )["obs"]

# --- parse_override_pairs: keep layer specs as strings ---
literal_string_keys = {
    "wac_layers",
    "wac_denoise_steps",
    "joint_energy_layers",
    "joint_energy_denoise_steps",
}
