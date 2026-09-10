# Reference snippet for wan_va/wan_va_server.py infer() attack branches.
# Insert after compute_kv_cache handling, before normal inference.

elif wac_attack is not None:
    if self.frame_st_id != 0:
        raise RuntimeError(
            "wac_attack is only supported for the first observation of an episode"
        )
    from experiments.lingbot_wac.attack import attack_observation
    with torch.enable_grad():
        attack_result = attack_observation(self, obs, wac_attack)
    if inference_seed is not None:
        torch.manual_seed(int(inference_seed))
        torch.cuda.manual_seed_all(int(inference_seed))
    action, _ = self._infer(
        attack_result.attacked_observation, frame_st_id=self.frame_st_id
    )
    return dict(
        action=action,
        wac_metrics=attack_result.metrics,
        wac_delta_by_key=attack_result.delta_by_key,
    )

elif vae_attack is not None:
    if self.frame_st_id != 0:
        raise RuntimeError(
            "vae_attack is only supported for the first observation of an episode"
        )
    from experiments.lingbot_vae_attack.attack import attack_observation
    with torch.enable_grad():
        attack_result = attack_observation(self, obs, vae_attack)
    if inference_seed is not None:
        torch.manual_seed(int(inference_seed))
        torch.cuda.manual_seed_all(int(inference_seed))
    action, _ = self._infer(
        attack_result.attacked_observation, frame_st_id=self.frame_st_id
    )
    return dict(
        action=action,
        vae_attack_metrics=attack_result.metrics,
        vae_attack_delta_by_key=attack_result.delta_by_key,
    )

elif joint_energy_attack is not None:
    if self.frame_st_id != 0:
        raise RuntimeError(
            "joint_energy_attack is only supported for the first observation "
            "of an episode"
        )
    from experiments.lingbot_joint_energy_attack.attack import attack_observation
    with torch.enable_grad():
        attack_result = attack_observation(self, obs, joint_energy_attack)
    if inference_seed is not None:
        torch.manual_seed(int(inference_seed))
        torch.cuda.manual_seed_all(int(inference_seed))
    action, _ = self._infer(
        attack_result.attacked_observation, frame_st_id=self.frame_st_id
    )
    return dict(
        action=action,
        joint_energy_attack_metrics=attack_result.metrics,
        joint_energy_attack_delta_by_key=attack_result.delta_by_key,
    )
