"""Differentiable Motus inference helpers for white-box input noise training."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def build_initial_latents(
    model,
    *,
    condition_frame_latent: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = condition_frame_latent.shape[0]
    _, channels, _, height, width = condition_frame_latent.shape
    num_total_latent_frames = 1 + model.config.num_video_frames // 4
    generator = torch.Generator(device=model.device).manual_seed(int(seed))
    try:
        video_latent = torch.randn(
            (batch_size, channels, num_total_latent_frames, height, width),
            device=model.device,
            dtype=model.dtype,
            generator=generator,
        )
        action_latent = torch.randn(
            (batch_size, model.config.action_chunk_size, model.config.action_dim),
            device=model.device,
            dtype=model.dtype,
            generator=generator,
        )
    except TypeError:
        torch.manual_seed(int(seed))
        video_latent = torch.randn(
            (batch_size, channels, num_total_latent_frames, height, width),
            device=model.device,
            dtype=model.dtype,
        )
        action_latent = torch.randn(
            (batch_size, model.config.action_chunk_size, model.config.action_dim),
            device=model.device,
            dtype=model.dtype,
        )
    return video_latent, action_latent


def encode_first_frame_latents(model, first_frame: torch.Tensor) -> torch.Tensor:
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2).to(model.dtype)
    return model.video_model.vae.encode(first_frame_norm)


def infer_actions_whitebox(
    model,
    *,
    first_frame: torch.Tensor,
    state: torch.Tensor,
    language_embeddings: List[torch.Tensor],
    vlm_inputs,
    num_inference_steps: int,
    video_latent_init: torch.Tensor,
    action_latent_init: torch.Tensor,
) -> torch.Tensor:
    """Run Motus joint denoising with gradients through the condition-frame VAE encode."""
    batch_size = first_frame.shape[0]
    first_frame = first_frame.to(model.device, dtype=model.dtype)
    state = state.to(model.device, dtype=model.dtype)
    language_embeddings = [emb.to(model.device, dtype=model.dtype) for emb in language_embeddings]

    condition_frame_latent = encode_first_frame_latents(model, first_frame)
    video_latent = video_latent_init.clone().to(model.device, dtype=model.dtype)
    video_latent[:, :, 0:1] = condition_frame_latent
    action_latent = action_latent_init.clone().to(model.device, dtype=model.dtype)

    with torch.no_grad():
        und_tokens = model.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = model.video_module.preprocess_t5_embeddings(language_embeddings)

    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=model.device, dtype=model.dtype)
    for step_idx in range(num_inference_steps):
        t = timesteps[step_idx]
        t_next = timesteps[step_idx + 1]
        dt = t_next - t
        video_t_scaled = (t * 1000).expand(batch_size).to(model.dtype)
        action_t_scaled = (t * 1000).expand(batch_size).to(model.dtype)

        video_tokens = model.video_module.prepare_input(video_latent.to(model.dtype))
        state_tokens = state.unsqueeze(1).to(model.dtype)
        registers = model.action_expert.registers.expand(batch_size, -1, -1)
        action_tokens = model.action_expert.input_encoder(state_tokens, action_latent, registers)

        with torch.autocast(device_type="cuda", dtype=model.video_model.precision):
            video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(
                video_t_scaled, video_tokens.shape[1]
            )
            action_head_time_emb, action_adaln_params = model.action_module.get_time_embedding(
                action_t_scaled, action_tokens.shape[1]
            )

            for layer_idx in range(model.config.num_layers):
                video_adaln_modulation = model.video_module.compute_adaln_modulation(
                    video_adaln_params, layer_idx
                )
                action_adaln_modulation = model.action_module.compute_adaln_modulation(
                    action_adaln_params, layer_idx
                )
                video_tokens, action_tokens, und_tokens = model.video_module.process_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    model.action_expert.blocks[layer_idx],
                    und_tokens,
                    model.und_expert.blocks[layer_idx],
                )
                video_tokens = model.video_module.process_cross_attention(
                    video_tokens, video_adaln_params, layer_idx, processed_t5_context
                )
                video_tokens = model.video_module.process_ffn(
                    video_tokens, video_adaln_modulation, layer_idx
                )
                action_tokens = model.action_module.process_ffn(
                    action_tokens, action_adaln_modulation, layer_idx
                )
                und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            video_velocity = model.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = model.action_expert.decoder(action_tokens, action_head_time_emb)
            action_velocity = action_pred_full[
                :, 1 : -model.action_expert.config.num_registers, :
            ]

        updated_video = video_latent + video_velocity * dt
        video_latent = torch.cat(
            [condition_frame_latent, updated_video[:, :, 1:]],
            dim=2,
        )
        action_latent = action_latent + action_velocity * dt

    return action_latent.float()


def compute_action_deviation_loss(
    adv_action: torch.Tensor,
    clean_action: torch.Tensor,
    *,
    loss_type: str = "mse",
) -> tuple[torch.Tensor, float]:
    adv = adv_action.float()
    clean = clean_action.float()
    if loss_type == "l1":
        objective = (adv - clean).abs().mean()
    else:
        objective = (adv - clean).pow(2).mean()
    return -objective, float(objective.detach().item())
