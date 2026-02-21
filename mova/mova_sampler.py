"""
MOVA dual-tower sampler for ComfyUI-WanVideoWrapper.

Adapts the MOVA pipeline's dual-tower inference (visual DiT + audio DiT + bridge)
to work with WanVideoWrapper's WanModel block interface.

The WanVideoWrapper WanModel blocks use a complex kwargs dict and return 4-tuples.
This module handles:
  - Computing all required embeddings via WanModel sub-modules
  - Building the full kwargs dict for each WanAttentionBlock call
  - Running the dual-tower block loop with DualTowerConditionalBridge interactions
  - Stage-1/stage-2 visual DiT switching at the boundary timestep
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from einops import rearrange


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal position embedding."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def build_visual_kwargs(
    transformer,
    e0: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    context: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    current_step: int,
    last_step: bool,
    F_frames: int,
    original_seq_len: int,
    transformer_options: Optional[Dict] = None,
) -> Dict:
    """
    Build the full kwargs dict required by WanVideoWrapper's WanAttentionBlock.forward().
    All MOVA-irrelevant features are set to None/False/0/empty defaults.
    """
    if transformer_options is None:
        transformer_options = {}

    return dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        clip_embed=clip_embed,
        current_step=torch.tensor(current_step),
        last_step=torch.tensor(last_step, dtype=torch.bool),
        chunked_self_attention=False,
        seq_chunks=0,
        camera_embed=None,
        audio_proj=None,
        num_latent_frames=F_frames,
        frame_tokens=None,      # will be set below after visual_x is known
        original_seq_len=original_seq_len,
        enhance_enabled=False,
        audio_scale=1.0,
        nag_params={},
        nag_context=None,
        multitalk_audio_embedding=None,
        ref_target_masks=None,
        human_num=0,
        inner_t=None,
        inner_c=None,
        cross_freqs=None,
        freqs_ip=None,
        e_ip=None,
        adapter_proj=None,
        ip_scale=1.0,
        reverse_time=False,
        mtv_motion_tokens=None,
        mtv_motion_rotary_emb=None,
        mtv_strength=1.0,
        mtv_freqs=None,
        humo_audio_input=None,
        humo_audio_scale=1.0,
        lynx_x_ip=None,
        lynx_ip_scale=1.0,
        lynx_ref_scale=1.0,
        longcat_num_cond_latents=None,
        longcat_avatar_options=None,
        onetoall_ref_scale=1.0,
        e_tr=None,
        tr_start=0,
        tr_num=0,
        transformer_options=transformer_options,
    )


@torch.no_grad()
def mova_inference_single_step(
    visual_dit,
    audio_dit,
    bridge,
    visual_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    timestep: torch.Tensor,
    audio_timestep: torch.Tensor,
    video_fps: float,
    clip_fea: Optional[torch.Tensor] = None,
    transformer_options: Optional[Dict] = None,
    condition_scale: float = 1.0,
    a2v_condition_scale: Optional[float] = None,
    v2a_condition_scale: Optional[float] = None,
    current_step: int = 0,
    last_step: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one denoising step through the MOVA dual-tower DiT.

    Args:
        visual_dit: WanVideoWrapper WanModel (stage-1 or stage-2)
        audio_dit: MOVA WanAudioModel
        bridge: MOVA DualTowerConditionalBridge
        visual_latents: [B, C, T, H, W] — combined noisy + condition latents
        audio_latents: [B, latent_dim, audio_T] — noisy audio latents
        context: [B, seq_len, text_dim] — text embeddings (already encoded by T5)
        timestep: [1] float32 — video timestep
        audio_timestep: [1] float32 — audio timestep
        video_fps: float — frames per second of the video
        clip_fea: optional CLIP image features for I2V [B, N, C]
        transformer_options: ComfyUI transformer options dict
        condition_scale: bridge conditioning strength
        a2v_condition_scale: audio->visual scale (overrides condition_scale)
        v2a_condition_scale: visual->audio scale (overrides condition_scale)
        current_step: int step index (for transformer_options compatibility)
        last_step: bool whether this is the final denoising step

    Returns:
        (visual_noise_pred, audio_noise_pred) — predicted noise for each modality
    """
    if transformer_options is None:
        transformer_options = {}

    device = visual_latents.device
    model_dtype = visual_dit.base_dtype if hasattr(visual_dit, 'base_dtype') else visual_latents.dtype

    # ------------------------------------------------------------------ #
    # 1. Time embeddings
    # ------------------------------------------------------------------ #
    with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=torch.float32):
        visual_t = visual_dit.time_embedding(
            sinusoidal_embedding_1d(visual_dit.freq_dim, timestep)
        )  # [B, dim]
        visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))  # [B, 6, dim]

        audio_t = audio_dit.time_embedding(
            sinusoidal_embedding_1d(audio_dit.freq_dim, audio_timestep)
        )
        audio_t_mod = audio_dit.time_projection(audio_t).unflatten(1, (6, audio_dit.dim))

    visual_t = visual_t.to(model_dtype)
    visual_t_mod = visual_t_mod.to(model_dtype)
    audio_t = audio_t.to(model_dtype)
    audio_t_mod = audio_t_mod.to(model_dtype)

    # ------------------------------------------------------------------ #
    # 2. Text/CLIP embeddings
    # ------------------------------------------------------------------ #
    # context can be:
    #   - a pre-stacked [B, seq_len, text_dim] tensor (when nodes.py already stacked it)
    #   - a list of variable-length tensors [seq_len_i, text_dim] (raw from T5)
    # WanModel.forward() pads each tensor to model.text_len before calling text_embedding.
    # Since we call text_embedding directly, we must replicate that padding.
    text_len = getattr(visual_dit, 'text_len', None)

    def _prepare_context(ctx, model, tlen):
        """Pad ctx to tlen and call model.text_embedding."""
        if isinstance(ctx, torch.Tensor):
            if ctx.dim() == 3:
                # Already [B, seq_len, dim] — pad if shorter than text_len
                if tlen is not None and ctx.shape[1] < tlen:
                    pad = ctx.new_zeros(ctx.shape[0], tlen - ctx.shape[1], ctx.shape[2])
                    ctx = torch.cat([ctx, pad], dim=1)
                elif tlen is not None and ctx.shape[1] > tlen:
                    ctx = ctx[:, :tlen]
            else:
                ctx = ctx.unsqueeze(0)
                if tlen is not None:
                    if ctx.shape[1] < tlen:
                        pad = ctx.new_zeros(ctx.shape[0], tlen - ctx.shape[1], ctx.shape[2])
                        ctx = torch.cat([ctx, pad], dim=1)
                    else:
                        ctx = ctx[:, :tlen]
        else:
            # List of variable-length tensors
            if tlen is not None:
                ctx = torch.stack([
                    torch.cat([u, u.new_zeros(tlen - u.size(0), u.size(1))]) if u.size(0) < tlen
                    else u[:tlen]
                    for u in ctx
                ])
            else:
                # No text_len — just stack (will fail if lengths differ, but that's an error)
                ctx = torch.stack(ctx, dim=0)
        return ctx.to(model_dtype)

    visual_context = visual_dit.text_embedding(_prepare_context(context, visual_dit, text_len)).to(model_dtype)
    # Audio DiT has its own text_embedding and text_len
    audio_text_len = getattr(audio_dit, 'text_len', text_len)
    audio_context = audio_dit.text_embedding(_prepare_context(context, audio_dit, audio_text_len)).to(model_dtype)

    # CLIP embedding for I2V
    clip_embed = None
    if clip_fea is not None and hasattr(visual_dit, 'img_emb'):
        clip_embed = visual_dit.img_emb(clip_fea.to(model_dtype))

    # ------------------------------------------------------------------ #
    # 3. Patchify visual latents
    # ------------------------------------------------------------------ #
    visual_x = visual_latents.to(model_dtype)
    # Wan's rope_encode_comfy expects pre-patch latent grid sizes (F, H, W),
    # not the post-patch (T, H, W) sizes from patch_embedding output.
    F_in, H_in, W_in = visual_x.shape[2], visual_x.shape[3], visual_x.shape[4]
    # WanModel.patch_embedding takes B,C,T,H,W
    visual_x_patched_list = [
        visual_dit.patch_embedding(visual_x[i:i+1].float()).to(model_dtype)
        for i in range(visual_x.shape[0])
    ]
    # grid_sizes: [B, 3] = [[T, H, W], ...]
    grid_sizes = torch.stack([
        torch.tensor(px.shape[2:], dtype=torch.long, device=device)
        for px in visual_x_patched_list
    ])  # shape: [B, 3]
    T, H, W = grid_sizes[0]
    T, H, W = int(T), int(H), int(W)

    visual_x_flat_list = [px.flatten(2).transpose(1, 2) for px in visual_x_patched_list]
    # seq_lens for padded batched attention
    seq_lens = torch.tensor([v.shape[1] for v in visual_x_flat_list], dtype=torch.int32, device=device)

    # Pad to same length and concat if batch > 1 (all same for uniform resolution)
    visual_x = torch.cat(visual_x_flat_list, dim=0)  # [B, T*H*W, dim]

    # ------------------------------------------------------------------ #
    # 4. RoPE frequencies for visual
    # ------------------------------------------------------------------ #
    # Use WanModel's rope_encode_comfy if available (preferred), else build manually.
    # IMPORTANT: pass pre-patch latent grid sizes to match WanModel.forward().
    if hasattr(visual_dit, 'rope_encode_comfy'):
        visual_freqs = visual_dit.rope_encode_comfy(F_in, H_in, W_in, device=device, dtype=model_dtype)
    else:
        visual_freqs_tuple = tuple(freq.to(device) for freq in visual_dit.freqs)
        visual_freqs = torch.cat([
            visual_freqs_tuple[0][:T].view(T, 1, 1, -1).expand(T, H, W, -1),
            visual_freqs_tuple[1][:H].view(1, H, 1, -1).expand(T, H, W, -1),
            visual_freqs_tuple[2][:W].view(1, 1, W, -1).expand(T, H, W, -1),
        ], dim=-1).reshape(T * H * W, 1, -1).to(model_dtype)

    # ------------------------------------------------------------------ #
    # 5. Patchify audio latents
    # ------------------------------------------------------------------ #
    audio_x = audio_latents.to(model_dtype)
    audio_x, (aud_f,) = audio_dit.patchify(audio_x, None)
    # audio_x: [B, aud_f, audio_dim]

    audio_freqs = torch.cat([
        audio_dit.freqs[0][:aud_f].view(aud_f, -1),
        audio_dit.freqs[1][:aud_f].view(aud_f, -1),
        audio_dit.freqs[2][:aud_f].view(aud_f, -1),
    ], dim=-1).reshape(aud_f, 1, -1).to(device)

    # ------------------------------------------------------------------ #
    # 6. Align RoPE for cross-modal bridge (if enabled)
    # ------------------------------------------------------------------ #
    visual_rope_cos_sin = None
    audio_rope_cos_sin = None
    if bridge.apply_cross_rope:
        visual_rope_cos_sin, audio_rope_cos_sin = bridge.build_aligned_freqs(
            video_fps=video_fps,
            grid_size=(T, H, W),
            audio_steps=aud_f,
            device=device,
            dtype=model_dtype,
        )

    # ------------------------------------------------------------------ #
    # 7. Build WanModel block kwargs
    # ------------------------------------------------------------------ #
    F_frames = T  # number of latent frames
    kwargs = build_visual_kwargs(
        transformer=visual_dit,
        e0=visual_t_mod,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=visual_freqs,
        context=visual_context,
        clip_embed=clip_embed,
        current_step=current_step,
        last_step=last_step,
        F_frames=F_frames,
        original_seq_len=getattr(visual_dit, 'original_seq_len', visual_x.shape[1]),
        transformer_options=transformer_options,
    )
    # Set frame_tokens now that we know visual_x shape
    kwargs['frame_tokens'] = visual_x.shape[1] // F_frames

    # ------------------------------------------------------------------ #
    # 8. Dual-tower block loop
    # ------------------------------------------------------------------ #
    visual_x, audio_x = _forward_dual_tower(
        visual_dit=visual_dit,
        audio_dit=audio_dit,
        bridge=bridge,
        visual_x=visual_x,
        audio_x=audio_x,
        visual_t_mod=visual_t_mod,
        audio_t_mod=audio_t_mod,
        audio_context=audio_context,
        audio_freqs=audio_freqs,
        kwargs=kwargs,
        grid_size=(T, H, W),
        visual_rope_cos_sin=visual_rope_cos_sin,
        audio_rope_cos_sin=audio_rope_cos_sin,
        condition_scale=condition_scale,
        a2v_condition_scale=a2v_condition_scale,
        v2a_condition_scale=v2a_condition_scale,
    )

    # ------------------------------------------------------------------ #
    # 9. Unproject / unpatchify
    # ------------------------------------------------------------------ #
    visual_output = visual_dit.head(visual_x, visual_t)
    visual_output = visual_dit.unpatchify(visual_output, grid_sizes[0])
    # visual_output: [B, C, T, H*patch, W*patch] — shaped as original latent

    audio_output = audio_dit.head(audio_x, audio_t)
    audio_output = audio_dit.unpatchify(audio_output, (aud_f,))
    # audio_output: [B, latent_dim, audio_T]

    return visual_output, audio_output


def _forward_dual_tower(
    visual_dit,
    audio_dit,
    bridge,
    visual_x: torch.Tensor,
    audio_x: torch.Tensor,
    visual_t_mod: torch.Tensor,
    audio_t_mod: torch.Tensor,
    audio_context: torch.Tensor,
    audio_freqs: torch.Tensor,
    kwargs: Dict,
    grid_size: Tuple[int, int, int],
    visual_rope_cos_sin: Optional[Tuple] = None,
    audio_rope_cos_sin: Optional[Tuple] = None,
    condition_scale: float = 1.0,
    a2v_condition_scale: Optional[float] = None,
    v2a_condition_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the dual-tower DiT block loop with the bridge.

    Visual blocks use WanVideoWrapper's WanAttentionBlock (returns 4-tuple).
    Audio blocks use MOVA's DiTBlock (returns single tensor).
    Bridge interactions fire at layers specified by the interaction_mapping.
    """
    visual_blocks = visual_dit.blocks
    audio_blocks = audio_dit.blocks
    num_visual = len(visual_blocks)
    num_audio = len(audio_blocks)
    min_layers = min(num_visual, num_audio)

    # These are unused by MOVA but expected in block return tuple
    x_ip = None
    lynx_ref_feature = None
    x_ovi = None

    for layer_idx in range(min_layers):
        # Bridge interactions come BEFORE each block (MOVA convention)
        if bridge.should_interact(layer_idx, 'a2v'):
            visual_x, audio_x = bridge(
                layer_idx,
                visual_x,
                audio_x,
                x_freqs=visual_rope_cos_sin,
                y_freqs=audio_rope_cos_sin,
                a2v_condition_scale=a2v_condition_scale,
                v2a_condition_scale=v2a_condition_scale,
                condition_scale=condition_scale,
                video_grid_size=grid_size,
            )

        # Visual block (WanAttentionBlock — returns 4-tuple)
        block_out = visual_blocks[layer_idx](
            visual_x,
            x_ip=x_ip,
            lynx_ref_feature=lynx_ref_feature,
            x_ovi=x_ovi,
            x_onetoall_ref=None,
            onetoall_freqs=None,
            attention_mode_override=None,
            **kwargs,
        )
        visual_x = block_out[0]
        x_ip = block_out[1]
        lynx_ref_feature = block_out[2]
        x_ovi = block_out[3]

        # Audio block (MOVA DiTBlock — returns single tensor)
        audio_x = audio_blocks[layer_idx](audio_x, audio_context, audio_t_mod, audio_freqs)

    # Remaining visual-only blocks (when visual has more layers than audio)
    for layer_idx in range(min_layers, num_visual):
        block_out = visual_blocks[layer_idx](
            visual_x,
            x_ip=x_ip,
            lynx_ref_feature=lynx_ref_feature,
            x_ovi=x_ovi,
            x_onetoall_ref=None,
            onetoall_freqs=None,
            attention_mode_override=None,
            **kwargs,
        )
        visual_x = block_out[0]
        x_ip = block_out[1]
        lynx_ref_feature = block_out[2]
        x_ovi = block_out[3]

    return visual_x, audio_x
