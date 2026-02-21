"""
MOVA ComfyUI nodes for ComfyUI-WanVideoWrapper.

Provides:
  - MOVAAudioDITLoader  : loads audio_dit/ only  → MOVA_AUDIO_DIT
  - MOVABridgeLoader    : loads dual_tower_bridge/ only  → MOVA_BRIDGE
  - MOVAAudioVAELoader  : loads audio_vae/ only  → MOVA_AUDIO_VAE
  - MOVASampler         : dual-tower denoising
        required: WANVIDEOMODEL (stage-1) + MOVA_AUDIO_DIT + MOVA_BRIDGE
        optional: WANVIDEOMODEL (stage-2, video_dit_2) + WANVIDEOTEXTEMBEDS
        outputs:  LATENT + MOVA_AUDIO_LATENTS
  - MOVADecodeAudio     : decodes MOVA_AUDIO_LATENTS → AUDIO

Checkpoint layout (HuggingFace MOVA-360p / MOVA-720p):
    <ckpt_path>/
        audio_dit/         (2.84 GB, single safetensors)
        audio_vae/         (743 MB,  single safetensors)
        dual_tower_bridge/ (5.32 GB, single safetensors)
        video_dit/         (28.6 GB, 3 shards) ← use WanVideoModelLoader
        video_dit_2/       (28.6 GB, 3 shards) ← use a 2nd WanVideoModelLoader

Workflow pattern:
    WanVideoModelLoader (video_dit merged)     → WANVIDEOMODEL  ─────────────────┬
    WanVideoModelLoader (video_dit_2 merged)   → WANVIDEOMODEL (opt.)  ───────┤
    MOVAAudioDITLoader                         → MOVA_AUDIO_DIT  ───────────┤
    MOVABridgeLoader                           → MOVA_BRIDGE  ──────────────┤→ MOVASampler
    WanVideoTextEncode                         → WANVIDEOTEXTEMBEDS  ───────|
    WanVideoImageToVideoEncode                 → WANVIDIMAGE_EMBEDS  ───────┘
"""
import gc
import json
import os
from copy import deepcopy
from typing import Optional

import torch
from tqdm import tqdm

import folder_paths
from comfy import model_management as mm
from comfy.utils import ProgressBar

from ..utils import log, set_module_tensor_to_device
from ..wanvideo.schedulers import get_scheduler, scheduler_list
from .mova_sampler import mova_inference_single_step, sinusoidal_embedding_1d

script_directory = os.path.dirname(os.path.abspath(__file__))

# Register a model folder for MOVA checkpoints
folder_paths.add_model_folder_path(
    "mova",
    os.path.join(folder_paths.models_dir, "mova")
)


def _load_safetensors_or_bin(model_dir: str, device: torch.device) -> dict:
    """
    Load model weights from a directory, supporting:
      1. diffusion_pytorch_model.safetensors  (single file, HuggingFace default)
      2. diffusion_pytorch_model.safetensors.index.json + sharded shard files
         (diffusion_pytorch_model-NNNNN-of-NNNNN.safetensors)
      3. model.safetensors  (single file, legacy/converted)
      4. model.bin  (PyTorch legacy)
    """
    from safetensors.torch import load_file as load_safetensors

    # Priority 1: diffusion_pytorch_model.safetensors (standard HuggingFace single-file)
    dp_sf_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(dp_sf_path):
        log.info(f"[MOVA] Loading single-file: {dp_sf_path}")
        return load_safetensors(dp_sf_path, device=str(device))

    # Priority 2: sharded diffusion_pytorch_model (index.json + shard files)
    index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Collect unique shard filenames in sorted order
        shard_files = sorted(set(weight_map.values()))
        log.info(f"[MOVA] Loading {len(shard_files)} shards from {model_dir}")
        merged: dict = {}
        for shard_name in shard_files:
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"[MOVA] Shard not found: {shard_path}")
            log.info(f"[MOVA]   Loading shard: {shard_name}")
            merged.update(load_safetensors(shard_path, device=str(device)))
        return merged

    # Priority 3: legacy model.safetensors
    sf_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(sf_path):
        log.info(f"[MOVA] Loading legacy single-file: {sf_path}")
        return load_safetensors(sf_path, device=str(device))

    # Priority 4: model.bin
    bin_path = os.path.join(model_dir, "model.bin")
    if os.path.exists(bin_path):
        log.info(f"[MOVA] Loading .bin file: {bin_path}")
        return torch.load(bin_path, map_location=device, weights_only=True)

    raise FileNotFoundError(
        f"[MOVA] No weights found in {model_dir}.\n"
        "Expected one of: diffusion_pytorch_model.safetensors, "
        "diffusion_pytorch_model.safetensors.index.json (+shards), "
        "model.safetensors, or model.bin"
    )


def _load_model_from_dir(
    model_class,
    model_dir: str,
    dtype: torch.dtype,
    offload_device: torch.device,
    fp8_mode: str = "disabled",   # "disabled" | "fp8_e4m3fn" | "fp8_e5m2"
):
    """
    Instantiate a diffusers-style ModelMixin model from a checkpoint directory using
    init_empty_weights + manual tensor placement with optional FP8 quantization.

    For very large models (e.g. video_dit_2 at 28.6 GB) fp8_mode="fp8_e4m3fn" can halve
    VRAM usage while preserving reasonable quality.
    """
    from accelerate import init_empty_weights

    fp8_dtype_map = {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    fp8_dtype = fp8_dtype_map.get(fp8_mode, None)  # None → use `dtype`
    # Only quantise linear weight tensors to FP8; keep norms/biases/embeddings in dtype
    FP8_KEEP_PATTERNS = ("norm", "bias", "pos_emb", "embedding", "time_embedding",
                         "text_embedding", "modulation", "head", "img_emb", "freqs")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in: {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)
    # Remove diffusers metadata keys
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    with init_empty_weights():
        model = model_class(**cfg)
    model.eval()

    state_dict = _load_safetensors_or_bin(model_dir, torch.device("cpu"))
    for name, param in model.named_parameters():
        tensor = state_dict.get(name)
        if tensor is None:
            # Try common prefix strip
            for prefix in ("model.",):
                key = prefix + name
                if key in state_dict:
                    tensor = state_dict[key]
                    break
        if tensor is None:
            log.warning(f"[MOVA] Parameter not found in checkpoint: {name}")
            continue

        # Decide target dtype for this tensor
        if fp8_dtype is not None and not any(p in name for p in FP8_KEEP_PATTERNS):
            target_dtype = fp8_dtype
        else:
            target_dtype = dtype
        set_module_tensor_to_device(model, name, device=offload_device, dtype=target_dtype, value=tensor)

    # Also load buffers (e.g. RotaryEmbedding.inv_freq)
    for name, buf in model.named_buffers():
        if buf is None:
            continue
        tensor = state_dict.get(name)
        if tensor is not None:
            set_module_tensor_to_device(model, name, device=offload_device, dtype=buf.dtype, value=tensor)

    if fp8_dtype is not None:
        log.info(f"[MOVA] {model_class.__name__} loaded with FP8 ({fp8_mode}) for linear weights")
    return model


# ---------------------------------------------------------------------------
# Node 1a: Audio DiT loader
# ---------------------------------------------------------------------------

class MOVAAudioDITLoader:
    """
    Loads the MOVA audio DiT (WanAudioModel) from the audio_dit/ sub-folder.
    Plug the output into MOVASampler's audio_dit input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the MOVA checkpoint root (e.g. /data/MOVA-360p). Must contain audio_dit/ sub-folder.",
                }),
                "base_precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "load_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device"},
                ),
            },
        }

    RETURN_TYPES = ("MOVA_AUDIO_DIT",)
    RETURN_NAMES = ("audio_dit",)
    FUNCTION = "load"
    CATEGORY = "WanVideoWrapper/MOVA"

    def load(self, ckpt_path: str, base_precision: str, load_device: str):
        from ..wanvideo.modules.mova.wan_audio_dit import WanAudioModel

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        target_device = device if load_device == "main_device" else offload_device

        audio_dit_dir = os.path.join(ckpt_path, "audio_dit")
        if not os.path.isdir(audio_dit_dir):
            raise ValueError(f"[MOVA] audio_dit/ not found in: {ckpt_path!r}")

        log.info(f"[MOVA] Loading audio DiT from {audio_dit_dir}")
        model = _load_model_from_dir(WanAudioModel, audio_dit_dir, dtype, target_device)
        log.info("[MOVA] Audio DiT loaded.")
        return ({"model": model, "dtype": dtype},)


# ---------------------------------------------------------------------------
# Node 1b: Bridge loader
# ---------------------------------------------------------------------------

class MOVABridgeLoader:
    """
    Loads the MOVA DualTowerConditionalBridge from the dual_tower_bridge/ sub-folder.
    Plug the output into MOVASampler's bridge input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the MOVA checkpoint root. Must contain dual_tower_bridge/ sub-folder.",
                }),
                "base_precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "load_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device"},
                ),
                "interaction_strategy": (
                    ["shallow_focus", "distributed", "progressive", "full"],
                    {
                        "default": "shallow_focus",
                        "tooltip": "Bridge interaction strategy. 'shallow_focus' (default from MOVA training) uses only the first ~1/3 of layers.",
                    }
                ),
                "condition_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Default bridge conditioning scale. Can be overridden per-direction in MOVASampler.",
                }),
            },
        }

    RETURN_TYPES = ("MOVA_BRIDGE",)
    RETURN_NAMES = ("bridge",)
    FUNCTION = "load"
    CATEGORY = "WanVideoWrapper/MOVA"

    def load(
        self,
        ckpt_path: str,
        base_precision: str,
        load_device: str,
        interaction_strategy: str,
        condition_scale: float,
    ):
        from ..wanvideo.modules.mova.interactionv2 import DualTowerConditionalBridge
        from accelerate import init_empty_weights

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        target_device = device if load_device == "main_device" else offload_device

        bridge_dir = os.path.join(ckpt_path, "dual_tower_bridge")
        if not os.path.isdir(bridge_dir):
            raise ValueError(f"[MOVA] dual_tower_bridge/ not found in: {ckpt_path!r}")

        config_path = os.path.join(bridge_dir, "config.json")
        with open(config_path) as f:
            bridge_cfg = json.load(f)
        bridge_cfg.pop("_class_name", None)
        bridge_cfg.pop("_diffusers_version", None)
        bridge_cfg["interaction_strategy"] = interaction_strategy

        log.info(f"[MOVA] Loading bridge from {bridge_dir} (strategy={interaction_strategy})")
        with init_empty_weights():
            bridge = DualTowerConditionalBridge(**bridge_cfg)
        bridge.eval()

        bridge_sd = _load_safetensors_or_bin(bridge_dir, torch.device("cpu"))
        for name, _param in bridge.named_parameters():
            if name in bridge_sd:
                set_module_tensor_to_device(bridge, name, device=target_device, dtype=dtype, value=bridge_sd[name])
            else:
                log.warning(f"[MOVA] Bridge param not in checkpoint: {name}")
        for name, buf in bridge.named_buffers():
            if buf is None:
                continue
            if name in bridge_sd:
                set_module_tensor_to_device(bridge, name, device=target_device, dtype=buf.dtype, value=bridge_sd[name])

        log.info("[MOVA] Bridge loaded.")
        return ({"model": bridge, "dtype": dtype, "condition_scale": condition_scale},)


# ---------------------------------------------------------------------------
# Node 2: Audio VAE loader
# ---------------------------------------------------------------------------

class MOVAAudioVAELoader:
    """Loads the MOVA DAC audio VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the MOVA checkpoint directory containing audio_vae/ sub-folder.",
                }),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
                "load_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device"},
                ),
            },
        }

    RETURN_TYPES = ("MOVA_AUDIO_VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load"
    CATEGORY = "WanVideoWrapper/MOVA"

    def load(self, ckpt_path: str, base_precision: str, load_device: str):
        from ..wanvideo.modules.mova.dac_vae import DAC

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        target_device = device if load_device == "main_device" else offload_device

        audio_vae_dir = os.path.join(ckpt_path, "audio_vae")
        if not os.path.isdir(audio_vae_dir):
            raise ValueError(f"[MOVA] audio_vae/ not found in: {ckpt_path!r}")

        log.info(f"[MOVA] Loading audio VAE from {audio_vae_dir}")
        audio_vae = _load_model_from_dir(DAC, audio_vae_dir, dtype, target_device)

        audio_vae_dict = {
            "model": audio_vae,
            "dtype": dtype,
            "sample_rate": audio_vae.sample_rate,
            "hop_length": int(audio_vae.hop_length),
            "latent_dim": audio_vae.latent_dim,
        }
        log.info(f"[MOVA] Audio VAE loaded: {audio_vae.sample_rate}Hz, hop={audio_vae.hop_length}, latent_dim={audio_vae.latent_dim}")
        return (audio_vae_dict,)


# ---------------------------------------------------------------------------
# Node 3: MOVA Sampler
# ---------------------------------------------------------------------------

class MOVASampler:
    """
    MOVA dual-tower denoising loop.

    Reuses WanVideoWrapper's existing WANVIDEOMODEL for the visual DiT(s) — the same
    model you get from WanVideoModelLoader — and adds audio generation via a separate
    audio DiT and cross-modal bridge.  Stage-2 video DiT (video_dit_2) is optional:
    connect a second WanVideoModelLoader output to video_dit_2 for the high-noise phase.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", {
                    "tooltip": "Stage-1 video DiT from WanVideoModelLoader (video_dit).",
                }),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", {}),
                "audio_dit": ("MOVA_AUDIO_DIT", {
                    "tooltip": "Audio DiT from MOVAAudioDITLoader.",
                }),
                "bridge": ("MOVA_BRIDGE", {
                    "tooltip": "Dual-tower bridge from MOVABridgeLoader.",
                }),
                "steps": ("INT", {"default": 30, "min": 1, "max": 500}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "scheduler": (scheduler_list, {"default": "unipc"}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1,
                                        "tooltip": "Frames per second — used for cross-modal RoPE alignment."}),
                "audio_latent_channels": ("INT", {"default": 128, "min": 1, "max": 512,
                                                   "tooltip": "latent_dim of the audio VAE. Default 128 for MOVA DAC."}),
                "audio_hop_length": ("INT", {"default": 2048, "min": 1, "max": 16384,
                                              "tooltip": "Audio VAE hop length. Default 2048 for MOVA DAC at 44100 Hz."}),
                "audio_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 96000}),
                "boundary_ratio": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Fraction of the noise schedule after which stage-2 video DiT takes over (if connected). 0.9 = MOVA default."}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_dit_2": ("WANVIDEOMODEL", {
                    "tooltip": "Optional stage-2 video DiT from a second WanVideoModelLoader (video_dit_2). Used for the low-noise phase.",
                }),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", {}),
                "samples": ("LATENT", {"tooltip": "Existing video latents for video2video."}),
                "audio_latents": ("MOVA_AUDIO_LATENTS", {"tooltip": "Existing audio latents for audio2audio."}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "a2v_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                                   "tooltip": "Audio→visual scale (overrides bridge's condition_scale when set)."}),
                "v2a_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                                   "tooltip": "Visual→audio scale (overrides bridge's condition_scale when set)."}),
                "sigmas": ("SIGMAS", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "MOVA_AUDIO_LATENTS",)
    RETURN_NAMES = ("samples", "audio_latents",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/MOVA"

    def process(
        self,
        model,
        image_embeds,
        audio_dit,
        bridge,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        scheduler: str,
        video_fps: float,
        audio_latent_channels: int,
        audio_hop_length: int,
        audio_sample_rate: int,
        boundary_ratio: float,
        force_offload: bool,
        video_dit_2=None,
        text_embeds=None,
        samples=None,
        audio_latents=None,
        denoise_strength: float = 1.0,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        sigmas=None,
    ):
        import copy

        patcher = model
        model_obj = model.model
        transformer = model_obj.diffusion_model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = getattr(model_obj, "base_dtype", None) or torch.bfloat16

        # ---- Extract image / conditioning from WANVIDIMAGE_EMBEDS ----
        # "image_embeds" key holds the condition latent [C, T, H, W] (no batch dim)
        # "mask" holds the temporal mask [4, T, H, W]
        # "clip_context" holds CLIP image features for I2V (may be None for T2V)
        image_cond_latent = image_embeds.get("image_embeds", None)   # [C, T, H, W]
        image_cond_mask   = image_embeds.get("mask", None)            # [4, T, H, W]
        clip_fea          = image_embeds.get("clip_context", None)    # CLIP features (I2V)
        num_frames        = image_embeds.get("num_frames", None)
        lat_h             = image_embeds.get("lat_h", None)
        lat_w             = image_embeds.get("lat_w", None)

        # ---- Text embeddings (WANVIDEOTEXTEMBEDS) ----
        # Keys are "prompt_embeds" (list of tensors) and "negative_prompt_embeds" (list of tensors)
        if text_embeds is not None:
            positive_embeds  = text_embeds.get("prompt_embeds", None)
            negative_embeds  = text_embeds.get("negative_prompt_embeds", None)
        else:
            positive_embeds  = None
            negative_embeds  = None

        if not positive_embeds or not negative_embeds:
            raise ValueError(
                "[MOVASampler] text_embeds must contain non-empty 'prompt_embeds' and "
                "'negative_prompt_embeds' lists."
            )

        # Unwrap single-item lists to tensors; stack multi-prompt lists
        if isinstance(positive_embeds, list):
            positive_context = torch.stack(positive_embeds, dim=0) if len(positive_embeds) > 1 else positive_embeds[0].unsqueeze(0)
        else:
            positive_context = positive_embeds.unsqueeze(0) if positive_embeds.dim() == 2 else positive_embeds

        if isinstance(negative_embeds, list):
            negative_context = torch.stack(negative_embeds, dim=0) if len(negative_embeds) > 1 else negative_embeds[0].unsqueeze(0)
        else:
            negative_context = negative_embeds.unsqueeze(0) if negative_embeds.dim() == 2 else negative_embeds

        # ---- Build initial noise latents ----
        torch.manual_seed(seed)
        if samples is not None:
            init_video_latents = samples["samples"].to(device, dtype=dtype)
            T_lat = init_video_latents.shape[2]
            H_lat = init_video_latents.shape[3]
            W_lat = init_video_latents.shape[4]
        else:
            # Derive shape from image_embeds metadata
            if image_cond_latent is None:
                raise ValueError(
                    "[MOVASampler] image_embeds must contain an 'image_embeds' condition latent. "
                    "Connect a WanVideoImageToVideoEncode node."
                )
            # image_cond_latent is [C, T, H, W] — no batch dimension
            C_cond, T_lat, H_lat, W_lat = image_cond_latent.shape
            B = 1  # MOVA processes one sample at a time
            # Noise latent channels: 16 for standard Wan, 48 for 5B model
            is_5b = transformer.out_dim == 48
            z_dim = 48 if is_5b else 16
            # Number of latent frames including start/end image frames
            latent_frames = (num_frames - 1) // 4 + 1
            init_video_latents = torch.randn(
                B, z_dim, latent_frames, H_lat, W_lat,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            T_lat = latent_frames

        # Compute audio length from video latents
        # video latent T -> num_video_frames via vae_temporal_stride=4
        vae_temporal_stride = 4
        T_lat = init_video_latents.shape[2]
        num_video_frames = (T_lat - 1) * vae_temporal_stride + 1
        audio_num_samples = int(audio_sample_rate * num_video_frames / video_fps)
        audio_latent_t = (audio_num_samples - 1) // audio_hop_length + 1

        if audio_latents is not None:
            init_audio_latents = audio_latents.to(device, dtype=dtype)
        else:
            init_audio_latents = torch.randn(
                init_video_latents.shape[0],
                audio_latent_channels,
                audio_latent_t,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + 1),
            )

        # ---- Scheduler ----
        # get_scheduler(name, steps, start_step, end_step, shift, device, ...) → (scheduler, timesteps, start_idx, end_idx)
        sample_scheduler, timesteps, _, _ = get_scheduler(
            scheduler,
            steps,
            0,      # start_step (controlled via denoise_strength below)
            -1,     # end_step (-1 = use all)
            shift,
            device,
            transformer.dim if hasattr(transformer, 'dim') else 5120,
            denoise_strength,
            sigmas=sigmas,
        )

        # Noise schedule for audio — mirror the video scheduler
        audio_sched = deepcopy(sample_scheduler)
        # Number of train timesteps (for boundary_timestep computation); default 1000
        num_train_timesteps = 1000

        # get_scheduler already handles denoise_strength by slicing timesteps.
        # start_step is always 0 (we iterate all returned timesteps).
        start_step = 0

        video_latents = init_video_latents.clone()
        aud_latents = init_audio_latents.clone()

        # Build the combined visual condition tensor once (mask + image_cond concatenated on channel dim)
        # This mirrors how WanVideoWrapper nodes_sampler.py prepares the I2V condition:
        #   image_cond = torch.cat([image_cond_mask, image_cond])  → [4+C, T, H, W]
        # Then unsqueeze(0) for batch dim: [1, 4+C, T, H, W]
        latent_condition = None
        if image_cond_latent is not None:
            img_cond = image_cond_latent.to(device, dtype=dtype)  # [C, T, H, W]
            if image_cond_mask is not None:
                msk = image_cond_mask.to(device, dtype=dtype)     # [4, T, H, W]
                img_cond_combined = torch.cat([msk, img_cond], dim=0)  # [4+C, T, H, W]
            else:
                # No explicit mask — zero out all but first latent frame (T2V/no-mask path)
                img_cond_combined = img_cond.clone()
                img_cond_combined[:, 1:] = 0.0
            latent_condition = img_cond_combined.unsqueeze(0)  # [1, 4+C, T, H, W]

        # ---- Stage setup ----
        # audio_dit and bridge are now plain dicts from their respective loaders
        audio_dit_model = audio_dit["model"]
        bridge_model = bridge["model"]
        bridge_condition_scale = bridge.get("condition_scale", 1.0)

        # video_dit_2 is an optional WANVIDEOMODEL patcher (same as stage-1 model)
        video_dit_2_transformer = None
        if video_dit_2 is not None:
            video_dit_2_transformer = video_dit_2.model.diffusion_model

        # Move to device
        audio_dit_model.to(device)
        bridge_model.to(device)
        if video_dit_2_transformer is not None:
            video_dit_2_transformer.to(device)

        transformer_options = copy.deepcopy(
            patcher.model_options.get("transformer_options", {})
        )

        boundary_timestep = boundary_ratio * num_train_timesteps
        cur_visual_dit = transformer
        switched = False

        progress_bar = ProgressBar(len(timesteps))

        total_steps = len(timesteps)
        for step_idx in range(total_steps):
            mm.throw_exception_if_processing_interrupted()

            t = timesteps[step_idx]
            is_last = step_idx == total_steps - 1
            audio_t = t  # same timestep schedule for audio

            # Stage switching: switch to stage-2 video DiT when timestep drops below boundary
            if not switched and video_dit_2_transformer is not None and t.item() < boundary_timestep:
                cur_visual_dit = video_dit_2_transformer
                switched = True

            # Build model input: [B, noise_C + cond_C, T, H, W]
            # Concatenate noisy video latents with the combined condition tensor
            # along the channel dimension — matching the WanVideoWrapper I2V convention.
            if latent_condition is not None:
                latent_model_input = torch.cat([video_latents, latent_condition], dim=1)
            else:
                latent_model_input = video_latents

            timestep_in = t.unsqueeze(0).to(dtype=torch.float32, device=device)
            audio_timestep_in = audio_t.unsqueeze(0).to(dtype=torch.float32, device=device)

            # ---- Positive pass ----
            noise_pred_pos_vid, noise_pred_pos_aud = mova_inference_single_step(
                visual_dit=cur_visual_dit,
                audio_dit=audio_dit_model,
                bridge=bridge_model,
                visual_latents=latent_model_input,
                audio_latents=aud_latents,
                context=positive_context.to(device, dtype=dtype),
                timestep=timestep_in,
                audio_timestep=audio_timestep_in,
                video_fps=video_fps,
                clip_fea=clip_fea.to(device, dtype=dtype) if clip_fea is not None else None,
                transformer_options=transformer_options,
                condition_scale=bridge_condition_scale,
                a2v_condition_scale=a2v_condition_scale,
                v2a_condition_scale=v2a_condition_scale,
                current_step=step_idx,
                last_step=is_last,
            )

            # ---- Negative pass for CFG ----
            if cfg != 1.0 and negative_context is not None:
                noise_pred_neg_vid, noise_pred_neg_aud = mova_inference_single_step(
                    visual_dit=cur_visual_dit,
                    audio_dit=audio_dit_model,
                    bridge=bridge_model,
                    visual_latents=latent_model_input,
                    audio_latents=aud_latents,
                    context=negative_context.to(device, dtype=dtype),
                    timestep=timestep_in,
                    audio_timestep=audio_timestep_in,
                    video_fps=video_fps,
                    clip_fea=clip_fea.to(device, dtype=dtype) if clip_fea is not None else None,
                    transformer_options=transformer_options,
                    condition_scale=bridge_condition_scale,
                    a2v_condition_scale=a2v_condition_scale,
                    v2a_condition_scale=v2a_condition_scale,
                    current_step=step_idx,
                    last_step=is_last,
                )
                # CFG for video
                noise_pred_vid = noise_pred_neg_vid.float() + cfg * (
                    noise_pred_pos_vid.float() - noise_pred_neg_vid.float()
                )
                # CFG for audio
                noise_pred_aud = noise_pred_neg_aud.float() + cfg * (
                    noise_pred_pos_aud.float() - noise_pred_neg_aud.float()
                )
            else:
                noise_pred_vid = noise_pred_pos_vid.float()
                noise_pred_aud = noise_pred_pos_aud.float()

            # ---- Scheduler step ----
            video_latents = sample_scheduler.step(
                noise_pred_vid, t, video_latents.float(), return_dict=False
            )[0].to(dtype)
            aud_latents = audio_sched.step(
                noise_pred_aud, audio_t, aud_latents.float(), return_dict=False
            )[0].to(dtype)

            progress_bar.update(1)

        # ---- Clean up MOVA models if force_offload ----
        if force_offload:
            audio_dit_model.to(offload_device)
            bridge_model.to(offload_device)
            if video_dit_2_transformer is not None:
                video_dit_2_transformer.to(offload_device)
            mm.soft_empty_cache()

        return ({"samples": video_latents}, aud_latents)


# ---------------------------------------------------------------------------
# Node 4: Decode audio latents to waveform
# ---------------------------------------------------------------------------

class MOVADecodeAudio:
    """
    Decodes MOVA audio latents to a raw waveform tensor using the DAC audio VAE.
    Output is a ComfyUI AUDIO dict: {"waveform": [B, 1, T], "sample_rate": int}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("MOVA_AUDIO_VAE", {}),
                "audio_latents": ("MOVA_AUDIO_LATENTS", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/MOVA"

    @torch.no_grad()
    def decode(self, audio_vae, audio_latents: torch.Tensor):
        vae_model = audio_vae["model"]
        sample_rate = audio_vae["sample_rate"]
        dtype = audio_vae["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae_model.to(device)

        # audio_latents: [B, latent_dim, T] — for continuous DAC, sample from posterior
        latents = audio_latents.to(device, dtype=dtype)

        # If continuous VAE, decode expects raw latent (after post_quant_conv)
        if vae_model.continuous:
            # latents may be DiagonalGaussianDistribution object or plain tensor
            # During inference we receive the denoised tensor directly.
            waveform = vae_model.decode(latents)  # [B, 1, audio_T]
        else:
            waveform = vae_model.decode(latents)  # [B, 1, audio_T]

        waveform = waveform.float().cpu()

        vae_model.to(offload_device)
        mm.soft_empty_cache()

        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        return (audio_out,)


# ---------------------------------------------------------------------------
# ComfyUI node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MOVAAudioDITLoader": MOVAAudioDITLoader,
    "MOVABridgeLoader": MOVABridgeLoader,
    "MOVAAudioVAELoader": MOVAAudioVAELoader,
    "MOVASampler": MOVASampler,
    "MOVADecodeAudio": MOVADecodeAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOVAAudioDITLoader": "MOVA Audio DiT Loader",
    "MOVABridgeLoader": "MOVA Bridge Loader",
    "MOVAAudioVAELoader": "MOVA Audio VAE Loader",
    "MOVASampler": "MOVA Sampler",
    "MOVADecodeAudio": "MOVA Decode Audio",
}
