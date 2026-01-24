"""
Nodes for combining SCAIL pose/reference with WanAnimate face adapter.
This allows using the merged SCAIL+WanAnimate model with both control systems.
"""

import torch
from ..utils import log
import comfy.model_management as mm
from comfy.utils import common_upscale

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


class WanVideoAddWanAnimateFaceEmbeds:
    """
    Add WanAnimate face embeddings to SCAIL or other image embeds.
    This allows using the face adapter from WanAnimate with SCAIL pose control.
    
    The face images should be cropped face images at 512x512 resolution.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeds": ("WANVIDIMAGE_EMBEDS",),
                "face_images": ("IMAGE", {"tooltip": "Cropped face images, will be resized to 512x512"}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the face adapter effect"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Add WanAnimate face images to use with a merged SCAIL+WanAnimate model"

    def add(self, embeds, face_images, face_strength):
        updated = dict(embeds)
        
        # Resize face images to 512x512 (required by WanAnimate face encoder)
        face_images = face_images[..., :3]
        if face_images.shape[1] != 512 or face_images.shape[2] != 512:
            resized_face_images = common_upscale(
                face_images.movedim(-1, 1), 512, 512, "lanczos", "center"
            ).movedim(0, 1)
        else:
            resized_face_images = face_images.permute(3, 0, 1, 2)  # B, C, T, H, W
        
        # Normalize to [-1, 1] and add batch dimension
        resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
        resized_face_images = resized_face_images.to(offload_device, dtype=torch.float32)
        
        log.info(f"WanAnimate face images shape: {resized_face_images.shape}")
        
        updated["face_pixels"] = resized_face_images
        updated["face_strength"] = face_strength
        
        return (updated,)


class WanVideoAddCombinedSCAILWanAnimateEmbeds:
    """
    Combined node for adding both SCAIL pose control AND WanAnimate face embeddings.
    Use this when you have a merged SCAIL+WanAnimate model and want to use both features.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeds": ("WANVIDIMAGE_EMBEDS",),
                "vae": ("WANVAE", {"tooltip": "VAE model for encoding pose"}),
            },
            "optional": {
                # SCAIL Reference
                "scail_ref_image": ("IMAGE", {"tooltip": "Reference image for SCAIL identity preservation"}),
                "scail_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of SCAIL reference"}),
                "scail_ref_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scail_ref_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scail_ref_clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image for SCAIL"}),
                
                # SCAIL Pose
                "scail_pose_images": ("IMAGE", {"tooltip": "Pose images for SCAIL pose control"}),
                "scail_pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of SCAIL pose control"}),
                "scail_pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scail_pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WanAnimate Face
                "wananim_face_images": ("IMAGE", {"tooltip": "Face images for WanAnimate face adapter"}),
                "wananim_face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of WanAnimate face adapter"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Combined node for SCAIL pose/reference + WanAnimate face control"

    def process(
        self, embeds, vae,
        # SCAIL Reference
        scail_ref_image=None, scail_ref_strength=1.0, 
        scail_ref_start_percent=0.0, scail_ref_end_percent=1.0, scail_ref_clip_embeds=None,
        # SCAIL Pose
        scail_pose_images=None, scail_pose_strength=1.0,
        scail_pose_start_percent=0.0, scail_pose_end_percent=1.0,
        # WanAnimate Face
        wananim_face_images=None, wananim_face_strength=1.0,
    ):
        updated = dict(embeds)
        
        # ============ SCAIL Reference ============
        if scail_ref_image is not None:
            vae.to(device)
            ref_image_in = (scail_ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
            ref_latent = vae.encode([ref_image_in], device, tiled=False)[0]
            log.info(f"SCAIL ref_latent shape: {ref_latent.shape}")

            ref_mask = torch.ones_like(ref_latent[:4])
            ref_latent = torch.cat([ref_latent, ref_mask], dim=0)
            vae.to(offload_device)

            updated.setdefault("scail_embeds", {})
            updated["scail_embeds"]["ref_latent_pos"] = ref_latent * scail_ref_strength
            updated["scail_embeds"]["ref_latent_neg"] = torch.zeros_like(ref_latent)
            updated["scail_embeds"]["ref_start_percent"] = scail_ref_start_percent
            updated["scail_embeds"]["ref_end_percent"] = scail_ref_end_percent
            if scail_ref_clip_embeds is not None:
                updated["clip_context"] = scail_ref_clip_embeds.get("clip_embeds", None)
        
        # ============ SCAIL Pose ============
        if scail_pose_images is not None:
            vae.to(device)
            pose_images_in = (scail_pose_images[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
            pose_latent = vae.encode([pose_images_in], device, tiled=False)[0]
            pose_mask = torch.ones_like(pose_latent[:4])
            pose_latent = torch.cat([pose_latent, pose_mask], dim=0)
            log.info(f"SCAIL pose_latent shape: {pose_latent.shape}")
            vae.to(offload_device)

            updated.setdefault("scail_embeds", {})
            updated["scail_embeds"]["pose_latent"] = pose_latent
            updated["scail_embeds"]["pose_strength"] = scail_pose_strength
            updated["scail_embeds"]["pose_start_percent"] = scail_pose_start_percent
            updated["scail_embeds"]["pose_end_percent"] = scail_pose_end_percent
        
        # ============ WanAnimate Face ============
        if wananim_face_images is not None:
            face_images = wananim_face_images[..., :3]
            if face_images.shape[1] != 512 or face_images.shape[2] != 512:
                resized_face_images = common_upscale(
                    face_images.movedim(-1, 1), 512, 512, "lanczos", "center"
                ).movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2)
            
            resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
            resized_face_images = resized_face_images.to(offload_device, dtype=torch.float32)
            
            log.info(f"WanAnimate face images shape: {resized_face_images.shape}")
            
            updated["face_pixels"] = resized_face_images
            updated["face_strength"] = wananim_face_strength
        
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddWanAnimateFaceEmbeds": WanVideoAddWanAnimateFaceEmbeds,
    "WanVideoAddCombinedSCAILWanAnimateEmbeds": WanVideoAddCombinedSCAILWanAnimateEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddWanAnimateFaceEmbeds": "WanVideo Add WanAnimate Face Embeds",
    "WanVideoAddCombinedSCAILWanAnimateEmbeds": "WanVideo Add Combined SCAIL+WanAnimate Embeds",
}
