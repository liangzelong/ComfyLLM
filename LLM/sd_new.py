import sys

import numpy as np
import torch
from mmengine import MODELS, Config
from mmengine.registry import init_default_scope
from torchvision import utils

init_default_scope("mmagic")


class sd_vae_cfg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["AutoencoderKL", "DenoisingUnet"],),
                "subfolder": (["vae", "unet"],),
                "from_pretrained": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("Config",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        **kwargs,
    ):
        return (kwargs,)


class sd_vae:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("Config",),
            },
        }

    RETURN_TYPES = ("VAE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        cfg,
    ):
        print(cfg)

        vae = MODELS.build(cfg)
        return (vae,)


class sd_vael:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "latents": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        prompt,
        latents,
    ):
        vae_cfg = dict(type="AutoencoderKL", from_pretrained=prompt, subfolder="vae")

        vae = MODELS.build(vae_cfg)
        return (vae.module.decode(latents)["sample"],)


class sd_unet_cfg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (
                    [
                        "UNet2DConditionModel",
                    ],
                ),
                "subfolder": (
                    [
                        "unet",
                    ],
                ),
                "from_pretrained": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("Config",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        **kwargs,
    ):
        return (kwargs,)


class sd_uent:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("Config",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        cfg,
    ):
        print(cfg)
        unet = MODELS.build(cfg)
        return (unet,)


class sd_scheduler:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        prompt,
    ):
        from mmagic.models.diffusion_schedulers import EditDDIMScheduler

        sch_cfg = dict(
            # type="EditDDIMScheduler",
            variance_type="learned_range",
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            set_alpha_to_one=False,
            clip_sample=False,
        )

        sch = EditDDIMScheduler(**sch_cfg)
        return (sch,)


class sd_encoder_cfg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (
                    [
                        "ClipWrapper",
                    ],
                ),
                "clip_type": (["huggingface"],),
                "subfolder": (["text_encoder"],),
                "pretrained_model_name_or_path": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("Config",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        **kwargs,
    ):
        return (kwargs,)


class sd_encoder:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("Config",),
            },
        }

    RETURN_TYPES = ("CLIP",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        cfg,
    ):

        unet = MODELS.build(cfg)
        return (unet,)


class sd_sd_cfg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer": (["/home/lzl/temp/sd1_5/"],),
                "enable_xformers": ([True],),
            },
        }

    RETURN_TYPES = ("Config",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        **kwargs,
    ):
        scheduler = dict(
            type="EditDDIMScheduler",
            variance_type="learned_range",
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            set_alpha_to_one=False,
            clip_sample=False,
        )
        res = dict(
            scheduler=scheduler,
            test_scheduler=scheduler,
        )
        kwargs.update(res)

        return (kwargs,)


class sd_sd_meta:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_meta": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("META",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(
        self,
        input_meta,
    ):

        return (input_meta,)


class sd_sd:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("META",),
                "vae": ("VAE",),
                "unet": ("MODEL",),
                "text_encoder": ("CLIP",),
                "cfg": ("Config",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, prompt, vae, unet, text_encoder, cfg):
        # scheduler = dict(
        #     type="EditDDIMScheduler",
        #     variance_type="learned_range",
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     beta_start=0.00085,
        #     num_train_timesteps=1000,
        #     set_alpha_to_one=False,
        #     clip_sample=False,
        # )
        module = (
            MODELS.children["mmagic"]
            .module_dict["StableDiffusion"](
                vae=vae, text_encoder=text_encoder, unet=unet, **cfg
            )
            .cuda()
        )

        image = module.infer(prompt)["samples"][0]

        image = torch.tensor(np.array(image)[None]) / 255.0
        return (image,)


NODE_CLASS_MAPPINGS = {
    "sd_vae": sd_vae,
    "sd_uent": sd_uent,
    "sd_scheduler": sd_scheduler,
    "sd_encoder": sd_encoder,
    "sd_sd": sd_sd,
    "sd_vael": sd_vael,
    "sd_vae_cfg": sd_vae_cfg,
    "sd_encoder_cfg": sd_encoder_cfg,
    "sd_sd_cfg": sd_sd_cfg,
    "sd_uent_cfg": sd_unet_cfg,
    "sd_sd_meta": sd_sd_meta,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "sd_vae": "sd_vae",
    "sd_uent": "sd_uent",
    "sd_scheduler": "sd_scheduler",
    "sd_encoder": "sd_encoder",
    "sd_sd": "sd_sd",
    "sd_vael": "sd_vael",
    "sd_vae_cfg": "sd_vae_cfg",
    "sd_encoder_cfg": "sd_encoder_cfg",
    "sd_sd_cfg": "sd_sd_cfg",
    "sd_uent_cfg": "sd_unet_cfg",
    "sd_sd_meta": "sd_sd_meta",
}
