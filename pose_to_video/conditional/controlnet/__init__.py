import os
from typing import List

import numpy as np
from pose_format import Pose
from itertools import islice, chain
import torch
from PIL import Image

from pose_to_video.utils import batched


def get_pipeline(model_name: str):
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                    controlnet=controlnet,
                                                                    safety_checker=None, torch_dtype=torch.float16)

    # Instead of using Stable Diffusion's default PNDMScheduler,
    # we use one of the currently fastest diffusion model schedulers, called UniPCMultistepScheduler.
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # the pipeline automatically takes care of GPU memory management.
    pipe.enable_model_cpu_offload()

    # attention layer acceleration
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def translate_images(pipeline, pose_images: List[np.ndarray], img2img: Image) -> iter:
    batch = len(pose_images)

    prompt = "Maayan Gazuli performing sign language in front of a green screen. Black shirt, blue jeans."
    negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, transparent"
    num_inference_steps = 20
    strength = 1.0
    guidance_scale = 7.5

    pose_images = [Image.fromarray(img) for img in pose_images]

    output = pipeline(
        prompt=[prompt] * batch,
        image=[img2img] * batch,
        control_image=pose_images,
        negative_prompt=[negative_prompt] * batch,
        generator=[torch.Generator().manual_seed(42) for _ in range(batch)],
        num_inference_steps=num_inference_steps,
        strength=strength,
        controlnet_conditioning_scale=strength,
        guidance_scale=guidance_scale,
    )

    for image in output.images:
        yield np.array(image)


def get_rgb_frames(pose: Pose) -> iter:
    import cv2
    from pose_format.pose_visualizer import PoseVisualizer

    visualizer = PoseVisualizer(pose, thickness=1)
    for pose_img_bgr in visualizer.draw():
        yield cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)


def pose_to_video(pose: Pose, model: str = "sign/sd-controlnet-mediapipe", batch_size=32) -> iter:
    pipeline = get_pipeline(model)

    # Scale pose to 512x512
    scale = 512
    scale_w = pose.header.dimensions.width / scale
    scale_h = pose.header.dimensions.height / scale
    pose.body.data /= np.array([scale_w, scale_h, 1])
    pose.header.dimensions.width = pose.header.dimensions.height = scale

    img2img_init = Image.new('RGB', (512, 512), '#3BAE3E')
    frames = get_rgb_frames(pose)

    # First batch includes only one frame, for the appearance of the initial image
    first_batch = [next(frames)]
    for frames_batch in chain([first_batch], batched(frames, batch_size)):
        images = list(translate_images(pipeline, frames_batch, img2img_init))
        yield from images

        # Update the initial image for the next batch
        img2img_init = Image.fromarray(images[-1])
