from typing import Iterable

import numpy as np
import torch
from PIL import Image
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter

from pose_to_video.utils import batched

# SD 1.5 based finetuned model
STABLE_DIFFUSION_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
ANIMATE_DIFF_MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
INPUT_RESOLUTION = (512, 512)


def get_pipeline():
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained(ANIMATE_DIFF_MOTION_ADAPTER_ID, torch_dtype=torch.float16)

    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(STABLE_DIFFUSION_MODEL_ID,
                                                           motion_adapter=adapter,
                                                           torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    scheduler = DDIMScheduler.from_pretrained(
        STABLE_DIFFUSION_MODEL_ID,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe


def process_frames(pipeline: AnimateDiffVideoToVideoPipeline, frames: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    frames = [Image.fromarray(frame) for frame in frames]

    output = pipeline(
        video=frames,
        prompt="white woman using sign language, green screen, black shirt, blue jeans. detailed, studio lighting, detailed iris, symmetrical circular eyes, natural color, dramatic highlights, light, shadow, relaxed, detailed skin, symmetrical ears and face shape, 90mm lens, by Martin Schoeller",
        negative_prompt="bad quality, worse quality, disfigured, cartoon, painting, doll, blurry, grainy, black and white, broken, cross-eyed, undead, photoshopped, overexposed, underexposed, rash, sunburn, mutated, alien, unthinking, unfeeling, unrealistic, cramped, flexing, soft lens, hard light, wobbly iris, square iris, flat iris, edge of iris following, surreal, surrealist, fiction, Wax sculpture, caricature, frame, fish, Neanderthal, reptile, rings, jewelry, cataracts, dumb eyes, creepy, zombie",
        guidance_scale=7.5,
        num_inference_steps=25,
        strength=0.3,
        generator=torch.Generator("cpu").manual_seed(42),
    )

    for image in output.frames[0]:
        yield np.array(image)


def process(frames: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    pipeline = get_pipeline()

    # Process frames at batches of 32 frames max (model limitation)
    for frames in batched(frames, 32):
        yield from process_frames(pipeline, frames)
