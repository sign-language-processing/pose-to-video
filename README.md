# Pose-to-Video

## Usage

To animate a `.pose` file into a video, run

```bash
pip install '.[pix2pix]'
wget https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media -O pix_to_pix.h5
pose_to_video --type=pix2pix --model=pix_to_pix.h5 --pose=assets/testing-reduced.pose --video=assets/outputs/pix2pix.mp4
# Or including upscaling
pip install '.[pix2pix,simple_upscaler]'
pose_to_video --type=pix2pix --model=pix_to_pix.h5 --pose=assets/testing-reduced.pose --video=assets/outputs/pix2pix-upscaled.mp4 --processors simple_upscaler
# Or including AnimateDiff
pip install '.[pix2pix,animatediff]'
pose_to_video --type=pix2pix --model=pix_to_pix.h5 --pose=assets/testing-reduced.pose --video=assets/outputs/pix2pix-animatediff.mp4 --processors animatediff
# Or including both!
pip install '.[pix2pix,simple_upscaler,animatediff]'
pose_to_video --type=pix2pix --model=pix_to_pix.h5 --pose=assets/testing-reduced.pose --video=assets/outputs/pix2pix-upscaled-animatediff.mp4 --processors simple_upscaler animatediff simple_upscaler
```

Using ControlNet:
```bash
pip install '.[controlnet]'
pose_to_video --type=controlnet --model=sign/sd-controlnet-mediapipe --pose=assets/testing-reduced.pose --video=assets/outputs/controlnet.mp4
# Or including AnimateDiff (Requiring more VRAM):
pip install '.[controlnet,animatediff]'
pose_to_video --type=controlnet --model=sign/sd-controlnet-mediapipe --pose=assets/testing-reduced.pose --video=assets/outputs/controlnet-animatediff.mp4 --processors animatediff
# Or also upscaling
pip install '.[controlnet,animatediff,simple_upscaler]'
pose_to_video --type=controlnet --model=sign/sd-controlnet-mediapipe --pose=assets/testing-reduced.pose --video=assets/outputs/controlnet-animatediff-upscaled.mp4 --processors animatediff simple_upscaler
```

## Implementations

This repository includes multiple implementations.

### Conditional Implementation

- [pix_to_pix](pose_to_video/conditional/pix_to_pix) - Pix2Pix model for video generation
- [controlnet](pose_to_video/conditional/controlnet) - ControlNet model for video generation

### Unconditional Implementation (Controlled)

- [stylegan3](pose_to_video/unconditional/stylegan3) - StyleGAN3 model for video generation
- [mixamo](pose_to_video/unconditional/mixamo) - Mixamo 3D avatar

### Post-processors

- [simple_upscaler](pose_to_video/processors/simple_upscaler) - Upscales 256x256 frames to 768x768
- [animatediff](pose_to_video/processors/animatediff) - Uses AnimateDiff for better temporal coherence

## Datasets

- [BIU-MG](data/BIU-MG) - Bar-Ilan University: Maayan Gazuli
- [SHHQ](data/SHHQ) - high-quality full-body human images

