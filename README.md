# Pose-to-Video

## Usage

To animate a `.pose` file into a video, run

```bash
pip install '.[pix2pix]'
pose_to_video --type=pix2pix --model=pix_to_pix/training/model.h5 --pose=assets/testing-reduced.pose --video=sign.mp4
```
Or including upscaling
```bash
pip install '.[pix2pix,upscaler]'
pose_to_video --type=pix2pix --model=pix_to_pix/training/model.h5 --pose=assets/testing-reduced.pose --video=sign.mp4 --upscale
```

Using controlnet:
```bash
pip install '.[controlnet]'
pose_to_video --type=controlnet --model=sign/sd-controlnet-mediapipe --pose=assets/testing-reduced.pose --video=sign.mp4
```

## Implementations

This repository includes multiple implementations.

### Conditional Implementation

- [pix_to_pix](pose_to_video/conditional/pix_to_pix) - Pix2Pix model for video generation
- [controlnet](pose_to_video/conditional/controlnet) - ControlNet model for video generation

### Unconditional Implementation (Controlled)

- [stylegan3](pose_to_video/unconditional/stylegan3) - StyleGAN3 model for video generation
- [mixamo](pose_to_video/unconditional/mixamo) - Mixamo 3D avatar

### Upscalers

- [simple-upscaler](pose_to_video/upscalers/simple) - Upscales 256x256 frames to 768x768

## Datasets

- [BIU-MG](data/BIU-MG) - Bar-Ilan University: Maayan Gazuli
- [SHHQ](data/SHHQ) - high-quality full-body human images

