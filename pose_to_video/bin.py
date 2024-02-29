#!/usr/bin/env python

import argparse
import importlib

import cv2
from pose_format.pose import Pose
from pose_format.utils.generic import correct_wrists, reduce_holistic
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--video', required=True, type=str, help='path to output video file')
    parser.add_argument('--type', required=True, type=str, choices=['pix2pix', 'controlnet', 'mixamo', 'stylegan3'],
                        help='system to use')
    parser.add_argument('--model', required=True, type=str, help='model path to use')
    parser.add_argument('--processors', nargs='+', default=[])

    return parser.parse_args()


def resize_if_needed(frames, resolution):
    for frame in frames:
        if frame.shape[0] != resolution[0] or frame.shape[1] != resolution[1]:
            yield cv2.resize(frame, resolution, interpolation=cv2.INTER_NEAREST)
        else:
            yield frame


def main():
    args = get_args()

    print('Loading input pose ...')
    with open(args.pose, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())
        pose = reduce_holistic(pose)
        correct_wrists(pose)

    print('Generating video ...')
    video = None
    conditional_module_path = f"pose_to_video.conditional.{args.type}"
    try:
        module = importlib.import_module(conditional_module_path)
    except ModuleNotFoundError as e:
        if conditional_module_path not in str(e):
            raise e
        module = importlib.import_module(f"pose_to_video.unconditional.{args.type}")

    pose_to_video = module.pose_to_video
    frames: iter = pose_to_video(pose, args.model)

    for processor in args.processors:
        print(f"pose_to_video.processors.{processor}")
        module = importlib.import_module(f"pose_to_video.processors.{processor}")
        process = module.process
        if hasattr(module, 'INPUT_RESOLUTION'):
            frames = resize_if_needed(frames, module.INPUT_RESOLUTION)
        frames = process(frames)

    for frame in tqdm(frames):
        if video is None:
            print('Saving to disk ...')
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(filename=args.video,
                                    apiPreference=cv2.CAP_FFMPEG,
                                    fourcc=fourcc,
                                    fps=pose.body.fps,
                                    frameSize=(height, width))

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


if __name__ == '__main__':
    main()
    # python -m pose_to_video.bin --type=controlnet --model=sign/sd-controlnet-mediapipe --pose=assets/testing-reduced.pose --video=sign.mp4
