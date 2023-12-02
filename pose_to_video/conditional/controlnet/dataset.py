import datasets
from PIL import Image
import numpy as np
import zipfile
from pathlib import Path
import argparse


class Pix2PixDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, frames_path: Path, poses_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames_path = frames_path
        self.poses_path = poses_path

    def _info(self):
        return datasets.DatasetInfo(
            dataset_name="pix2pix_hf",
            features=datasets.Features(
                {
                    "control_image": datasets.Image(),
                    "image": datasets.Image(),
                    "caption": datasets.Value(dtype='string', id=None)
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "frames_path": self.frames_path,
                    "poses_path": self.poses_path
                }
            )
        ]

    def _generate_examples(self, frames_path: Path, poses_path: Path):
        with zipfile.ZipFile(frames_path) as frames_zip, zipfile.ZipFile(poses_path) as poses_zip:
            frames_files = sorted(frames_zip.infolist(), key=lambda x: x.filename)
            poses_files = sorted(poses_zip.infolist(), key=lambda x: x.filename)

            assert len(frames_files) == len(poses_files)

            for i, (frame_info, pose_info) in enumerate(zip(frames_files, poses_files)):
                with poses_zip.open(pose_info) as pose_file, frames_zip.open(frame_info) as frame_file:
                    pose = Image.open(pose_file)
                    frame = Image.open(frame_file)

                    yield i, {
                        "control_image": pose,
                        "image": frame,
                        "caption": "Maayan Gazuli performing sign language in front of a green screen."
                    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, required=True)
    parser.add_argument("--poses-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    frames_path = Path(args.frames_path)
    poses_path = Path(args.poses_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    dataset = Pix2PixDataset(frames_path, poses_path)
    dataset.download_and_prepare(output_path)
    dataset.as_dataset().save_to_disk(output_path)
