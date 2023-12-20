#!/bin/bash

#SBATCH --job-name=train-controlnet-hf
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=controlnet-job.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load gpu
module load cuda

module load anaconda3
source activate diffusers

# Download the data
DATA_DIR="/data/$(whoami)"
[ ! -f "$DATA_DIR/GreenScreen/mp4/Maayan_1/CAM3_norm.mp4" ] && \
wget --no-clobber --convert-links --random-wait \
    -r -p --level 3 -E -e robots=off --adjust-extension -U mozilla \
    "https://nlp.biu.ac.il/~amit/datasets/GreenScreen/" \
    -P "$DATA_DIR"

# Process the data
PROCESSED_DATA_DIR="/scratch/$(whoami)/GreenScreen"
mkdir -p $PROCESSED_DATA_DIR

[ ! -f "$PROCESSED_DATA_DIR/frames512.zip" ] && \
pip install pose-format mediapipe && \
python ../../../data/BIU-MG/video_to_images.py \
    --input_video="$DATA_DIR/GreenScreen/mp4/Maayan_1/CAM3_norm.mp4" \
    --input_pose="$DATA_DIR/GreenScreen/mp4/Maayan_1/CAM3.holistic.pose" \
    --output_path="$PROCESSED_DATA_DIR/frames512.zip" \
    --pose_output_path="$PROCESSED_DATA_DIR/poses512.zip" \
    --resolution=512

# Install dependencies
pip install diffusers transformers accelerate xformers wandb datasets argparse torchvision huggingface-hub
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Convert to huggingface dataset
HF_DATASET_DIR="$PROCESSED_DATA_DIR/huggingface"
mkdir -p $HF_DATASET_DIR

[ ! -f "$HF_DATASET_DIR/dataset_dict.json" ] && \
python dataset.py --frames-path="$PROCESSED_DATA_DIR/frames512.zip" \
    --poses-path="$PROCESSED_DATA_DIR/poses512.zip" \
    --output-path="$HF_DATASET_DIR"

# Verify GPU with PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

CURRENT_DIR=$(pwd)

CACHE_DIR="/scratch/$(whoami)/huggingface/cache"
mkdir -p $CACHE_DIR

# Download diffusers repository if not exists
[ ! -d "diffusers" ] && \
git clone https://github.com/huggingface/diffusers.git

# Install diffusers
pip install ./diffusers

OUTPUT_DIR="/scratch/$(whoami)/models/sd-controlnet-mediapipe"
mkdir -p $OUTPUT_DIR

! accelerate launch diffusers/examples/controlnet/train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-openpose" \
 --output_dir="$OUTPUT_DIR" \
 --train_data_dir="$HF_DATASET_DIR" \
 --conditioning_image_column=control_image \
 --image_column=image \
 --caption_column=caption \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./validation/landmarks1.png" "./validation/landmarks2.png" "./validation/landmarks3.png" \
 --validation_prompt "Maayan Gazuli performing sign language in front of a green screen." "Maayan Gazuli performing sign language in front of a green screen." "Barack Obama performing sign language in front of a green screen." \
 --train_batch_size=4 \
 --num_train_epochs=20 \
 --tracker_project_name="sd-controlnet-mediapipe" \
 --hub_model_id="sign/sd-controlnet-mediapipe" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=1000 \
 --report_to wandb \
 --push_to_hub





# sbatch train.sh
# srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1 --constraint=GPUMEM80GB --mem=32G bash -l
# srun --pty -n 1 -c 2 --time=02:00:00 --mem=32G bash -l
# conda activate diffusers
# cd /home/amoryo/sign-language/pose-to-video/pose_to_video/conditional/controlnet

# srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1 --mem=32G bash -l
# srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1 --constraint=GPUMEM80GB --mem=32G bash -l
# cd /home/amoryo/sign-language/signwriting-illustration/signwriting_illustration/controlnet/ControlNet
# conda activate controlnet
# python predict.py --data-path="../../../train" --checkpoint-path="/home/amoryo/sign-language/signwriting-illustration/signwriting_illustration/controlnet/ControlNet/lightning_logs/version_6635784/checkpoints/epoch=499-step=88999.ckpt"
# python predict.py --data-path="../../../train" --checkpoint-path="/home/amoryo/sign-language/signwriting-illustration/signwriting_illustration/controlnet/ControlNet/lightning_logs/version_6645717/checkpoints/epoch=999-step=198999.ckpt"

# Download lora:
# wget https://civitai.com/api/download/models/{modelVersionId} --content-disposition