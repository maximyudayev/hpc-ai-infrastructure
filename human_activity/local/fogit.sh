#!/bin/bash -l
KERNEL=9
RECEPTIVE_FIELD=50
EPOCHS=1
SEGMENT=100

# export SLURM_JOB_NAME="train_$(date +'%d-%m-%y-%T')_fogit_stgcn_${KERNEL}_${RECEPTIVE_FIELD}_${EPOCHS}"
export SLURM_JOB_NAME="debug_$(date +'%d-%m-%y-%T')_fogit_rtstgcn_${KERNEL}_${EPOCHS}"
export SLURM_ARRAY_TASK_ID=13
export MASTER_ADDR="localhost"
export MASTER_PORT=42069

mv data/imu_fogit_ABCD/train/features/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* data/imu_fogit_ABCD/val/features
mv data/imu_fogit_ABCD/train/labels/$(printf '%0.3d' ${SLURM_ARRAY_TASK_ID})* data/imu_fogit_ABCD/val/labels

# conda activate ml

(>&1 echo "$@")
(>&1 echo 'starting computation')
# Execute and time the script
python3 ./main.py train \
    --batch_size "$(ls data/imu_fogit_ABCD/train/labels | wc -l)" \
    --kernel $KERNEL \
    --receptive_field $RECEPTIVE_FIELD \
    --segment $SEGMENT \
    --epochs $EPOCHS \
    --demo $(shuf -i 0-$(($(ls data/imu_fogit_ABCD/val/labels | wc -l)-1)) -n 3) \
    --config "config/imu_fogit_ABCD/realtime_local.json"

mv data/imu_fogit_ABCD/val/features/* data/imu_fogit_ABCD/train/features
mv data/imu_fogit_ABCD/val/labels/* data/imu_fogit_ABCD/train/labels

export SLURM_JOB_NAME=
export SLURM_ARRAY_TASK_ID=
