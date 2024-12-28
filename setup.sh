#!/bin/bash

# Prepare the dataset
python scripts/prepare_dataset.py

# Prepare the 33k dataset
python scripts/prepare_dataset_33k.py

# Train the model
torchrun --nproc_per_node=1 main.py configs/stage1/m0.py --out ../output/stage1/m0/update_last.pth

# Prepare final weights for submission
python scripts/prepare_gemma2_for_submission.py