#!/bin/bash

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

# Run the training script
python training/finetune_pixtral_12b.py --config configs/pixtral_12b_casters.yaml 