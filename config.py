"""
Configuration File

Defines global configuration parameters including dataset paths and
pretrained model paths. Please modify the `root` variable to point to
your dataset root directory before use.
"""

import os

# Dataset root directory (modify this path according to your environment)
root = ''

# COD10K dataset root directory
datasets_root = os.path.join(root, 'COD10K')

# Training dataset path
cod_training_root = os.path.join(datasets_root, 'TrainDataset1')

# Testing dataset paths
chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')  # CHAMELEON dataset
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')            # CAMO dataset
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')        # COD10K test set
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')            # NC4K dataset

# PVTv2 pretrained model checkpoint directory
pvtv2_checkpoint_dir = 'PVTv2_Seg/checkpoint/'