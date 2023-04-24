import os

root = '/home/et21-xinghz/deep_learning_project'
datasets_root = os.path.join(root,'SODTR_1/data/Camouflaged_Object_Detection/COD10K')
cod_training_root = os.path.join(datasets_root, 'TrainDataset1')
chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')
pvtv2_checkpoint_dir = '/home/et21-xinghz/deep_learning_project/PVTv2_Seg/checkpoint/'