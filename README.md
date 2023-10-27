# SARNet
## prediction maps and pretrained model
1. The prediction map of our method can be downloaded at https://drive.google.com/drive/folders/1rZ9IrbFz4dggik1vnK53PnNNy7l-1X_r?usp=sharing.
2. The pretrained model of our model can be downloaded at https://drive.google.com/drive/folders/1n80O0RIAe4KT08SZG-UK6HsWgD_qouoQ?usp=sharing.
## environment
1. python = Python 3.8.13
2. others packages can be found at requirement.txt
## start
git clone https://github.com/Haozhe-Xing/SARNet.git

conda create --name myenv python=3.8.13

conda activate myenv

pip install -r requirements.txt

## New: about the features map visualization！ 
![image](https://github.com/Haozhe-Xing/SARNet/assets/41740840/5a916d02-8f6e-4437-b446-e5fff48e6c49)

if you want to visualize your features maps like the above imgages, you can user the code in folder "display_heatmpas".
![image](https://github.com/Haozhe-Xing/SARNet/assets/41740840/73971d14-da0f-4eb5-808f-67652d9b4719)

if you want to visualize your model predictions like the above imgages, you can user the code in folder "display_heatmpas/combine.py".
