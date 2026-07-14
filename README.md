# FoundDiff
Official implementation of "FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising" 

## Approach
![](figs/network.png)


## Updates
Feb, 2026: Upload DA-CLIP and FoundDiff model weight (https://drive.google.com/drive/folders/1B33XyPqC9KkmzmfrCq20-7Xxuf-23PMc?usp=sharing)   
July, 2025: initial commit.  


## Data Preparation
The 2016 AAPM-Mayo dataset can be downloaded from: [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/) (B30 kernel)  
The 2020 AAPM-Mayo dataset can be downloaded from: [cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)   


## Requirements
```
- Linux Platform
- torch==1.12.1+cu113 # depends on the CUDA version of your machine
- torchvision==0.13.1+cu113
- Python==3.8.0
- numpy==1.22.3
```

## Traning and & Inference


#### Training:  
```
CUDA_VISIBLE_DEVICES=1 python train.py --name FoundDiff --is_train --train_num_steps 400000
```

#### Inference & testing:
Put DA-CLIP.pth in src/DA-Diff.py and model-400.pt in checkpoints/FoundDiff/sample  
```
CUDA_VISIBLE_DEVICES=4 python train.py --name FoundDiff --epoch 400 --dataset 2020_seen
```
Please refer to options files for more setting.



#### .nii.gz denoising utility:
## Step 0: Pick a folder begin:
cd RootOfYourInstallation

## Step 1: Github Clone FoundDiff
'''git clone https://github.com/hao1635/FoundDiff.git /n
cd FoundDiff'''

## Step 2: Create an environment
'''conda create -n FoundDiff python=3.7.9
conda activate FoundDiff'''

## Step 3: Download folder of Necessary Documents
'''mv RootToNecessaryDocuments/NecessaryDocument RootOfYourInstallation 
pip install -r requirements.txt'''

## Step 4: Download official pretrained model from github:
https://drive.google.com/drive/folders/1B33XyPqC9KkmzmfrCq20-7Xxuf-23PMc?usp=sharing

## Step 5: Put DA-CLIP.pth in src/DA-Diff.py and model-400.pt in checkpoints/FoundDiff/sample 


## Step 6: run following prompt at root folder
'''conda activate FoundDiff     
python denoise_folder.py \    
  --in_dir  data/mydata \    
  --out_dir data/Output \    
  --batch_size 4'''

#### Official website of FoundDiff: 
https://github.com/hao1635/FoundDiff/blob/main/README.md

#### Official paper for FoundDiff: 
'''FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising[1].    
[1]Z. Chen et al., “FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising,” IEEE Transactions on Medical Imaging, pp. 1–1, 2026, doi: 10.1109/tmi.2026.3698474.'''

