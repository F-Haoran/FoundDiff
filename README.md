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



```markdown
### `.nii.gz` Denoising Utility

**Step 0: Choose your working directory**
Navigate to the root folder where you want to install the project.
```bash
cd /path/to/RootOfYourInstallation

```

**Step 1: Clone the FoundDiff repository**
Clone the official GitHub repository and navigate into it.

```bash
git clone [https://github.com/hao1635/FoundDiff.git](https://github.com/hao1635/FoundDiff.git)
cd FoundDiff

```

**Step 2: Create and activate the Conda environment**
Set up a dedicated Python environment to prevent dependency conflicts.

```bash
conda create -n FoundDiff python=3.7.9 -y
conda activate FoundDiff

```

**Step 3: Move necessary documents and install dependencies**
Move your required documents into the installation folder, then install the Python requirements.

```bash
mv /path/to/NecessaryDocuments /path/to/RootOfYourInstallation 
pip install -r requirements.txt

```

**Step 4: Download the official pre-trained models**
Download the required model weights from the official Google Drive link:
🔗 [FoundDiff Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/1B33XyPqC9KkmzmfrCq20-7Xxuf-23PMc?usp=sharing)

**Step 5: Place the model weights in the correct directories**
Once downloaded, move the specific weight files to their corresponding folders:

* Place `DA-CLIP.pth` into the `src/` directory.
* Place `model-400.pt` into the `checkpoints/FoundDiff/sample/` directory.

**Step 6: Run the denoising script**
Ensure your environment is active, then run the batch inference script from the root folder.

```bash
conda activate FoundDiff     
python denoise_folder.py \    
  --in_dir data/mydata \    
  --out_dir data/Output \    
  --batch_size 4

```

---

### References & Links

* **Official Repository:** [FoundDiff on GitHub](https://github.com/hao1635/FoundDiff/blob/main/README.md)
* **Official Paper:** > Z. Chen et al., *"FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising,"* IEEE Transactions on Medical Imaging, pp. 1–1, 2026, doi: [10.1109/tmi.2026.3698474](https://www.google.com/search?q=https://doi.org/10.1109/tmi.2026.3698474).

```

```

