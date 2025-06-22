README - Deep Learning Models for Spinal Ependymoma

This repository contains the official implementation of our manuscript titled "Non-invasive Prediction of Ki-67 and p53 Biomarkers in Spinal Ependymoma via Deep Learning: Using Multimodal MRI and Clinical Data". The project integrates segmentation and classification modules for spinal ependymoma analysis from MRI images.

1. Repository Structure

- `segformer.py`: SegFormer-B2-based segmentation pipeline for SAG and TRA T2WI MRI images.
- `nn ensemble.py`: Definition and training of the custom MLP-LightGBM ensemble model (LGBMNet).
- `nn ensemble plot.py`: Graphviz-based visualization of model architecture.
- `README.md`: This guidance file.

2. Environment Setup

Recommended Python version: 3.8  
Install dependencies:
```
pip install -r requirements.txt
```
Note: PaddleSeg and Graphviz must be correctly installed. For GPU usage, CUDA 11.6 and PyTorch >=1.13 are recommended.

3. Segmentation Module (SegFormer-B2)

Run `segformer.py` to train or evaluate the segmentation model.
- Input: Preprocessed SAG and TRA T2WI MRI slices with manual annotations.
- Output: Binary lesion masks.
Training settings:
- Optimizer: Adam
- Batch size: 4
- Epochs: 100
- Loss: Dice loss
- Augmentation: rotation, flipping, scaling, translation
- Checkpoints saved by best validation Dice score

4. Classification Module (LGBMNet)

Run `nn ensemble.py` to train the ensemble classifier using radiomic and clinical features.
Pipeline:
1. Train base MLP on selected features
2. Extract prediction probabilities
3. Combine with original features
4. Train LightGBM on enhanced feature set
Evaluation includes: accuracy, precision, recall, F1-score, and AUC.

5. Model Architecture Visualization

Run `nn ensemble plot.py` to generate a visual graph of the model architecture. Output is in PNG format using Graphviz.

6. Data Format & Reproducibility

- All datasets are expected in NumPy array or PNG image format
- Please organize folders as: `data/train`, `data/val`, `data/test` with matching labels
- For reproducibility, we provide full training code, seed fixation, and logs in the repository.

7. Contact

For any issues or questions, please contact the corresponding author or open an issue in this repository.
