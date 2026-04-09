# CNN Food101 Classification

This project contains two Jupyter notebooks for transfer learning and fine-tuning on the Food101 dataset using PyTorch.

## Project Structure

- `partA_transfer_learning.ipynb`: trains and compares frozen-backbone models (`GoogLeNet`, `MobileNetV3`, `ResNet-50`) with a custom classification head.
- `partB_fine_tuning.ipynb`: loads the best Part A checkpoint and runs controlled fine-tuning experiments by unfreezing deeper backbone layers.
- `checkpoints/`: stores best model weights (`*.pth`) produced during training.
- `data/food-101/`: dataset directory containing `images/` and `meta/`.

## Workflow

1. Run `partA_transfer_learning.ipynb` end-to-end.
2. Identify the best model from Part A and ensure its checkpoint is in `checkpoints/`.
3. Set `BEST_MODEL_NAME` in `partB_fine_tuning.ipynb`.
4. Run Part B to compare unfreezing strategies and evaluate on the test set.

## Environment

- Python 3.10+ recommended
- PyTorch + torchvision
- matplotlib, seaborn, pandas, numpy
- pytorch-ignite

Install dependencies with pip (example):

```bash
pip install torch torchvision matplotlib seaborn pandas numpy pytorch-ignite tqdm
```

## Dataset Notes

- The notebooks are configured to run both locally and in Google Colab.
- Local run expects:
  - dataset at `data/food-101/`
  - checkpoints in `checkpoints/`
- Colab paths are already configured in notebook cells for Drive-based dataset/checkpoint persistence.

## Reproducibility

Both notebooks set random seeds (`SEED = 42`) for reproducible data shuffling and training behavior across runs.
