# Adaptive Multi-factor Component Tree Loss for Topology-Aware Medical Image Segmentation

This repository implements topology-aware loss functions (clDice, CTree, multiCTree) for vessel segmentation tasks. It provides full training, evaluation, and reproducibility on datasets such as DRIVE, LESAV, and CREMI using UNet/ATTUNet architectures.

## Features

- Flexible loss: BCE+Dice (Baseline), clDice, CTree, and multiCTree
- Full training and evaluation pipeline for multiple datasets and folds
- Topological metrics: Dice Score, Accuracy, clDice, Δ#(TPCCs), #(FPCCs), FPCC size, etc.


## System Requirements

- Ubuntu 20.04+ (recommended)
- Python 3.8+
- CUDA-compatible GPU (for acceleration, optional)

## Dataset

In our paper, we conduct experiments on the public datasets including [DRIVE](https://drive.grand-challenge.org/), [LES-AV](https://figshare.com/articles/dataset/LES-AV_dataset/11857698?file=21732282), and [CREMI](https://cremi.org/data/). Please download each dataset and place it in the `./datasets/{dataset}/training/` directory. Store the input images in `./datasets/{dataset}/training/images/` and the ground truth labels in `./datasets/{dataset}/training/labels/`.

## Training

Run the following command:
```
python3 train.py \
    --dataset <DATASET_NAME> \
    --model <MODEL_NAME> \
    --loss <LOSS_FUNCTION> \
    --output <OUTPUT_DIR>
```
**Arguments:**

- `--dataset` — Name of dataset. Options: `DRIVE`, `LESAV`, or `CREMI`  
- `--model` — Name of model. Options: `UNet`, `ATTUNet`  
- `--loss` — Name of loss function. Options: `'bce+dice'`, `'ctree'`, `'multiCTree'`, `'cldice'`  
- `--output` — Directory to save the trained model

**Example**
```
python3 train.py \
    --dataset DRIVE \
    --model UNet \
    --loss multiCTree \
    --output multiCTree
```

## Inference

To run inference on a trained model, execute the following command:

```bash
python3 test.py \
    --dataset <DATASET_NAME> \
    --output <OUTPUT_NAME>
```

**Arguments**

`--dataset` — Name of the dataset to evaluate. Supported options: DRIVE, LESAV, CREMI \
`--output` — Name or identifier of the output folder containing model predictions and results

**Example**
```
python3 test.py \
    --dataset DRIVE \
    --output multiCTree
```

After inference, use the `rename.py` script to standardize filenames across folds:

```bash
python3 rename.py \
    --dataset <DATASET_NAME> \
    --root_dir <GROUND_TRUTH_DIR> \
    --pred_imgs_dir <PREDICTION_DIR> \
    --output_dir <RENAMED_OUTPUT_DIR>
```

Aggregate the renamed prediction outputs across folds:
```
python3 combine_all.py \
    --root_dir <RENAMED_OUTPUT_DIR> \
    --output_dir <FINAL_OUTPUT_DIR>
```

