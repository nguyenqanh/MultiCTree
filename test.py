import csv
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk

from unet_model import Unet
from attnunet_model import ATTUNet
from utils import paste_and_save, eval_print_metrics
from data_loader_cv import load_dataset_cv
from losses.clDice.cldice_metric import cldice

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (e.g., 0, 1, ...), or "cpu"')
    parser.add_argument('--model', default='Unet', choices=['Unet', 'ATTUNet'])
    parser.add_argument('--folds', nargs='+', default=['fold_1_1', 'fold_1_2', 'fold_2_1', 'fold_2_2'])
    parser.add_argument('--dataset', default='DRIVE', choices=['DRIVE', 'LESAV', 'CREMI'])
    parser.add_argument('--model_dir', type=str, default='./checkpoint')
    parser.add_argument('--pred_dir', type=str, default='./pred_imgs')
    parser.add_argument('--output', type=str, default='baseline', help='Subfolder for saving under each dataset/fold')
    return parser.parse_args()

def to_notebook_rgb(mask):
    """
    Converts a [1, H, W], [H, W], or [H, W, 1] mask to [H, W, 3] uint8 binary (0/255).
    """
    mask = np.squeeze(mask)
    # Binarize to 0/255 (to match cv2.imread's result)
    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    if mask_bin.ndim == 2:
        mask_rgb = np.stack([mask_bin]*3, axis=-1)
    elif mask_bin.ndim == 3 and mask_bin.shape[-1] == 1:
        mask_rgb = np.concatenate([mask_bin]*3, axis=-1)
    elif mask_bin.ndim == 3 and mask_bin.shape[-1] == 3:
        mask_rgb = mask_bin
    else:
        raise ValueError(f"Unexpected mask shape: {mask_bin.shape}")
    return mask_rgb

def labeling_notebook(img):
    # img: [H, W], np.uint8, 0/255
    ConnectedComponentImageFilter = sitk.ConnectedComponentImageFilter()
    ConnectedComponentImageFilter.FullyConnectedOn()
    img = sitk.GetImageFromArray(img.astype(np.int8))
    label = ConnectedComponentImageFilter.Execute(img)
    label = sitk.GetArrayFromImage(label)
    cc = np.unique(label)
    return label, len(cc) - 1

def topological_metrics(gt_mask, pr_mask):
    # Convert to [H,W,3] with 0/255
    gt_rgb = to_notebook_rgb(gt_mask)
    pr_rgb = to_notebook_rgb(pr_mask)
    gt_bin = gt_rgb[:, :, 0]
    pr_bin = pr_rgb[:, :, 0]
    gt_label, gt_cc = labeling_notebook(gt_bin)
    pr_label, pr_cc = labeling_notebook(pr_bin)
    labels = np.unique(pr_label)
    labels = labels[labels != 0]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    TP_flags = np.zeros(len(labels), dtype=bool)
    gt_mask_for_overlap = gt_label != 0
    for label in labels:
        pr_mask_l = pr_label == label
        if np.any(pr_mask_l & gt_mask_for_overlap):
            TP_flags[label_to_index[label]] = True
    FP_labels = [label for label in labels if not TP_flags[label_to_index[label]]]
    FP_mask = np.isin(pr_label, FP_labels)
    delta_CCs = abs(gt_cc - pr_cc)
    delta_TPCCs = abs(np.sum(TP_flags) - gt_cc)
    FPCCs = len(FP_labels)
    FPCC_size = np.sum(FP_mask)
    return {
        "delta_CC": delta_CCs,
        "delta_TPCC": delta_TPCCs,
        "FPCC": FPCCs,
        "FPCC_size": FPCC_size
    }


def labeling(img):
    img = img.squeeze(0)
    img_array = img.detach().cpu().numpy().astype(np.uint8)
    ConnectedComponentImageFilter = sitk.ConnectedComponentImageFilter()
    ConnectedComponentImageFilter.FullyConnectedOn()
    img_sitk = sitk.GetImageFromArray(img_array)
    label = ConnectedComponentImageFilter.Execute(img_sitk)
    label = sitk.GetArrayFromImage(label)
    cc = np.unique(label)
    return len(cc[cc > 0])

def calculate_betti_errors(true_mask, pred_mask):
    pred_components = labeling(pred_mask)
    true_components = labeling(true_mask)
    b0_error = abs(pred_components - true_components)
    return b0_error

def calculate_accuracy(true_mask, pred_mask):
    true_mask = true_mask.float()
    pred_mask = (pred_mask > 0.5).float()
    TP = torch.sum((pred_mask == 1) & (true_mask == 1)).float()
    TN = torch.sum((pred_mask == 0) & (true_mask == 0)).float()
    FP = torch.sum((pred_mask == 1) & (true_mask == 0)).float()
    FN = torch.sum((pred_mask == 0) & (true_mask == 1)).float()
    denominator = (TP + TN + FP + FN)
    if denominator == 0:
        return 0.0
    accuracy = (TP + TN) / denominator
    print(f"Final accuracy: {accuracy.item()}")
    return accuracy.item()

def dice_score(true_mask, pred_mask):
    intersection = (true_mask * pred_mask).sum()
    union = true_mask.sum() + pred_mask.sum()
    if union.item() == 0:
        return 1.0
    return (2.0 * intersection / union).item()

def get_filenames(dataset, fold):
    base = f"datasets/{dataset}_dataset/{fold}"
    return dict(
        train_image=f"{base}/train_image.npy",
        train_label=f"{base}/train_label.npy",
        train_mask=f"{base}/train_mask.npy",
        val_image=f"{base}/val_image.npy",
        val_label=f"{base}/val_label.npy",
        val_mask=f"{base}/val_mask.npy",
        test_image=f"{base}/test_image.npy",
        test_label=f"{base}/test_label.npy",
        test_mask=f"{base}/test_mask.npy",
    )

def model_test(net, fold, dataset, pred_dir, output, batch_size=2, metrics_csv="metrics.csv"):
    file_dict = get_filenames(dataset, fold)
    # Load .npy files
    x_test = np.load(file_dict['test_image'])
    y_test = np.load(file_dict['test_label'])
    # Mask only for DRIVE
    if dataset == 'DRIVE':
        m_test = np.load(file_dict['test_mask'])
    else:
        m_test = np.ones_like(y_test)
    num_samples = x_test.shape[0]
    fold_pred_dir = os.path.join(pred_dir, dataset, output, fold)
    os.makedirs(fold_pred_dir, exist_ok=True)

    with open(os.path.join(fold_pred_dir, metrics_csv), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "Precision", "Recall", "F1-Score", "AUC", "Dice Score", "Accuracy", "clDice", "B0 Error",
            "delta_CC", "delta_TPCC", "FPCC", "FPCC_size"
        ])
        for img_idx in range(num_samples):
            print(f"[*] Predicting for image {img_idx + 1}/{num_samples} (fold {fold}, dataset {dataset})")
            bat_img = torch.Tensor(x_test[img_idx: img_idx + 1, :, :, :])
            bat_label = torch.Tensor(y_test[img_idx: img_idx + 1, 0: 1, :, :])
            bat_mask = torch.Tensor(m_test[img_idx: img_idx + 1, 0: 1, :, :])

            bat_pred = net(bat_img)
            bat_pred_class = (bat_pred > 0.5).float() * bat_mask

            metrics = eval_print_metrics(bat_label, bat_pred, bat_mask)
            dice = dice_score(bat_label, bat_pred_class)
            accuracy = calculate_accuracy(bat_label, bat_pred_class)
            clDice_val = cldice.clDice(bat_pred_class.squeeze(0).squeeze(0).numpy(), bat_label.squeeze(0).squeeze(0).numpy())
            B0_error = calculate_betti_errors(bat_label, bat_pred_class)
            bat_label_np = bat_label.squeeze().cpu().numpy()
            bat_pred_class_np = bat_pred_class.squeeze().cpu().numpy()
            topometrics = topological_metrics(bat_label_np, bat_pred_class_np)
            writer.writerow(
                [img_idx + 1]
                + list(metrics)
                + [dice]
                + [accuracy]
                + [clDice_val]
                + [B0_error]
                + [
                    topometrics['delta_CC'],
                    topometrics['delta_TPCC'],
                    topometrics['FPCC'],
                    topometrics['FPCC_size']
                ]
            )
            paste_and_save(
                bat_img, bat_label, bat_pred, bat_pred_class,
                batch_size=1, cur_bat_num=img_idx + 1,
                save_img=fold_pred_dir
            )
    print(f"[+] Metrics saved to {os.path.join(fold_pred_dir, metrics_csv)}")

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != "cpu" else ""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != "cpu" else "cpu")

    # Run for each selected fold
    for fold in args.folds:
        # Model directory and best model file selection
        model_dir = os.path.join(args.model_dir, args.dataset, args.output, fold)
        pred_dir = os.path.join(args.pred_dir)
        model_pattern = os.path.join(model_dir, f"{args.model}_best_epoch*.model")
        best_models = glob(model_pattern)
        if not best_models:
            print(f"No model found in {model_dir} with pattern {model_pattern}")
            continue
        selected_model = best_models[-1]
        print(f"[*] Selected model for testing: {selected_model}")

        # Instantiate model
        if args.model == 'Unet':
            model = Unet(img_ch=3, isDeconv=True, isBN=True)
        elif args.model == 'ATTUNet':
            model = ATTUNet()
        else:
            raise ValueError("Unknown model!")

        # Load weights
        model.load_state_dict(torch.load(selected_model, map_location="cpu"))
        model = model.cpu()  # always run on CPU for now (easy debugging), can change to device

        # Run test on the fold
        model_test(
            model, fold, args.dataset,
            pred_dir,
            args.output,
            batch_size=2,
            metrics_csv="test_metrics.csv"
        )
