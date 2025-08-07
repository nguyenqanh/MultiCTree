import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import random
import numpy as np

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
import higra as hg
import numpy as np
import pandas as pd
from torch import nn
from unet_model import Unet
from attnunet_model import ATTUNet
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_loader_cv import load_dataset_cv
from losses.clDice.cldice_loss.pytorch.soft_skeleton import SoftSkeletonize
from losses.multiCTree import loss_multiCTree
from losses.CTree import loss_maxima
from scipy import ndimage
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DRIVE', choices=['DRIVE', 'CREMI', 'LESAV'])
    parser.add_argument('--model', default='UNet', choices=['UNet', 'ATTUNet'])
    parser.add_argument('--loss', default='bce+dice', choices=[
        'bce+dice', 'ctree', 'multiCTree', 'cldice'
    ])
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--min_delta', default=0.0001, type=float)
    parser.add_argument('--output', type=str, default='baseline', help='Subfolder for saving under each dataset/fold')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (e.g., 0, 1, 2, ...), or "cpu"')
    return parser.parse_args()


def component(ans):
        # labeling
        s = ndimage.generate_binary_structure(2,2) # 8-connected labeling
        label_ans, nb_labels_a = ndimage.label(ans, structure=s)
        return nb_labels_a

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
    
def clDice(y_true, y_pred):
    y_true = y_true[:, :, :, :]
    y_pred = y_pred[:, :, :, :]
    soft_skeletonize = SoftSkeletonize(num_iter=10)
    skel_pred = soft_skeletonize(y_pred)
    skel_true = soft_skeletonize(y_true)
    tprec = (torch.sum(torch.multiply(skel_pred, y_true))+1.)/(torch.sum(skel_pred)+1.)    
    tsens = (torch.sum(torch.multiply(skel_true, y_pred))+1.)/(torch.sum(skel_true)+1.)    
    cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
    return cl_dice

def multi_ctree_loss(input_, target):
    input_ = input_.squeeze(1).cpu()  # Move input to CPU
    target = target.squeeze(1).cpu()  # Move target to CPU
    batch_size = input_.size(0)
    
    loss_ct_last = torch.tensor(0.0, device="cpu", requires_grad=True)  # Ensure it's a tensor
    graph = hg.get_8_adjacency_implicit_graph((input_.shape[1], input_.shape[2]))
    
    # kernel = torch.ones(1, 1, 3, 3).to("cpu")  # Move kernel to CPU
    # padding = int((kernel.size(2) - 1) / 2)
    # kernel = torch.ones(1, 1, 2, 2).to(input_.device).float()

    for i in range(batch_size):
        # eroded_input = F.conv2d(
        #     input[i:i+1].unsqueeze(1).to(torch.double),
        #     kernel.to(torch.double),
        #     padding=(padding, padding)
        # ).squeeze().float()

        # image_padded = F.pad(input_[i, :, :], (0, 1, 0, 1))
        # eroded_input = F.conv2d(image_padded.unsqueeze(0), kernel, padding=0).squeeze().float()
        
        loss_ct, threshold = loss_multiCTree(
            graph,
            input_[i, :, :],  # Keep it on CPU
            # eroded_input,
            target[i, :, :].numpy(),  # Convert safely
            ["altitude", "connection", "disconnection"],
            "precision",
            p=1,
        )
        loss_ct_last = loss_ct_last + loss_ct 

    loss = loss_ct_last / batch_size
    return loss.to(input_.device)


def ctree_loss(input):
    input = input.squeeze(1).cpu()  # Move input to CPU
    batch_size = input.size(0)
    
    loss_ct_last = torch.tensor(0.0, device="cpu", requires_grad=True)  # Ensure it's a tensor
    graph = hg.get_8_adjacency_implicit_graph((584, 565))
    
    for i in range(batch_size):
        loss_ct = loss_maxima(graph, input[i,:,:].cpu().data, "dynamics", "volume", num_target_maxima=1, margin=1, p=1)
        loss_ct_last = loss_ct_last + loss_ct

    loss = loss_ct_last / batch_size
    return loss.to(input.device)

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return loss.mean()

def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def plot_losses_and_dice(losses_df, plot_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(losses_df['Epoch'], losses_df['Train BCE Loss'], label='Train BCE Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Train Dice Loss'], label='Train Dice Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Train topology-aware Loss'], label='Train topology-aware Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Train Total Loss'], label='Train Total Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Val BCE Loss'], label='Val BCE Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Val Dice Loss'], label='Val Dice Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Val topology-aware Loss'], label='Val topology-aware Loss')
    ax1.plot(losses_df['Epoch'], losses_df['Val Total Loss'], label='Val Total Loss')

    ax1.set_title('Training and Validation Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(losses_df['Epoch'], losses_df['Val Dice Score'], label='Validation Dice Score', color='green')
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0.01, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, val_loss):
        if self.best_loss == float('inf'):
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'[!] EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'[!] Early stopping triggered. Best loss was at epoch {self.best_epoch + 1}')
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def model_train(
    net,
    dataset='DRIVE',
    fold='fold_1_1',
    loss_type='bce+dice',
    epochs=500,
    batch_size=8,
    lr=1e-3,
    patience=100,
    min_delta=0.0001,
    output='baseline'
    ):
    set_seeds()

    device = next(net.parameters()).device
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=lr, weight_decay=0.01)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    
    epoch_list = []
    train_bce_losses = []
    train_dice_losses = []
    train_topology_aware_losses = []
    train_total_losses = []
    val_bce_losses = []
    val_dice_losses = []
    val_topology_aware_losses = []
    val_total_losses = []
    val_dice_scores = []
    
    dataset_folder_map = {
        'DRIVE': f"datasets/DRIVE_dataset/{fold}/",
        'CREMI': f"datasets/CREMI_dataset/{fold}/",
        'LESAV': f"datasets/LESAV_dataset/{fold}/"
    }
    dataset_folder = dataset_folder_map[dataset]

    has_mask = (dataset == 'DRIVE')

    if os.path.exists(dataset_folder):
        print(f"[*] Loading preprocessed {dataset} dataset ({fold}).")
        train_input_tensor = np.load(os.path.join(dataset_folder, "train_image.npy"))
        train_label_tensor = np.load(os.path.join(dataset_folder, "train_label.npy"))
        val_input_tensor = np.load(os.path.join(dataset_folder, "val_image.npy"))
        val_label_tensor = np.load(os.path.join(dataset_folder, "val_label.npy"))
        test_input_tensor = np.load(os.path.join(dataset_folder, "test_image.npy"))
        test_label_tensor = np.load(os.path.join(dataset_folder, "test_label.npy"))
        if has_mask:
            train_mask_tensor = np.load(os.path.join(dataset_folder, "train_mask.npy"))
            val_mask_tensor = np.load(os.path.join(dataset_folder, "val_mask.npy"))
            test_mask_tensor = np.load(os.path.join(dataset_folder, "test_mask.npy"))
        else:
            train_mask_tensor = val_mask_tensor = test_mask_tensor = None
    else:
        print(f"[*] {dataset} dataset not found. Preprocessing and saving dataset.")
        load_dataset_cv(dataset, rel_path='datasets')
        train_input_tensor = np.load(os.path.join(dataset_folder, "train_image.npy"))
        train_label_tensor = np.load(os.path.join(dataset_folder, "train_label.npy"))
        val_input_tensor = np.load(os.path.join(dataset_folder, "val_image.npy"))
        val_label_tensor = np.load(os.path.join(dataset_folder, "val_label.npy"))
        test_input_tensor = np.load(os.path.join(dataset_folder, "test_image.npy"))
        test_label_tensor = np.load(os.path.join(dataset_folder, "test_label.npy"))
        if has_mask:
            train_mask_tensor = np.load(os.path.join(dataset_folder, "train_mask.npy"))
            val_mask_tensor = np.load(os.path.join(dataset_folder, "val_mask.npy"))
            test_mask_tensor = np.load(os.path.join(dataset_folder, "test_mask.npy"))
        else:
            train_mask_tensor = val_mask_tensor = test_mask_tensor = None

    x_train, y_train, m_train = train_input_tensor, train_label_tensor, train_mask_tensor
    x_val, y_val, m_val = val_input_tensor, val_label_tensor, val_mask_tensor

    num_samples = x_train.shape[0]
    num_val_samples = x_val.shape[0]

    best_val_total_loss = float("inf")
    best_model_path = None

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        net.train()
        epoch_tot_loss = 0
        epoch_bce_loss = 0
        epoch_dice_loss = 0
        epoch_topology_aware_loss = 0

        print(f"[+] ====== Start training... {fold} epoch {epoch + 1} ======")

        # Shuffle training data
        num_iters = int(np.ceil(num_samples / batch_size))
        shuffle_ids = np.arange(num_samples)
        np.random.shuffle(shuffle_ids)
        x_train = x_train[shuffle_ids, :, :, :]
        y_train = y_train[shuffle_ids, :, :, :]
        if m_train is not None:
            m_train = m_train[shuffle_ids, :, :, :]

        # Training loop
        for ite in range(num_iters):
            start_id, end_id = ite * batch_size, min((ite + 1) * batch_size, num_samples)
            bat_img = torch.Tensor(x_train[start_id:end_id, :, :, :]).to(device)
            bat_img = bat_img.permute(0, 1, 2, 3)
            bat_label = torch.Tensor(y_train[start_id:end_id, 0:1, :, :]).to(device)

            optimizer.zero_grad()
            bat_pred = net(bat_img)
            bce_loss = criterion(bat_pred, bat_label.float())
            dice_l = dice_loss(bat_pred, bat_label.float())

            # --- LOSS selection ---
            if loss_type == 'bce+dice':
                topology_aware_loss = torch.tensor(0.0, device=device)
                loss = bce_loss + dice_l
            elif loss_type == 'ctree':
                topology_aware_loss = ctree_loss(bat_pred)
                loss = bce_loss + dice_l + 0.001 * topology_aware_loss
            elif loss_type == 'multiCTree':
                topology_aware_loss = multi_ctree_loss(bat_pred, bat_label.float())
                loss = bce_loss + dice_l + 1e-4 * topology_aware_loss
            elif loss_type == 'cldice':
                topology_aware_loss = clDice(bat_label, bat_pred)
                loss = bce_loss + dice_l + topology_aware_loss
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            # -----------------------

            print(f"[*] Fold: {fold}, Epoch: {epoch + 1}, Iter: {ite + 1}, Loss: {loss.item():.8f}, Topology-aware Loss: {topology_aware_loss.item():.8f}")

            loss.backward()
            optimizer.step()

            epoch_tot_loss += loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_dice_loss += dice_l.item()
            epoch_topology_aware_loss += topology_aware_loss.item()

        epoch_avg_loss = epoch_tot_loss / num_iters
        epoch_avg_bce_loss = epoch_bce_loss / num_iters
        epoch_avg_dice_loss = epoch_dice_loss / num_iters
        epoch_avg_topology_aware_loss = epoch_topology_aware_loss / num_iters

        print(f"[+] ====== {fold} Epoch {epoch + 1} finished, Avg Loss: {epoch_avg_loss:.8f} ======")

        # Validation step
        net.eval()
        val_bce_loss = 0
        val_dice_loss = 0
        val_topology_aware_loss = 0
        val_total_loss = 0
        val_dice_score_epoch = 0

        with torch.no_grad():
            num_val_iters = int(np.ceil(num_val_samples / batch_size))
            for val_ite in range(num_val_iters):
                start_id, end_id = val_ite * batch_size, min((val_ite + 1) * batch_size, num_val_samples)
                val_img = torch.Tensor(x_val[start_id:end_id, :, :, :]).to(device)
                val_img = val_img.permute(0, 1, 2, 3)
                val_label = torch.Tensor(y_val[start_id:end_id, 0:1, :, :]).to(device)
                val_pred = net(val_img)

                val_bce = criterion(val_pred, val_label.float())
                val_dice = dice_loss(val_pred, val_label.float())
                current_dice_score = dice_score(val_pred, val_label.float())

                # --- LOSS selection ---
                if loss_type == 'bce+dice':
                    val_mct = torch.tensor(0.0, device=device)
                    val_loss = val_bce + val_dice
                elif loss_type == 'ctree':
                    val_mct = ctree_loss(val_pred)
                    val_loss = val_bce + val_dice + 0.001 * val_mct
                elif loss_type == 'multiCTree':
                    val_mct = multi_ctree_loss(val_pred, val_label.float())
                    val_loss = val_bce + val_dice + 1e-4 * val_mct
                elif loss_type == 'cldice':
                    val_mct = clDice(val_label, val_pred)
                    val_loss = val_bce + val_dice + val_mct
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
                # -----------------------

                val_bce_loss += val_bce.item()
                val_dice_loss += val_dice.item()
                val_topology_aware_loss += val_mct.item()
                val_total_loss += val_loss.item()
                val_dice_score_epoch += current_dice_score.item()

            val_bce_loss /= num_val_iters
            val_dice_loss /= num_val_iters
            val_topology_aware_loss /= num_val_iters
            val_total_loss /= num_val_iters
            val_dice_score_epoch /= num_val_iters

            print(f"[+] Validation Loss: {val_total_loss:.8f}, Dice Score: {val_dice_score_epoch:.8f}, Topology-aware Loss: {val_topology_aware_loss:.8f}")
            if val_total_loss < best_val_total_loss:
                best_val_total_loss = val_total_loss
                best_model_path = (
                    f"./checkpoint/{dataset}/{output}/{fold}/{net.__class__.__name__}_best_epoch{epoch + 1}_loss{best_val_total_loss:.4f}.model"
                )
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(net.state_dict(), best_model_path)
                print(f"[+] New best model saved at epoch {epoch + 1} with validation total loss {best_val_total_loss:.4f}")

        epoch_list.append(epoch + 1)
        train_bce_losses.append(epoch_avg_bce_loss)
        train_dice_losses.append(epoch_avg_dice_loss)
        train_topology_aware_losses.append(epoch_avg_topology_aware_loss)
        train_total_losses.append(epoch_avg_loss)
        val_bce_losses.append(val_bce_loss)
        val_dice_losses.append(val_dice_loss)
        val_topology_aware_losses.append(val_topology_aware_loss)
        val_total_losses.append(val_total_loss)
        val_dice_scores.append(val_dice_score_epoch)

        early_stopping(epoch, val_total_loss)
        if early_stopping.early_stop:
            print(f"[!] Training stopped early at epoch {epoch + 1}")
            break

    losses_df = pd.DataFrame({
        'Epoch': epoch_list,
        'Train BCE Loss': train_bce_losses,
        'Train Dice Loss': train_dice_losses,
        'Train topology-aware Loss': train_topology_aware_losses,
        'Train Total Loss': train_total_losses,
        'Val BCE Loss': val_bce_losses,
        'Val Dice Loss': val_dice_losses,
        'Val topology-aware Loss': val_topology_aware_losses,
        'Val Total Loss': val_total_losses,
        'Val Dice Score': val_dice_scores
    })

    # Save metrics and plot per fold!
    metrics_path = f'checkpoint/{dataset}/{output}/{fold}/training_metrics.csv'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    losses_df.to_csv(metrics_path, index=False)
    plot_path = f'checkpoint/{dataset}/{output}/{fold}/metrics_plot.png'
    plot_losses_and_dice(losses_df, plot_path)

    print(f"[+] Training completed. Best model saved at {best_model_path} with total loss {best_val_total_loss:.4f}")
    if early_stopping.early_stop:
        print(f"[+] Best validation total loss ({best_val_total_loss:.4f}) was achieved at epoch {early_stopping.best_epoch + 1}")

    return net

if __name__ == "__main__":
    set_seeds()
    args = parse_args()
    if args.gpu.lower() == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    folds = ['fold_1_1', 'fold_1_2', 'fold_2_1', 'fold_2_2']

    for fold in folds:
        print(f"===== Training {args.dataset} {fold} with {args.model} using {args.loss} loss on device {device} =====")
        if args.model == 'UNet':
            net = Unet(img_ch=3, isDeconv=True, isBN=True)
        elif args.model == 'ATTUNet':
            net = ATTUNet()
        else:
            raise ValueError(f"Unknown model: {args.model}")

        set_seeds()
        net.to(device)
        model_train(
            net,
            dataset=args.dataset,
            fold=fold,
            loss_type=args.loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            min_delta=args.min_delta,
            output=args.output
        )
