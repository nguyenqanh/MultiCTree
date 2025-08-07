from PIL import Image
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def eval_print_metrics(bat_label, bat_pred, bat_mask=None, use_mask=True):
    """
    If use_mask is True (for DRIVE), metrics and AUC are computed inside the mask.
    If use_mask is False (for STARE or others), no mask is used.
    """
    assert len(bat_label.size()) == 4 and len(bat_pred.size()) == 4
    assert bat_label.size()[1] == 1 and bat_pred.size()[1] == 1
    if use_mask:
        assert bat_mask is not None and len(bat_mask.size()) == 4 and bat_mask.size()[1] == 1
        masked_pred = bat_pred * bat_mask
        masked_label = bat_label * bat_mask
        masked_pred_class = (masked_pred > 0.5).float()
        pred_ls = np.array(bat_pred[bat_mask > 0].detach().cpu())
        label_ls = np.array(bat_label[bat_mask > 0].detach().cpu(), dtype=int)
    else:
        masked_pred = bat_pred
        masked_label = bat_label
        masked_pred_class = (masked_pred > 0.5).float()
        pred_ls = np.array(bat_pred.detach().cpu()).reshape(-1)
        label_ls = np.array(bat_label.detach().cpu(), dtype=int).reshape(-1)

    precision = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_pred_class)) + 1)
    recall = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_label.float())) + 1)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)
    try:
        bat_auc = roc_auc_score(label_ls, pred_ls)
    except:
        bat_auc = 0.0

    print("[*] ...... Evaluation ...... ")
    print(" >>> precision: {:.4f} recall: {:.4f} f1_score: {:.4f} auc: {:.4f}".format(precision, recall, f1_score, bat_auc))

    return precision, recall, f1_score, bat_auc

def eval_print_metrics_STARE(bat_label, bat_pred):
    # STARE does not use a mask.
    return eval_print_metrics(bat_label, bat_pred, bat_mask=None, use_mask=False)

def paste_and_save(bat_img, bat_label, bat_pred, bat_pred_class, batch_size, cur_bat_num, save_img='pred_imgs', bat_mask=None, use_mask=True):
    # Create subdirectories if they don't exist
    prob_dir = os.path.join(save_img, 'probabilities')
    binary_dir = os.path.join(save_img, 'binary')
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    for bat_id in range(bat_img.size()[0]):
        res_id = (cur_bat_num - 1) * batch_size + bat_id

        # Save probability map
        prob_map = bat_pred[bat_id, 0, :, :].cpu().detach().numpy()
        prob_img = Image.fromarray((prob_map * 255.0).astype(np.uint8))
        prob_img.save(os.path.join(prob_dir, f"prob_{res_id+1}.png"))

        # Save binary prediction, mask only for DRIVE (use_mask==True)
        binary_map = bat_pred_class[bat_id, 0, :, :].detach().cpu().numpy()
        if use_mask and bat_mask is not None:
            mask_map = bat_mask[bat_id, 0, :, :].detach().cpu().numpy()
            binary_map = binary_map * mask_map

        binary_pred = Image.fromarray((binary_map * 255.0).astype(np.uint8))
        binary_pred.save(os.path.join(binary_dir, f"binary_{res_id+1}.png"))
