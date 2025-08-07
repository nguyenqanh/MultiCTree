from PIL import Image, ImageEnhance
import numpy as np
import os
from glob import glob
import random
from sklearn.model_selection import train_test_split

def load_dataset_cv(dataset_name, rel_path='datasets', resize=True, resize_shape=None, random_state=42):
    # Set dataset-specific paths and parameters
    if dataset_name == 'CREMI':
        image_files = sorted(glob(os.path.join(rel_path, 'CREMI/{}/images/*.png'.format('training'))))
        label_files = sorted(glob(os.path.join(rel_path, 'CREMI/{}/labels/*.png'.format('training'))))
        mask_files = None
        if resize_shape is None:
            resize_shape = (625, 625)
    elif dataset_name == 'DRIVE':
        image_files = sorted(glob(os.path.join(rel_path, 'DRIVE/{}/images/*.tif'.format('training'))))
        label_files = sorted(glob(os.path.join(rel_path, 'DRIVE/{}/1st_manual/*.gif'.format('training'))))
        mask_files = sorted(glob(os.path.join(rel_path, 'DRIVE/{}/mask/*.gif'.format('training'))))
        if resize_shape is None:
            resize_shape = (565, 584)
    elif dataset_name == 'LESAV':
        image_files = sorted(glob(os.path.join(rel_path, 'LES-AV/{}/images/*.png'.format('training'))))
        label_files = sorted(glob(os.path.join(rel_path, 'LES-AV/{}/labels/*.png'.format('training'))))
        mask_files = None
        if resize_shape is None:
            resize_shape = (810, 722)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    print(f"Loaded {len(image_files)} images from {dataset_name}")

    indices = list(range(len(image_files)))
    first_half_idx, second_half_idx = train_test_split(indices, test_size=0.5, random_state=random_state)
    second_half_val_idx, second_half_test_idx = train_test_split(second_half_idx, test_size=0.5, random_state=random_state)
    first_half_val_idx, first_half_test_idx = train_test_split(first_half_idx, test_size=0.5, random_state=random_state)

    configs = {
        'fold_1_1': {'train': first_half_idx, 'val': second_half_val_idx, 'test': second_half_test_idx},
        'fold_1_2': {'train': first_half_idx, 'val': second_half_test_idx, 'test': second_half_val_idx},
        'fold_2_1': {'train': second_half_idx, 'val': first_half_val_idx, 'test': first_half_test_idx},
        'fold_2_2': {'train': second_half_idx, 'val': first_half_test_idx, 'test': first_half_val_idx},
    }

    def process_files(image_files, label_files, mask_files=None, mode='train', augment=False):
        input_tensor, label_tensor = [], []
        mask_tensor = [] if mask_files is not None else None

        for i, filename in enumerate(image_files):
            img = Image.open(filename).convert('RGB')
            label_img = Image.open(label_files[i]).convert('L')
            if resize:
                img = img.resize(resize_shape, Image.ANTIALIAS)
                label_img = label_img.resize(resize_shape, Image.NEAREST)
            if augment and mode == 'train':
                random_gen = random.Random()
                if random_gen.random() < 0.5:
                    angle = random_gen.uniform(-15, 15)
                    img = img.rotate(angle)
                    label_img = label_img.rotate(angle)
                if random_gen.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                if random_gen.random() < 0.5:
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(random_gen.uniform(0.8, 1.2))

            imgmat = np.array(img).astype('float32') / 255.0
            label = np.array(label_img).astype('float32') / 255.0

            input_tensor.append(np.expand_dims(imgmat, axis=0))
            label_tensor.append(np.expand_dims(label, axis=0))

            if mask_files is not None:
                mask_img = Image.open(mask_files[i]).convert('L')
                if resize:
                    mask_img = mask_img.resize(resize_shape, Image.NEAREST)
                mask = np.array(mask_img).astype('float32') / 255.0
                mask_tensor.append(np.expand_dims(mask, axis=0))

        new_input_tensor = np.moveaxis(np.concatenate(input_tensor, axis=0), 3, 1)
        new_label_tensor = np.stack((np.concatenate(label_tensor, axis=0), 1 - np.concatenate(label_tensor, axis=0)), axis=1)
        if mask_files is not None:
            new_mask_tensor = np.stack((np.concatenate(mask_tensor, axis=0), np.concatenate(mask_tensor, axis=0)), axis=1)
            return new_input_tensor, new_label_tensor, new_mask_tensor
        else:
            return new_input_tensor, new_label_tensor

    def save_filenames(fold_dir, split_name, filenames):
        with open(f"{fold_dir}/{split_name}_files.txt", "w") as f:
            for file in filenames:
                f.write(file + "\n")

    for fold_name, idxs in configs.items():
        fold_dir = f"datasets/{dataset_name}_dataset/{fold_name}"
        os.makedirs(fold_dir, exist_ok=True)

        train_idx, val_idx, test_idx = idxs['train'], idxs['val'], idxs['test']

        train_filenames = [image_files[i] for i in train_idx]
        val_filenames = [image_files[i] for i in val_idx]
        test_filenames = [image_files[i] for i in test_idx]

        save_filenames(fold_dir, "train", train_filenames)
        save_filenames(fold_dir, "val", val_filenames)
        save_filenames(fold_dir, "test", test_filenames)

        if mask_files is not None:
            train_data = process_files([image_files[i] for i in train_idx],
                                       [label_files[i] for i in train_idx],
                                       [mask_files[i] for i in train_idx], "train", augment=True)
            val_data = process_files([image_files[i] for i in val_idx],
                                     [label_files[i] for i in val_idx],
                                     [mask_files[i] for i in val_idx], "val", augment=False)
            test_data = process_files([image_files[i] for i in test_idx],
                                      [label_files[i] for i in test_idx],
                                      [mask_files[i] for i in test_idx], "test", augment=False)
            np.save(f"{fold_dir}/train_image.npy", train_data[0])
            np.save(f"{fold_dir}/train_label.npy", train_data[1])
            np.save(f"{fold_dir}/train_mask.npy", train_data[2])
            np.save(f"{fold_dir}/val_image.npy", val_data[0])
            np.save(f"{fold_dir}/val_label.npy", val_data[1])
            np.save(f"{fold_dir}/val_mask.npy", val_data[2])
            np.save(f"{fold_dir}/test_image.npy", test_data[0])
            np.save(f"{fold_dir}/test_label.npy", test_data[1])
            np.save(f"{fold_dir}/test_mask.npy", test_data[2])
        else:
            train_data = process_files([image_files[i] for i in train_idx],
                                       [label_files[i] for i in train_idx], None, "train", augment=True)
            val_data = process_files([image_files[i] for i in val_idx],
                                     [label_files[i] for i in val_idx], None, "val", augment=False)
            test_data = process_files([image_files[i] for i in test_idx],
                                      [label_files[i] for i in test_idx], None, "test", augment=False)
            np.save(f"{fold_dir}/train_image.npy", train_data[0])
            np.save(f"{fold_dir}/train_label.npy", train_data[1])
            np.save(f"{fold_dir}/val_image.npy", val_data[0])
            np.save(f"{fold_dir}/val_label.npy", val_data[1])
            np.save(f"{fold_dir}/test_image.npy", test_data[0])
            np.save(f"{fold_dir}/test_label.npy", test_data[1])
    return
