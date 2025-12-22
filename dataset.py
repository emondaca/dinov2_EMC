import os
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
from torchvision import transforms
import cv2 as cv
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd

from rad_dino.utils import safetensors_to_state_dict
from extract_features import extract_features_rad_dino
from rad_dino import RadDino
from transformers import pipeline
from PIL import Image
from model import RadDINOFirst11Extractor
from clean_sentence import clean_sentence_label, clean_suspects_terms, remove_exclusive_terms
import multilabel_oversampling as mo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CachedFeatureDataset(Dataset):
    def __init__(self, cached_data):
        self.data = cached_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['image_feat']            # (D,)
        y = torch.as_tensor(self.data[idx]['label'], dtype=torch.float32)         # (C,)
        return {'image_feat': x, 'label': y}
    
class CustomImageDataset(Dataset):
    def __init__(self, df, image_dir, IMAGE_SIZE = 224, split="train", DataAug = True):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.image_ids = df['ImageID'].values
        self.labels = torch.tensor(np.array(df['multi_hot'].to_list()), dtype=torch.float32)
        self.image_dir = image_dir

        if DataAug:
            self.augData = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.augData = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.originalData = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.split = split

        self.transform = self.augData if self.split == "train" else self.originalData

    def __len__(self):
        return len(self.image_ids)
    
    def set_split(self, split: str):
        self.split = split
        self.transform = self.augData if split == "train" else self.originalData

    def get_labels_only(self):
        return self.labels.numpy()

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = cv.imread(os.path.join(self.image_dir, img_id))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = self.transform(img_rgb)

        y = self.labels[idx]
        return x, y
    
        #pil_raw = Image.fromarray(img_rgb)            # for HF pipeline
        #return pil_raw ,y

    
def print_splits(train_dataset, val_dataset, test_dataset, classes):
    # Access original dataset
    full_dataset = train_dataset.dataset
    full_labels = full_dataset.get_labels_only()

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    y_train = full_labels[train_indices]
    y_val = full_labels[val_indices]
    y_test = full_labels[test_indices]

    class_col_width = max(len(c) for c in classes) + 2
    header = f"{'Class':<{class_col_width}} | {'Train':>7} | {'Val':>7} | {'Test':>7}"
    print("\n" + header)
    print("-" * len(header))

    for idx, class_name in enumerate(classes):
        train_count = int(y_train[:, idx].sum())
        val_count = int(y_val[:, idx].sum())
        test_count = int(y_test[:, idx].sum())
        print(f"{class_name:<{class_col_width}} | {train_count:>7} | {val_count:>7} | {test_count:>7}")



def stratified_split_multilabel(labels, n_splits=5, fold=0):
    # Extract all labels from the dataset
    #labels = dataset.get_labels_only()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (split1_idx, split2_idx) in enumerate(mskf.split(np.zeros(len(labels)), labels)):
        if i == fold:
            return split1_idx, split2_idx


def make_weighted_sampler_for_subset(base_dataset, subset_indices):
    print("weighted random sampler MAX")
    Y = base_dataset.get_labels_only()[subset_indices]  # [N_subset, C]
    class_counts  = Y.sum(axis=0)                       # [C]
    class_weights = 1.0 / np.clip(class_counts, 1, None)

    pos_per_sample = Y.sum(axis=1)                      # [N_subset]
    sample_weights = (Y * class_weights).sum(axis=1) / np.maximum(pos_per_sample, 1)

    # handle any all-zero rows (rare)
    #if (pos_per_sample == 0).any():
    #    sample_weights[pos_per_sample == 0] = sample_weights[pos_per_sample > 0].mean()

    # optional: normalize weights
    #sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())

    # --- key change: make the epoch long enough to target ~max count per class ---
    target   = int(class_counts.max())                  # majority count (e.g., 856)
    avg_k    = float(pos_per_sample.mean()) if len(pos_per_sample) else 1.0  # avg labels/sample
    epoch_sz = int(round(target * Y.shape[1] / max(avg_k, 1.0)))             # ≈ target per class

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=epoch_sz,          # longer epoch → ~target positives per class
        replacement=True
    )
    return sampler

def make_weighted_sampler_normal(base_dataset, subset_indices):
    print("weighted random sampler MEAN")
    # labels: [N_subset, C] with {0,1}
    Y = base_dataset.get_labels_only()[subset_indices]
    Y = np.asarray(Y, dtype=np.float64)

    # per-class inverse frequency (clip to avoid div-by-zero)
    class_counts  = Y.sum(axis=0)                       # [C]
    class_weights = 1.0 / np.clip(class_counts, 1, None)

    # per-sample weight = average of its positive class weights
    pos_per_sample = Y.sum(axis=1)                      # [N_subset]
    sample_weights = (Y * class_weights).sum(axis=1) / np.maximum(pos_per_sample, 1)

    # handle rare all-zero rows (no positives)
    #if (pos_per_sample == 0).any():
    #    nz = sample_weights[pos_per_sample > 0]
    #    sample_weights[pos_per_sample == 0] = nz.mean() if nz.size else 1.0

    # (optional) scale so average weight ≈ 1; not required by the sampler
    #s = sample_weights.sum()
    #if s > 0:
    #    sample_weights = sample_weights * (len(sample_weights) / s)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),   # one pass worth of draws
        replacement=True
    )

def weightedClass(all_labels, abs_train_idx):
    Y_train = all_labels[abs_train_idx]            # NumPy array
    pos_counts = Y_train.sum(axis=0)               # use axis, not dim
    N = Y_train.shape[0]
    pos_weight = (N - pos_counts) / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)


    return pos_weight

def overSampling(temp_idx,train_idx, dataset_train):
    train_orig_idx = [temp_idx[i] for i in train_idx]

    y_all = dataset_train.get_labels_only()              
    y_train = y_all[train_orig_idx]                      

    num_labels = y_train.shape[1]
    label_cols = [f"y{i+1}" for i in range(num_labels)]  # y1, y2, ...
    df_train = pd.DataFrame(y_train, columns=label_cols).astype(int)

    df_train.insert(0, "orig_idx", train_orig_idx)

    ml_oversampler = mo.MultilabelOversampler(number_of_adds=6000, number_of_tries=6000)
    df_train_os = ml_oversampler.fit(df_train, target_list=label_cols)

    new_train_idx = df_train_os["orig_idx"].tolist()
    train_dataset_os = torch.utils.data.Subset(dataset_train, new_train_idx)

    return train_dataset_os

def print_20_test_image_ids(test_idx, df_grouped, all_labels, label_names, max_samples=20):

    selected_abs_idx = []

    for abs_idx in test_idx:
        y = all_labels[abs_idx]
        pos = np.where(y == 1)[0]
        if len(pos) == 0:
            continue  # skip images with no labels

        img_id = df_grouped.iloc[abs_idx]['ImageID']
        class_names = [label_names[j] for j in pos]

        print(f"{len(selected_abs_idx):02d} | abs_idx={abs_idx} | ImageID={img_id} | labels={class_names}")

        selected_abs_idx.append(abs_idx)

        if len(selected_abs_idx) >= max_samples:
            break

    return selected_abs_idx



def get_data(data_filtered, images_src,RadDino_src,RadDinoWeights, subLabel, IMAGE_SIZE = 448, batch_size = 16, wrs_mode = None, radDinoType = 1, xlsx_path = "", DataAug = True, oversampler = False, label_count = 25):
    #from dinov2.hub.backbones import dinov2_vitb14
    #rad_dino_gh = dinov2_vitb14()

    """rad_dino_gh = torch.hub.load(RadDino_src, "dinov2_vitb14", source="local")
    rad_dino_gh = rad_dino_gh
    print(rad_dino_gh)
    backbone_state_dict = safetensors_to_state_dict(RadDinoWeights)

    rad_dino_gh.load_state_dict(backbone_state_dict, strict=True)"""

    #rad_dino_gh = pipeline(task="image-feature-extraction", model="microsoft/rad-dino-maira-2", pool=False)
    rad_dino_gh = RadDINOFirst11Extractor(RadDino_src,RadDinoWeights, radDinoType = radDinoType)


    if subLabel:
        label_counts = Counter(data_filtered['label'])
        valid_labels = {label for label, count in label_counts.items() if count >= 3}
        data_filtered = data_filtered[data_filtered['label'].isin(valid_labels)]
        label_output_count = len(valid_labels)
    else:
        top_labels = (
            data_filtered['label_group']
            .value_counts()
            .nlargest(label_count)     
            .index            
        )

        data_filtered = data_filtered[data_filtered['label_group'].isin(top_labels)]
        label_output_count = len(top_labels)

        print("Label Number Types: ", label_output_count)


    # Group labels by ImageID
    if subLabel:
        data_grouped = (
            data_filtered
            .groupby('ImageID')
            .agg({
                'label': lambda x: list(x),
                'sentence_en': lambda x: ' '.join(x.dropna())
            })
            .reset_index()
        )
    else: 
        data_grouped = (
            data_filtered
            .groupby('ImageID')
            .agg({
                'label_group': lambda x: list(x),
                'sentence_en': lambda x: ' '.join(x.dropna())
            })
            .reset_index()
        )

        print("Remove label terms")
        data_grouped['sentence_en_clean'] = data_grouped.apply(clean_sentence_label, axis=1)

        #eliminate the empty sentences after removing the label terms
        data_grouped = data_grouped[data_grouped['sentence_en_clean'].str.strip() != '']
        data_grouped = data_grouped.reset_index(drop=True)

        if xlsx_path != "":
            print("Remove suspects terms")
            phrase_re = clean_suspects_terms(xlsx_path)
            data_grouped['sentence_en_clean_terms'] = data_grouped['sentence_en_clean'].apply(lambda txt: remove_exclusive_terms(txt, regex=phrase_re))

            # Replace the empty sentences with the previous filter after aplying the removing of the suspects terms
            data_grouped['final_sentence'] = np.where(
                data_grouped['sentence_en_clean_terms'].str.strip() == '',
                data_grouped['sentence_en_clean'],   
                data_grouped['sentence_en_clean_terms']  
            )


    #multi_hot
    mlb = MultiLabelBinarizer()
    if subLabel:
        multi_hot_labels = mlb.fit_transform(data_grouped['label'])
    else:
        multi_hot_labels = mlb.fit_transform(data_grouped['label_group'])

    data_grouped['multi_hot'] = list(multi_hot_labels)

    dataset_train = CustomImageDataset(data_grouped, images_src, IMAGE_SIZE=IMAGE_SIZE, split="train", DataAug = DataAug)
    dataset_val   = CustomImageDataset(data_grouped, images_src, IMAGE_SIZE=IMAGE_SIZE, split="val", DataAug = DataAug)
    dataset_test  = CustomImageDataset(data_grouped, images_src, IMAGE_SIZE=IMAGE_SIZE, split="test", DataAug = DataAug)


    #image_encoder = ResNet50()

    #features_dataset, feat_dim = extract_features(dataset, image_encoder)
    #print("FF")
    all_labels = dataset_train.get_labels_only()
    temp_idx, test_idx = stratified_split_multilabel(all_labels, n_splits=5, fold=0)
    #print(len(temp_idx))
    test_dataset = torch.utils.data.Subset(dataset_test, test_idx)

    #temp_dataset_full = [dataset[i] for i in temp_idx]
    temp_dataset_full = all_labels[temp_idx]
    #print("Done")
    train_idx, val_idx = stratified_split_multilabel(temp_dataset_full, n_splits=4, fold=0)

    train_dataset = torch.utils.data.Subset(dataset_train, [temp_idx[i] for i in train_idx])
    val_dataset = torch.utils.data.Subset(dataset_val, [temp_idx[i] for i in val_idx])

    #Print the splits result
    print_splits(train_dataset, val_dataset, test_dataset, mlb.classes_)

    # ---- print 20 test samples with ImageID for Grad-CAM ----
    print("\n=== 20 test samples (ImageID + labels) for Grad-CAM ===")
    selected_abs_idx = print_20_test_image_ids(
        test_idx=test_idx,
        df_grouped=data_grouped,
        all_labels=all_labels,
        label_names=mlb.classes_,
        max_samples=20
    )

    #Aplicat RadDino
    #image_encoder = RadDino()
    image_encoder = rad_dino_gh

    if oversampler:
        train_dataset = overSampling(temp_idx,train_idx, dataset_train)
        print_splits(train_dataset, val_dataset, test_dataset, mlb.classes_)

        train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset, image_encoder, radDinoType = radDinoType)
        train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
        pos_weight = None
    else:
        if wrs_mode == "max" or wrs_mode == "mean":
            if wrs_mode == "max": train_sampler = make_weighted_sampler_for_subset(dataset_train, train_idx)
            else: train_sampler = make_weighted_sampler_normal(dataset_train, train_idx)

            with torch.no_grad():
                Y_train = dataset_train.get_labels_only()[train_idx]
                picked = list(train_sampler)               # indices 0..len(train_ds)-1
                approx_counts = Y_train[picked].sum(axis=0)
                print("Approx. train counts this epoch:", approx_counts.astype(int).tolist())

                """Y_all = dataset_train.get_labels_only()              # [N, C]
                Y_train = torch.as_tensor(Y_all[train_idx], dtype=torch.float32)  # [N_train, C]
                P = Y_train.sum(dim=0)                               # positives per class, [C]
                N = Y_train.shape[0] - P                             # negatives per class, [C]

                eps = 1e-6
                pos_weight = (N + eps) / (P + eps)"""                   # [C], float32

                Y_train_raw = torch.as_tensor(dataset_train.get_labels_only()[train_idx], dtype=torch.float32)
                P = Y_train_raw.sum(dim=0)                  # [C]
                N_total = Y_train_raw.shape[0]
                pos_weight = (N_total - P) / (P + 1e-6)     # (#negatives / #positives)

                # Optional: clamp to avoid huge gradients for ultra-rare classes
                pos_weight = torch.clamp(pos_weight, max=100.0)
                #print("pos_weight:", pos_weight.tolist())

            train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset, image_encoder, radDinoType = radDinoType)
            train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,sampler=train_sampler, pin_memory=True)
        
        else:

            train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset, image_encoder,radDinoType = radDinoType)
            train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
            #pos_weight = weightedClass(dataset_train,train_idx)


            abs_train_idx = [temp_idx[i] for i in train_idx]
            all_labels = dataset_train.get_labels_only()         # torch [N, C]
            pos_weight = weightedClass(all_labels, abs_train_idx)

    val_dataset_extractedFeatures, _ = extract_features_rad_dino(val_dataset, image_encoder,radDinoType = radDinoType)
    test_dataset_extractedFeatures, feat_dim = extract_features_rad_dino(test_dataset, image_encoder,radDinoType = radDinoType)

    val_dataset = CachedFeatureDataset(val_dataset_extractedFeatures)
    test_dataset = CachedFeatureDataset(test_dataset_extractedFeatures)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, label_output_count, pos_weight