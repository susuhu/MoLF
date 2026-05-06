# preprocess_embeddings.py
import yaml
import h5py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm # For a nice progress bar

import pandas as pd
import numpy as np
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

from utils.hest_utils import read_assets_from_h5

import torch
from torchvision import transforms

def load_uni_v2(FIRST_TIME=False):
    hf_token = os.getenv("HF_TOKEN")
    login(
        hf_token
    )  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    local_dir = "Pathology_FM/ckpts/uni2-h/"
    if FIRST_TIME:
        print("[INFO_preprocessing] Downloading UNI2-h model...")
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        hf_hub_download(
            "MahmoodLab/UNI2-h",
            filename="pytorch_model.bin",
            local_dir=local_dir,
            force_download=True,
        )
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    model.load_state_dict(
        torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
        strict=True,
    )
    print("[INFO_preprocessing] UNI2-h model loaded")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def load_uni_v1(FIRST_TIME=False):
    hf_token = os.getenv("HF_TOKEN")
    login(
        hf_token
    )  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    local_dir = "Pathology_FM/ckpts/uni/"
    if FIRST_TIME:
        print("[INFO_preprocessing] Downloading UNI model...")
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        hf_hub_download(
            "MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=local_dir,
            force_download=True,
        )

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(
        torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
        strict=True,
    )
    print("[INFO_preprocessing] UNI model loaded")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def load_hoptimus0():
    hf_token = os.getenv("HF_TOKEN")
    login(
        hf_token
    )  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )

    print("[INFO_preprocessing] hoptimus0 model loaded")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )
    model.eval()
    return model, transform


def load_virchow2():
    hf_token = os.getenv("HF_TOKEN")
    login(
        hf_token
    )  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )

    print("[INFO_preprocessing] virchow2 model loaded")
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()
    return model, transform


def load_gigapath():
    hf_token = os.getenv("HF_TOKEN")
    login(
        hf_token
    )  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    print("[INFO_preprocessing] gigapath model loaded")

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform

def get_PFM_model(config):
    if config["PFM_name"] in ["univ2"]:
        # Load the pretrained pathology foundation model and transform
        PFM_model, transform = load_uni_v2(FIRST_TIME=False)
    elif config["PFM_name"] == "uni":
        PFM_model, transform = load_uni_v1(FIRST_TIME=False)
    elif config["PFM_name"] == "virchow2":
        PFM_model, transform = load_virchow2()
    elif config["PFM_name"] == "hoptimus0":
        PFM_model, transform = load_hoptimus0()
    elif config["PFM_name"] == "gigapath":
        PFM_model, transform = load_gigapath()
    else:
        raise ValueError(f"[ERROR_preprocessing] Unknown PFM name: {config['PFM_name']}")
    return PFM_model, transform

def decode_barcode(b):
    # If it's a numpy array with one element, extract it
    if isinstance(b, np.ndarray):
        b = b.item()
    if isinstance(b, bytes):
        return b.decode()
    elif isinstance(b, np.bytes_):
        return b.astype(str)
    elif isinstance(b, np.str_):
        return str(b)
    elif isinstance(b, str) and b.startswith("[b'") and b.endswith("']"):
        # Handle stringified list of bytes: "[b'AAACAACGAATAGTTC-1']"
        return b[3:-2]
    else:
        return str(b)


def get_all_sample_ids(config, return_split=None):
    # Load all sample IDs from the split CSV file
    split_path = config["paths_config"]["splits_path"]
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"[ERROR_preprocessing] {split_path} does not exist.")
    train_path = os.path.join(split_path, "train_split.csv")
    val_path = os.path.join(split_path, "val_split.csv")
    test_path = os.path.join(split_path, "test_split.csv")
    
    train_df = pd.read_csv(train_path, header=0)
    val_df = pd.read_csv(val_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    train_ids = train_df["sample_id"].tolist()
    val_ids = val_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()
    
    try:
        tess_mouse_path = os.path.join(split_path, "test_mouse_split.csv")
        test_mouse_df = pd.read_csv(tess_mouse_path, header=0)
        test_mouse_ids = test_mouse_df["sample_id"].tolist()
    except:
        print("No test_mouse.csv")
    all_sample_ids = train_ids + val_ids+ test_ids

    if return_split =="test":
        return test_ids 
    elif return_split =="test_mouse":
        return test_mouse_ids
    else:
        return all_sample_ids


def main(config, return_split=None):
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = config["paths_config"]["splits_path"]
    patches_path = config["paths_config"]["patches_path"]   
    save_dir = os.path.join("/mnt/cluster/datasets/HEST1k/", config["PFM_name"])
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Load Model ONCE ---
    print("Loading Pathology Foundation Model...")
    # Load with bfloat16 to save memory, as discussed before
    PFM , transform = get_PFM_model(config)
    PFM= PFM.to(device)
    sample_ids = get_all_sample_ids(config,return_split=return_split)

    # --- Main Processing Loop ---
    for sample_id in tqdm(sample_ids, desc="Preprocessing Samples"):
        save_path = os.path.join(save_dir, f"{sample_id}_embeddings.pt")
        if os.path.exists(save_path):
            continue # Skip if already processed
        print("DEBUG proceessing", sample_id)
        # This logic is copied directly from your Dataset's else block
        asset_path = os.path.join(patches_path, f"{sample_id}.h5")
        # ... load patches from HDF5 ...
        if not os.path.exists(asset_path):
            print(asset_path)
            raise FileNotFoundError(
                f"[ERROR_uni:]Asset file {asset_path} does not exist."
            )
        with h5py.File(asset_path, "r") as f:
            # Assuming the asset is stored in a dataset named 'asset'
            assets, _ = read_assets_from_h5(asset_path)
        patches = assets["img"]
        try:
            barcode = assets["barcode"]
        except:
            barcode = assets["barcodes"]
        coords = assets["coords"]
        assert patches.shape[0] == len(barcode), f"[ERROR_preprocesing] {patches.shape[0]} != {len(barcode)}"   

        patches_transformed_dict = {}
        batch_size = 32  # Adjust based on your GPU memory
        n_patches = patches.shape[0]
        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                batch_patches = []
                batch_indices = range(i, min(i + batch_size, n_patches))
                # loop over batch indices in each batch
                for j in batch_indices:
                    patch = patches[j]
                    if transform:
                        if isinstance(patch, np.ndarray):
                            patch = Image.fromarray(patch.astype(np.uint8))
                        patch_transformed = transform(
                            patch
                        )  # batch_size x 3 x 224 x 224
                    else:
                        patch_transformed = torch.tensor(patch).permute(
                            2, 0, 1
                        )  # batch_size x 3 x 224 x 224
                    # append after each batch
                    batch_patches.append(patch_transformed)  # n_patches x 3 x 224 x 224
                # finshed all batches
                batch_tensor = torch.stack(batch_patches).to(device)  # Shape: (n_patches, 3, 224, 224)
                with torch.no_grad():
                    batch_embeddings = PFM(batch_tensor)  # Shape: (n_patches, 1536)
                for n, j in enumerate(batch_indices):
                    patches_transformed_dict[decode_barcode(barcode[j])] = (
                        batch_embeddings[n].detach().cpu()
                    )  # Store the embedding for each barcode

            # Save the transformed patches to a file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(patches_transformed_dict, save_path)
            print(f"[INFO_preprocessing] Transformed patches saved to: {save_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    with open("/mnt/nct-zfs/hususu/LatentFlow_Pathway/configs/training_config_latent_moe.yml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
