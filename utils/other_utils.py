import os
import numpy as np
import torch
import random
import pandas as pd
from typing import Any
from ruamel.yaml import YAML
import matplotlib.pyplot as plt

def patchify_wsi(
    wsi_path, 
    save_dir, 
    patch_size=256, 
    stride=256, 
    level=0, 
    tissue_threshold=0.5
):
    slide = openslide.OpenSlide(wsi_path)
    slide_dims = slide.level_dimensions[level]
    w, h = slide_dims

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    count = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            # Optional: tissue filter
            if np.mean(patch_np) / 255.0 < (1 - tissue_threshold):
                patch.save(os.path.join(save_dir, f"patch_{count:05d}.png"))
                count += 1

    print(f"Saved {count} patches to {save_dir}")


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current = config
    for i, key in enumerate(keys):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default


def seed_everything(seed: int = 2025):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): Random seed to use.
    """
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic behavior (slower, but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables (for hash-based sources of randomness)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # CUDA >= 10.2 for full determinism
    )


def plot_dataset_distribution():
    # --- Load your CSV files ---
    train = pd.read_csv("data/Hest_Bench/hest_pathway_clean_split_0/train_split.csv")
    val = pd.read_csv("data/Hest_Bench/hest_pathway_clean_split_0/val_split.csv")
    test = pd.read_csv("data/Hest_Bench/hest_pathway_clean_split_0/test_split.csv")

    # --- Add a split column ---
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    # --- Combine all ---
    df = pd.concat([train, val, test], ignore_index=True)

    # --- Count samples per oncotree_code_filled per split ---
    count_df = df.groupby(["oncotree_code_filled", "split"]).size().reset_index(name="count")

    # --- Pivot for grouped bar plot ---
    pivot_df = count_df.pivot(index="oncotree_code_filled", columns="split", values="count").fillna(0)

    # --- Plot setup ---
    splits = ["train", "val", "test"]
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    x = np.arange(len(pivot_df))  # positions
    width = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(15, 8))

    # --- Plot bars ---
    for i, split in enumerate(splits):
        y = pivot_df[split] if split in pivot_df.columns else [0]*len(x)
        bars = ax.bar(x + i*width - width, y, width, label=split, color=colors[split])
        # Add labels on top of bars
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)

    # --- Formatting ---
    ax.set_title("Distribution of Oncotree Code (Filled) Across Splits", fontsize=14)
    ax.set_xlabel("Oncotree Code (filled)")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.legend(title="Dataset Split")
    plt.tight_layout()
    plt.savefig("/mnt/nct-zfs/hususu/LatentFlow_Pathway/data/Hest_Bench/hest_pathway_clean_split_0/oncotree_code_distribution.png")

if __name__ == "__main__":
    # Example usage
    # patchify_wsi(
    #     wsi_path="/mnt/ceph/tco/TCO-Staff/Homes/hususu/Data/hest_data/wsis/INT12.tif", 
    #     save_dir="/mnt/nct-zfs/ST_Gen/figures/INT12_patches", 
    #     patch_size=256, 
    #     stride=256, 
    #     level=1, 
    #     tissue_threshold=0.1
    # )
    yaml = YAML()
    yaml.preserve_quotes = True  # enables quote preservation
    # load the training configuration
    with open("configs/training_config.yml", "r") as f:
        config = yaml.load(f)
    print(config)
    sample = get_nested(config,["paths_config","gene_list_path"])
    print(sample)
