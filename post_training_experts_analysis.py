import os
import torch
import yaml
import argparse
import h5py 
from tqdm import tqdm
import joblib 
import umap

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.hest_utils import load_gene_list
from utils.custom_dataset import PrecomputedEmbeddingDataset
from utils.other_utils import seed_everything

def create_expert_montage(
    sampling_dir:str, 
    config: dict,
    num_top_patches: int ,
    save_dir: str,
    test_dataset: PrecomputedEmbeddingDataset,
):
    """
    Analyzes saved gating logits to find and visualize the top activating
    patches for each expert by re   ading from HDF5 files.
    """

    patch_file_dir=config["paths_config"]["patches_path"]

    print("Loading all gating logits...")
    all_logits = []
    patch_metadata = [] # List of {'sample_id': str, 'patch_index': int}
    
    # In this new logic, sample_data_index is the index for test_dataset.sample_ids
    for sample_data_index, sample_id in enumerate(test_dataset.sample_ids):
        logits_path = os.path.join(sampling_dir, f"{sample_id}_gating_logits.pt")
        
        if os.path.exists(logits_path):
            logits = torch.load(logits_path, map_location='cpu')
            all_logits.append(logits)
            
            num_patches = logits.shape[0]
            for patch_idx in range(num_patches):
                # We store the sample_id and the patch_index *within that sample*
                patch_metadata.append({'sample_id': sample_id, 'patch_index': patch_idx})

    if not all_logits:
        raise FileNotFoundError("No gating logit files found.")

    all_logits = torch.cat(all_logits, dim=0)
    num_experts = all_logits.shape[1]
    
    # --- 2. For each expert, find top patches and create a montage ---
    for expert_i in range(num_experts):
        print(f"\n--- Analyzing Expert {expert_i+1} ---")
        
        expert_scores = all_logits[:, expert_i]
        top_indices = torch.topk(expert_scores, num_top_patches).indices
        
        fig, axes = plt.subplots(int(num_top_patches ** 0.5), int(num_top_patches ** 0.5), figsize=(12, 12))
        axes = axes.flatten()
        
        for i, patch_global_idx in enumerate(top_indices):
            meta = patch_metadata[patch_global_idx]
            sample_id = meta['sample_id']
            patch_idx = meta['patch_index']
            
            # --- NEW: Retrieve the image patch from the HDF5 file ---
            ax = axes[i]
            try:
                h5_path = os.path.join(patch_file_dir, f"{sample_id}.h5")
                with h5py.File(h5_path, 'r') as h5f:
                    # Read the specific patch image using its index
                    patch_img = h5f['img'][patch_idx] # Shape (224, 224, 3), dtype uint8
                
                ax.imshow(patch_img)
                ax.set_title(f"{sample_id}\nPatch #{patch_idx}", fontsize=10)

            except Exception as e:
                print(f"Warning: Could not load patch for {sample_id}, index {patch_idx}. Error: {e}")
                ax.set_title("Load Error", color='red')
            
            ax.axis('off')
   
        fig.suptitle(f'Top Activating Patches for Expert {expert_i+1}')
        plt.tight_layout()
        montage_save_path = os.path.join(save_dir, f"expert_{expert_i+1}_montage.jpg")
        plt.savefig(montage_save_path)
        # plt.show()
        print(f"Montage saved to {montage_save_path}")


def analyze_expert_utilization(
    sampling_dir: str, 
    config: dict,
    save_dir: str,
    test_dataset: PrecomputedEmbeddingDataset,
    k: int = 2,
):
    """
    Analyzes saved gating logits to count expert activation frequency per-sample
    and aggregated per-oncotree-code.
    """
    print(f"Analyzing expert utilization from directory: {sampling_dir}")
   
    # --- Load the test split file to get oncotree codes for each sample ---
    test_split_path = os.path.join(config["paths_config"]["splits_path"], "test_split.csv")
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split file not found at: {test_split_path}")
    split_df = pd.read_csv(test_split_path)
    # Create a simple mapping from sample_id to oncotree_code
    split_df['sample_id'] = split_df['sample_id'].astype(str)
    oncotree_map = split_df.set_index('sample_id')['oncotree_code_filled'].to_dict()

    results = []
    num_experts = -1

    print("Processing logits for each sample...")
    for sample_id in tqdm(test_dataset.sample_ids, desc="Analyzing Samples"):
        logits_path = os.path.join(sampling_dir, f"{sample_id}_gating_logits.pt")
        
        if not os.path.exists(logits_path):
            print(f"Warning: Logits file not found for sample {sample_id}. Skipping.")
            continue

        logits = torch.load(logits_path, map_location='cpu')
        
        if num_experts == -1:
            num_experts = logits.shape[1]
            print(f"Inferred {num_experts} experts from the data.")

        num_patches = logits.shape[0]

        _, top_indices = torch.topk(logits, k=k, dim=1)
        all_activations = top_indices.flatten()
        counts = torch.bincount(all_activations, minlength=num_experts).numpy()
        
        sample_result = {
            'sample_id': sample_id, 
            'oncotree_code': oncotree_map.get(sample_id, 'Unknown'), # Add oncotree code
            'num_patches': num_patches
        }
        for i in range(num_experts):
            sample_result[f'expert_{i}_count'] = counts[i]
        
        results.append(sample_result)

    if not results:
        raise RuntimeError("No logit files were found and processed.")

    df_sample = pd.DataFrame(results)
    
    # --- 1. Per-Sample Analysis (Same as before) ---
    total_activations_col = df_sample['num_patches'] * k
    for i in range(num_experts):
        df_sample[f'expert_{i}_percent'] = (df_sample[f'expert_{i}_count'] / total_activations_col) * 100

    count_cols = [f'expert_{i}_count' for i in range(num_experts)]
    percent_cols = [f'expert_{i}_percent' for i in range(num_experts)]
    df_sample = df_sample[['sample_id', 'oncotree_code', 'num_patches'] + count_cols + percent_cols]
    
    save_path_sample = os.path.join(save_dir, "expert_utilization_per_sample.csv")
    df_sample.to_csv(save_path_sample, index=False)
    print(f"\nDetailed per-sample expert utilization saved to: {save_path_sample}")

    # --- 2. NEW: Per-Oncotree Code Aggregation ---
    print("\nAggregating expert utilization by oncotree code...")
    
    # Sum the counts for each oncotree code
    oncotree_grouped = df_sample.groupby('oncotree_code')[['num_patches'] + count_cols].sum().reset_index()
    
    # Calculate the percentage utilization based on the *total* counts for that oncotree code
    total_oncotree_activations = oncotree_grouped['num_patches'] * k
    for i in range(num_experts):
        oncotree_grouped[f'expert_{i}_percent'] = (oncotree_grouped[f'expert_{i}_count'] / total_oncotree_activations) * 100

    df_oncotree = oncotree_grouped[['oncotree_code', 'num_patches'] + count_cols + percent_cols]

    save_path_oncotree = os.path.join(save_dir, "expert_utilization_per_oncotree.csv")
    df_oncotree.to_csv(save_path_oncotree, index=False)
    print(f"Aggregated per-oncotree expert utilization saved to: {save_path_oncotree}")
    
    # --- Print Summaries ---
    print("\n--- Average Expert Utilization Across All Samples ---")
    summary_sample = df_sample[percent_cols].mean().reset_index()
    summary_sample.columns = ['Expert', 'Average Utilization (%)']
    print(summary_sample.to_string(index=False))

    print("\n--- Expert Utilization Profile Per Oncotree Code (%) ---")
    print(df_oncotree[['oncotree_code'] + percent_cols].to_string(index=False))


def visualize_expert_umap(
    sampling_dir: str,   # Specific sampling dir with logits
    vae_log_dir: str,    # Log dir for the PRETRAINED VAE
    save_dir: str,
    test_dataset: PrecomputedEmbeddingDataset,
    num_top_patches: int = 20, # Number of top patches to show per expert
    PAPER = False,
):
    """
    Visualizes the UMAP embeddings of the top activating patches for each expert.
    """
    print("--- Step 1: Loading Pre-computed UMAP Data ---")
    
    # --- Load the UMAP reducer and embedding map created by your VAE vis script ---
    # NOTE: You must have run your VAE visualization script first to create these files.
    reducer_path = os.path.join(vae_log_dir, "umap_reducer.joblib")
    embedding_map_path = os.path.join(vae_log_dir, "umap_embedding.npy")

    if not os.path.exists(reducer_path) or not os.path.exists(embedding_map_path):
        raise FileNotFoundError(
            "UMAP reducer/embedding not found. Please run the "
            "`gene_autoencoder_latent_visualization.py` script first to generate the map."
        )

    # reducer = joblib.load(reducer_path)
    embedding_map = np.load(embedding_map_path)
    print("Loaded pre-fitted UMAP reducer and full embedding map.")

    # --- Step 2: Load all gating logits and create metadata ---
    print("\n--- Step 2: Loading All Gating Logits ---")
    
    all_logits = []
    patch_metadata = [] # List of {'sample_id': str, 'patch_index': int}
    
    for sample_data_index, sample_id in enumerate(test_dataset.sample_ids):
        logits_path = os.path.join(sampling_dir, f"{sample_id}_gating_logits.pt")
        
        if os.path.exists(logits_path):
            logits = torch.load(logits_path, map_location='cpu')
            all_logits.append(logits)
            
            num_patches = logits.shape[0]
            for patch_idx in range(num_patches):
                patch_metadata.append({'sample_id': sample_id, 'patch_index': patch_idx})

    if not all_logits:
        raise FileNotFoundError("No gating logit files found in the sampling directory.")

    all_logits = torch.cat(all_logits, dim=0)
    num_experts = all_logits.shape[1]
    
    # We need a mapping from the global patch index (in `all_logits`) to its
    # corresponding index in the `embedding_map`. Assuming they are in the same order.
    # A robust check would be to re-calculate latents, but we can assume order for now.
    print(f"Total patches with logits: {len(patch_metadata)}")
    print(f"Total patches in UMAP map: {embedding_map.shape[0]}")
    assert len(patch_metadata) == embedding_map.shape[0], "Mismatch between logit count and UMAP point count!"

    # --- Step 3: Create the Layered Visualization ---
    print("\n--- Step 3: Creating the Visualization ---")
    
    # It's good practice to create the figure and axis objects explicitly
    fig, ax = plt.subplots(figsize=(10, 10)) # Increased height slightly for title and legend
    
    # Layer 1: Plot the full "world map" in faint gray
    ax.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=3, alpha=0.2) # , label='All Test Patches')

    # Create a color palette for the experts
    expert_palette = sns.color_palette("bright", num_experts)
    
    # Layer 2: For each expert, find and plot its top activating patches
    for expert_i in range(num_experts):
        expert_scores = all_logits[:, expert_i]
        top_indices = torch.topk(expert_scores, num_top_patches).indices
        
        top_embeddings_2d = embedding_map[top_indices]
        
        ax.scatter(
            top_embeddings_2d[:, 0],
            top_embeddings_2d[:, 1],
            color=expert_palette[expert_i],
            s=15,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.2,
            label=f'Expert {expert_i}'
        )

    if not PAPER:
        ax.set_title(f'UMAP Visualization of Top {num_top_patches} Activating Patches per Expert', pad=10) # Add padding
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # We position the legend slightly higher to ensure it's clear of the plot.
    ax.legend(
        ncol=1,  # Arrange items in 3 cols and 2 rows
        loc='lower left',       # Anchor point on the legend is its bottom-center
        borderaxespad=0.,         # No padding between axes and legend
        markerscale=2
    )
    
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Use fig.tight_layout() for better automatic spacing
    fig.subplots_adjust(top=0.85)
    
    save_path = os.path.join(save_dir, f"expert_specialization_umap_paper_{PAPER}.png")
    plt.savefig(save_path, dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nExpert specialization UMAP saved to: {save_path}")



def create_comparison_plot(
    sampling_dir: str,
    vae_log_dir: str,
    test_dataset: PrecomputedEmbeddingDataset,
    save_dir: str,
    num_top_patches: int = 100,
    PAPER = False,
    trajectory_log_dir = None,
):
    """
    Creates a side-by-side UMAP visualization:
    1. Left panel: Colored by ground truth oncotree code.
    2. Right panel: Colored by top activating expert.
    """
    print("--- Step 1: Loading All Necessary Data ---")

    # --- Load UMAP data (reducer, map, and now labels) ---
    if trajectory_log_dir is not None:
        print("Using trajectory log dir for UMAP data.")
        reducer_path = os.path.join(trajectory_log_dir, "umap_reducer_test_set.joblib")
        embedding_map_path = os.path.join(trajectory_log_dir, "umap_embedding_map_test_set.npy")
        labels_path = os.path.join(trajectory_log_dir, "umap_labels_map_test_set.npy")
    else:
        print("Using VAE log dir for UMAP data.")
        reducer_path = os.path.join(vae_log_dir, "umap_reducer.joblib")
        embedding_map_path = os.path.join(vae_log_dir, "umap_embedding.npy")
        labels_path = os.path.join(vae_log_dir, "umap_labels.npy") # Assumes you saved this

    if not all(os.path.exists(p) for p in [reducer_path, embedding_map_path, labels_path]):
        raise FileNotFoundError(
            "UMAP reducer, embedding map, or labels not found in VAE log dir. "
            "Please run the VAE visualization script first and ensure it saves these files."
        )

    # reducer = joblib.load(reducer_path)
    embedding_map = np.load(embedding_map_path)
    oncotree_labels = np.load(labels_path)
    print("Loaded UMAP reducer, map, and oncotree labels.")

    # --- Load Gating Logits ---
    all_logits = []
    
    for sample_id in test_dataset.sample_ids:
        logits_path = os.path.join(sampling_dir, f"{sample_id}_gating_logits.pt")
        if os.path.exists(logits_path):
            all_logits.append(torch.load(logits_path, map_location='cpu'))

    if not all_logits:
        raise FileNotFoundError("No gating logit files found.")
    
    all_logits = torch.cat(all_logits, dim=0)
    num_experts = all_logits.shape[1]
    
    assert all_logits.shape[0] == embedding_map.shape[0], "Mismatch between logit count and UMAP point count!"

    # --- Step 2: Create the Side-by-Side Figure ---
    print("\n--- Step 2: Creating the Comparative Visualization ---")
    
    # --- THE CHANGE IS HERE: 2 rows, 1 column ---
    # We choose a figure size that is tall and narrow.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10)) # 2 rows, 1 col; figsize=(width, height)

    # --- Panel 1: UMAP Colored by Oncotree Code (Top Panel) ---
    ax1 = axes[0]
    sns.scatterplot(
        x=embedding_map[:, 0],
        y=embedding_map[:, 1],
        hue=oncotree_labels,
        palette="tab10",
        s=5,
        alpha=0.6,
        ax=ax1,
        legend='full'
    )
    if not PAPER:
        ax1.set_title("A) Latent Space Colored by Cancer Type")
    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")
    # Placing the legend inside the plot is often better for vertical layouts
    ax1.legend(markerscale=5.0, loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- Panel 2: UMAP Colored by Expert Activation (Bottom Panel) ---
    ax2 = axes[1]
    
    # Plot the gray background without a label
    ax2.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=3, alpha=0.1)
    
    expert_palette = sns.color_palette("bright", num_experts)
    
    for expert_i in range(num_experts):
        expert_scores = all_logits[:, expert_i]
        top_indices = torch.topk(expert_scores, num_top_patches).indices
        top_embeddings_2d = embedding_map[top_indices]
        
        ax2.scatter(
            top_embeddings_2d[:, 0],
            top_embeddings_2d[:, 1],
            color=expert_palette[expert_i],
            s=30,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            label=f'Expert {expert_i}'
        )
    if not PAPER:
        ax2.set_title("B) Latent Space Colored by Top Expert Activations")
    ax2.set_xlabel("UMAP Dimension 1")
    ax2.set_ylabel("UMAP Dimension 2")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # Create the legend for the second panel
    ax2.legend(markerscale=2.0, loc='lower left')
    
    # --- CRITICAL: Synchronize Axis Limits ---
    xlim = (embedding_map[:, 0].min() - 1, embedding_map[:, 0].max() + 1)
    ylim = (embedding_map[:, 1].min() - 1, embedding_map[:, 1].max() + 1)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    
    # No need for a separate suptitle; the individual titles are clear
    # Use fig.tight_layout() to automatically manage spacing between the plots
    fig.tight_layout(pad=3.0) # `pad` adds some spacing between subplots and title
    
    save_path = os.path.join(save_dir, f"expert_specialization_comparison_umap_paper_{PAPER}.png")
    plt.savefig(save_path, dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nComparative UMAP saved to: {save_path}")


def visualize_expert_umap_on_top_patches(
    sampling_dir: str,
    save_dir: str,
    test_dataset: PrecomputedEmbeddingDataset,
    num_top_patches: int = 100, # Increased to get a better UMAP
    pfm_embedding_dir = "/mnt/cluster/datasets/HEST1k/univ2/"
):
    """
    Creates a UMAP projection using ONLY the embeddings of the top activating
    patches for each expert.
    """

    # --- Step 1: Load all PFM embeddings and Gating Logits ---
    print("--- Step 1: Loading All PFM Embeddings and Gating Logits ---")
    all_pfm_embeddings = []
    all_logits = []
    
    for sample_id in tqdm(test_dataset.sample_ids, desc="Loading Data"):
        embedding_path = os.path.join(pfm_embedding_dir, f"{sample_id}_embeddings.pt")
        logits_path = os.path.join(sampling_dir, f"{sample_id}_gating_logits.pt")

        if os.path.exists(embedding_path) and os.path.exists(logits_path):
            embeddings = torch.load(embedding_path, map_location='cpu')
            if isinstance(embeddings, dict):
                sorted_items = sorted(embeddings.items())
                embeddings = torch.stack([item[1] for item in sorted_items])
            
            logits = torch.load(logits_path, map_location='cpu')

            # Ensure they have the same number of patches
            if embeddings.shape[0] == logits.shape[0]:
                all_pfm_embeddings.append(embeddings)
                all_logits.append(logits)
            else:
                print(f"Warning: Mismatch in patch count for {sample_id}. Skipping.")
    
    if not all_pfm_embeddings:
        raise FileNotFoundError("No matching PFM embeddings and logit files were loaded.")

    all_pfm_embeddings = torch.cat(all_pfm_embeddings, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    num_experts = all_logits.shape[1]

    # --- Step 2: Select Top Patches and their Embeddings for UMAP ---
    print("\n--- Step 2: Selecting Top Patches for Each Expert ---")
    top_embeddings_list = []
    top_expert_labels = []

    for expert_i in range(num_experts):
        expert_scores = all_logits[:, expert_i]
        top_indices = torch.topk(expert_scores, num_top_patches).indices
        
        # Get the high-dimensional embeddings for these top patches
        top_embeddings = all_pfm_embeddings[top_indices]
        
        top_embeddings_list.append(top_embeddings)
        # Create labels to remember which expert these embeddings belong to
        top_expert_labels.extend([expert_i] * num_top_patches)

    # Create a single dataset to run UMAP on
    umap_input_embeddings = torch.cat(top_embeddings_list, dim=0).numpy()
    umap_input_labels = np.array(top_expert_labels)
    
    print(f"Created a dataset of {umap_input_embeddings.shape[0]} top patches for UMAP.")

    # --- Step 3: Fit UMAP and Create Visualization ---
    print("\n--- Step 3: Fitting UMAP and Creating Visualization ---")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=2025)
    embedding_2d = reducer.fit_transform(umap_input_embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use seaborn for a clean scatter plot with automatic legend creation
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=umap_input_labels,
        palette=sns.color_palette("bright", num_experts),
        s=10,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    
    ax.set_title(f'UMAP of Top {num_top_patches} PFM Embeddings per Expert', pad=70)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # Get the legend from seaborn and customize it
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[f"Expert {l}" for l in labels], # Add "Expert" prefix
        # title="Top Activating Expert",
        ncol=1,
        loc='lower left',
        borderaxespad=0.,
        markerscale=5.0
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ax.grid(True, linestyle='--', alpha=0.4)
    
    save_path = os.path.join(save_dir, "expert_specialization_top_patches_umap.png")
    plt.savefig(save_path, dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nExpert specialization UMAP saved to: {save_path}")



if __name__ == '__main__':
    seed_everything(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_dir", "-s", type=str, required=True)
    parser.add_argument("--trajectory_log_dir", "-t", type=str, required=False)
    parser.add_argument("--paper", "-p", action="store_true", help="Enable paper mode") # if plotting for paper, no titles
    args = parser.parse_args()
    PAPER = args.paper

    sns.set_style("whitegrid")
    sns.set_palette("bright")        # or "deep", "colorblind", etc.
    sns.set_context(
        "paper", font_scale=2
    )  # Options: 'paper', 'notebook', 'talk', 'poster'
    plt.rcParams["figure.dpi"] = 300  # global DPI

    log_dir = os.path.dirname(args.sampling_dir)
    print(log_dir)
    expert_analyisis_dir = os.path.join(args.sampling_dir, "expert_analysis")
    os.makedirs(expert_analyisis_dir, exist_ok=True)

    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."

    with open(os.path.join(log_dir, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    num_top_patches = 100 
    gene_list = load_gene_list(config)
    test_dataset = PrecomputedEmbeddingDataset(config=config, gene_list=gene_list,split="test")

    create_expert_montage(
        sampling_dir=args.sampling_dir,
        num_top_patches=num_top_patches,
        config=config,
        save_dir=expert_analyisis_dir,
        test_dataset = test_dataset
    )

    analyze_expert_utilization(
        sampling_dir=args.sampling_dir,
        config=config,
        k=2,
        save_dir=expert_analyisis_dir,
        test_dataset = test_dataset
    )

    visualize_expert_umap(
        sampling_dir=args.sampling_dir,
        vae_log_dir=config["gene_vae_pretrained_path"],
        save_dir=expert_analyisis_dir,
        num_top_patches = num_top_patches,
        test_dataset = test_dataset,
        PAPER=PAPER
    )

    create_comparison_plot(
        sampling_dir=args.sampling_dir,
        vae_log_dir=config["gene_vae_pretrained_path"],
        test_dataset=test_dataset,
        save_dir=expert_analyisis_dir,
        num_top_patches = num_top_patches,
        PAPER=PAPER,
        trajectory_log_dir = args.trajectory_log_dir,
    )

    visualize_expert_umap_on_top_patches(
        sampling_dir=args.sampling_dir,
        save_dir=expert_analyisis_dir,
        test_dataset=test_dataset,
        num_top_patches = num_top_patches
        )