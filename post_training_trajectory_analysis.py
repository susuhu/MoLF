import os
import sys
import yaml
import json
import pickle
import argparse
import umap
from tqdm import tqdm
import joblib 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio 

import torch
from torch.utils.data import DataLoader

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from utils.hest_utils import load_gene_list
from utils.training_utils import (
    get_emb_dim,
    get_model,
    load_checkpoint,
)
from utils.custom_dataset import PrecomputedEmbeddingDataset
from utils.other_utils import seed_everything


def reduce_batch_dim(input_var):
    if input_var.ndim == 3 and input_var.shape[0] == 1:
        return input_var.squeeze(0)
    else:
        return input_var

def get_pcc(sampling_dir, test_split_path):
    # 1. Loop over sampling directory to get average_pcc across all sample ids
    assert os.path.exists(
        sampling_dir
    ), f"Sampling directory does not exist: {sampling_dir}"
    average_pccs = {}
    for fname in os.listdir(sampling_dir):
        if fname.endswith(f"_gene_pcc.json"):
            if fname != f"all_samples_gene_pcc.json":
                sample_id = fname.replace(f"_gene_pcc.json", "")
                with open(os.path.join(sampling_dir, fname), "r") as f:
                    data = json.load(f)
                    if "average_pcc" in data:
                        average_pccs[sample_id] = data["average_pcc"]
                    else:
                        print(
                            f"Warning: 'average_pcc' not found in {fname}. Skipping this file."
                        )

    # 2. Load test split dataframe
    df = pd.read_csv(test_split_path)

    # 3. Aggregate gene expression data by sample id, group by organ
    df["sample_id"] = df["sample_id"].astype(str)
    df["average_pcc"] = df["sample_id"].map(average_pccs)

    # Report on any samples that did not have a corresponding PCC value
    unmatched_ids = df[df["average_pcc"].isnull()]
    if not unmatched_ids.empty:
        print(
            f"[WARNING_SAMPLING]: {len(unmatched_ids)} samples from the test split did not have a matching PCC file."
        )

    # Group by oncotree_code and aggregate average_pcc, skipping empty codes
    oncotree_pcc = (
        df[
            df["oncotree_code"].notnull()
            & (df["oncotree_code"].astype(str).str.strip() != "")
        ]
        .groupby("oncotree_code")["average_pcc"]
        .mean()
        .reset_index()
    )
    print("Average PCC per oncotree_code (non-empty):",sampling_dir.split('/')[-1])
    print(oncotree_pcc)
    save_filename = os.path.join(sampling_dir, f"oncotree_average_pcc.csv")
    oncotree_pcc.to_csv(save_filename, index=False)
    print(f"\nSuccessfully saved results to {save_filename}")

def create_animation(
    solution_trajectory,
    destinations_2d,
    reducer,
    embedding_map,
    sample_id,
    save_dir,
    start_epoch,
):
    """
    Generates and saves a GIF animation of the latent space trajectory for a single sample.
    """
    print(f"[INFO_ANIMATION] Starting animation creation for sample {sample_id}...")
    num_steps, _, _, _ = solution_trajectory.shape
    animation_path = os.path.join(
        save_dir, f"trajectory_animation_{sample_id}_epoch_{start_epoch}.gif"
    )
    frames_dir = os.path.join(save_dir, f"frames_{sample_id}")
    os.makedirs(frames_dir, exist_ok=True)

    # --- Generate a frame for each time step ---
    for step_idx in tqdm(range(num_steps), desc=f"Generating frames for {sample_id}"):
        time_t = step_idx / (num_steps - 1)

        patches_high_dim_t = solution_trajectory[step_idx, 0, :, :].cpu().numpy()
        patches_2d_t = reducer.transform(patches_high_dim_t)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        # Plot the layers: background, ground truth, and prediction
        ax.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=1, alpha=0.3)
        ax.scatter(destinations_2d[:, 0], destinations_2d[:, 1], c='blue', s=15, marker='*', alpha=0.5, label='Ground Truth')
        ax.scatter(patches_2d_t[:, 0], patches_2d_t[:, 1], c='red', s=10, marker='o', alpha=0.7, label='Prediction')

        # Set fixed axis limits for a stable animation view
        ax.set_xlim(embedding_map[:, 0].min() - 1, embedding_map[:, 0].max() + 1)
        ax.set_ylim(embedding_map[:, 1].min() - 1, embedding_map[:, 1].max() + 1)

        ax.set_title(f'Sample {sample_id} at Time t = {time_t:.2f}', fontsize=16)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend(markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Save the frame to the temporary directory
        frame_path = os.path.join(frames_dir, f"frame_{step_idx:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)  # Crucial to free up memory in a long loop

    # --- Stitch the frames into a GIF ---
    print(f"[INFO_ANIMATION] Stitching frames into GIF: {animation_path}")
    images = []
    for i in range(num_steps):
        filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
        images.append(imageio.imread(filename))
    
    # Make the last frame pause for longer
    durations = [0.1] * (num_steps - 1) + [3.0]
    imageio.mimsave(animation_path, images, duration=durations)

    print(f"[INFO_ANIMATION] Animation saved successfully to {animation_path}")

    # --- Optional: Clean up the individual frame files ---
    # for i in range(num_steps):
    #     os.remove(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    # os.rmdir(frames_dir)


def create_animation_with_hero_path(
    solution_trajectory,
    destinations_2d,
    reducer,
    embedding_map,
    sample_id,
    save_dir,
    start_epoch,
    hero_patch_index=0,  # <-- NEW: Parameter to choose which patch to highlight
):
    """
    Generates a GIF animation that shows the full point cloud moving,
    plus the complete trajectory line for a single "hero" patch.
    """
    print(f"[INFO_ANIMATION] Starting animation with hero path for sample {sample_id}...")
    num_steps, _, _, _ = solution_trajectory.shape
    animation_path = os.path.join(
        save_dir, f"trajectory_animation_hero_{sample_id}_epoch_{start_epoch}.gif"
    )
    frames_dir = os.path.join(save_dir, f"frames_hero_{sample_id}")
    os.makedirs(frames_dir, exist_ok=True)

    # --- NEW: Pre-compute the full 2D path for our hero patch ---
    print(f"[INFO_ANIMATION] Pre-computing hero path for patch {hero_patch_index}...")
    hero_path_high_dim = solution_trajectory[:, 0, hero_patch_index, :].cpu().numpy()
    hero_path_2d = reducer.transform(hero_path_high_dim)

    # --- Generate a frame for each time step ---
    for step_idx in tqdm(range(num_steps), desc=f"Generating frames for {sample_id}"):
        time_t = step_idx / (num_steps - 1)

        # Get the current positions of ALL patches
        all_patches_high_dim_t = solution_trajectory[step_idx, 0, :, :].cpu().numpy()
        all_patches_2d_t = reducer.transform(all_patches_high_dim_t)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        # --- Layered Plotting ---
        # 1. Background Map
        ax.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=1, alpha=0.3)
        # 2. Ground Truth Destinations
        ax.scatter(destinations_2d[:, 0], destinations_2d[:, 1], c='blue', s=15, marker='*', alpha=0.5, label='Ground Truth')
        # 3. All moving patches
        ax.scatter(all_patches_2d_t[:, 0], all_patches_2d_t[:, 1], c='red', s=10, marker='o', alpha=0.6, label='All Patches (t)')
        
        # --- NEW: Overlay the hero patch's full trajectory ---
        # 4. Plot the full path as a semi-transparent line
        ax.plot(hero_path_2d[:, 0], hero_path_2d[:, 1], color='purple', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Patch {hero_patch_index} Full Path')
        # 5. Highlight the hero patch's CURRENT position to make it stand out
        ax.scatter(all_patches_2d_t[hero_patch_index, 0], all_patches_2d_t[hero_patch_index, 1], c='magenta', s=50, marker='o', edgecolor='black', zorder=10, label=f'Patch {hero_patch_index} (t)')
        
        # Set fixed axis limits for a stable animation view
        ax.set_xlim(embedding_map[:, 0].min() - 1, embedding_map[:, 0].max() + 1)
        ax.set_ylim(embedding_map[:, 1].min() - 1, embedding_map[:, 1].max() + 1)

        ax.set_title(f'Sample {sample_id} at Time t = {time_t:.2f}', fontsize=16)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend(markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.6)

        frame_path = os.path.join(frames_dir, f"frame_{step_idx:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)

    # --- Stitch the frames into a GIF (same as before) ---
    print(f"[INFO_ANIMATION] Stitching frames into GIF: {animation_path}")
    images = []
    for i in range(num_steps):
        filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
        images.append(imageio.imread(filename))
    
    durations = [0.1] * (num_steps - 1) + [3.0]
    imageio.mimsave(animation_path, images, duration=durations)
    print(f"[INFO_ANIMATION] Animation saved successfully to {animation_path}")


def plot_static_multi_trajectory(
    solution_trajectory,
    destinations_2d,
    reducer,
    embedding_map,
    sample_id,
    save_dir,
    start_epoch,
    patch_indices_to_viz: list,
    PAPER,
):
    """
    Creates a static grid of plots, where each subplot shows the full
    trajectory of a single selected patch.
    """
    print(f"[INFO_STATIC_PLOT] Creating multi-trajectory static plot for {sample_id}...")
    num_patches_to_plot = len(patch_indices_to_viz)
    
    # Dynamically determine the grid size (e.g., 2x3 for 6 patches) 3x2 for ICML two-column format  
    ncols = 4
    nrows = (num_patches_to_plot + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6), squeeze=False)
    axes = axes.flatten()

    for i, patch_idx in enumerate(patch_indices_to_viz):
        ax = axes[i]
        
        # Pre-compute the full 2D path for this patch
        path_high_dim = solution_trajectory[:, 0, patch_idx, :].cpu().numpy()
        path_2d = reducer.transform(path_high_dim)

        # --- Layered Plotting for this subplot ---
        # 1. Background Map
        ax.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=1, alpha=0.3)
        # 2. All Ground Truth Destinations for the sample
        ax.scatter(destinations_2d[:, 0], destinations_2d[:, 1], c='blue', s=10, marker='*', alpha=0.3, label='Ground Truths Distribution')
        # 3. The specific path for this patch
        ax.plot(path_2d[:, 0], path_2d[:, 1], color='purple', linestyle='-', linewidth=2, label='Trajectory')
        # 4. Highlight the start, end, and true destination for this specific path
        ax.scatter(path_2d[0, 0], path_2d[0, 1], c='green', s=80, marker='o', edgecolor='black', zorder=10, label='Start (t=0)')
        ax.scatter(path_2d[-1, 0], path_2d[-1, 1], c='red', s=80, marker='X', edgecolor='black', zorder=10, label='End (t=1)')
        ax.scatter(destinations_2d[patch_idx, 0], destinations_2d[patch_idx, 1], c='cyan', s=120, marker='*', edgecolor='black', zorder=10, label='Patch Ground Truth')
        
        ax.set_title(f'Trajectory for Patch #{patch_idx}')
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.legend(loc="upper right", markerscale=2)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    if not PAPER:
        fig.suptitle(f'Selected Patch Trajectories for Sample {sample_id}', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"trajectory_static_multi_{sample_id}_epoch_{start_epoch}_paper_{PAPER}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"[INFO_STATIC_PLOT] Plot saved to {save_path}")

if __name__ == "__main__":
    print("[INFO_TRAJECTORY]Starting sampling...")
    seed_everything(seed=2025)
    
    sns.set_style("whitegrid")
    sns.set_palette("bright")        # or "deep", "colorblind", etc.
    sns.set_context(
        "paper", font_scale=2
    )  # Options: 'paper', 'notebook', 'talk', 'poster'
    plt.rcParams["figure.dpi"] = 300  # global DPI

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", type=str)  # logging dir
    parser.add_argument("--method", "-m", default="euler", type=str)  # dorpi5, euler, rk4
    parser.add_argument("--num_steps", "-n", default=2, type=int)  # logging dir
    parser.add_argument("--guidance_scale", "-g", type=float)  # logging dir
    parser.add_argument("--paper", "-p", action="store_true", help="Enable paper mode") # if plotting for paper, no titles
    args = parser.parse_args()
    log_dir = args.log_dir  
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."
    PAPER = args.paper

    cfg_guidance  = args.guidance_scale
    print(f"[INFO_SAMPLING] CFG guidance: {cfg_guidance}")
     
    solver_config = {"method": args.method, "num_steps": args.num_steps}

    # load config
    with open(os.path.join(log_dir, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_TRAJECTORY] Using device: {device}")

    with open(config["paths_config"]["path_way_dict_path"], "rb") as f:
        pathway_dict = pickle.load(f)

    gene_list = load_gene_list(config)
    gene_dim = len(gene_list)
    split_path = os.path.join(config["paths_config"]["splits_path"], "test_split.csv")
    split_df = pd.read_csv(split_path)

    print(f"[INFO_TRAJECTORY] Using standard dataloader for model {config['model']}")
    test_dataset = PrecomputedEmbeddingDataset(
        config,
        split="test",
        gene_list=gene_list,
        single_oncotree_code=None,
        DEBUG=config["debug"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    emb_dim = get_emb_dim(config)

    flow_path = AffineProbPath(scheduler=CondOTScheduler())
    model = get_model(config, gene_dim, emb_dim, device, flow_path)
    model.to(device)
    print(f"[INFO_TRAJECTORY] Flow model initialized")

    checkpoint_path = os.path.join(log_dir, "ckpt", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"[INFO_TRAJECTORY] Loading model from {checkpoint_path}")
        start_epoch, best_val_loss, extra_info = load_checkpoint(
            path= checkpoint_path,
            model=model,
            # optimizer=optimizer,
            # scheduler=scheduler,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Please check the path.")
    model.eval()
    print(f"[INFO_TRAJECTORY] Loaded checkpoint: {checkpoint_path}")

    # ode solver parameters
    # A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5","rk4", "midpoint" and "heun3". For a complete list, see torchdiffeq.
    # trajectory_log_dir = os.path.join(
    #     log_dir, f"trajectory_" + str(solver_config["num_steps"]) + solver_config["method"] + "_epoch"+str(start_epoch)
    # ) 
    trajectory_log_dir = os.path.join(
        log_dir, f"trajectory_" + str(solver_config["num_steps"]) + solver_config["method"] +"_cfg_guidance_"+ str(cfg_guidance) + "_epoch"+str(start_epoch)
    ) 
    os.makedirs(trajectory_log_dir, exist_ok=True)
    # Create a unique subfolder in sampling for each run
    log_path = os.path.join(trajectory_log_dir, "trajectory_analysis_epoch"+str(start_epoch)+".txt")
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout

    # --- Define the file path for your saved reducer ---
    UMAP_REDUCER_PATH = os.path.join(trajectory_log_dir, "umap_reducer_test_set.joblib")
    UMAP_EMBEDDING_PATH = os.path.join(trajectory_log_dir, "umap_embedding_map_test_set.npy") # Also save the pre-computed map
    UMAP_LABELS_PATH = os.path.join(trajectory_log_dir, "umap_labels_map_test_set.npy")

    # --- New Logic: Check if the reducer and map already exist ---
    if os.path.exists(UMAP_REDUCER_PATH) and os.path.exists(UMAP_EMBEDDING_PATH) and os.path.exists(UMAP_LABELS_PATH):
        print(f"[INFO_TRAJECTORY] Loading pre-fitted UMAP reducer from {UMAP_REDUCER_PATH}...")
        reducer = joblib.load(UMAP_REDUCER_PATH)
        print(f"[INFO_TRAJECTORY] Loading pre-computed UMAP embedding map from {UMAP_EMBEDDING_PATH}...")
        embedding_map = np.load(UMAP_EMBEDDING_PATH)
        print("[INFO_TRAJECTORY] UMAP map and reducer loaded successfully.")
        labels = np.load(UMAP_LABELS_PATH)
        print("[INFO_TRAJECTORY] UMAP labels loaded successfully.")

    else:
        print("No pre-fitted UMAP reducer found. Creating a new one...")
        # --- Step 1: Create the "Map" by collecting all ground truth latents (Your existing code) ---
        all_gt_z_means = []
        all_labels = [] 
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Creating UMAP Map"):
                _, _, genes_gt, genes_mask, _, sample_id = batch
                genes_gt = genes_gt.to(device)
                genes_mask = genes_mask.to(device)
                sample_id = sample_id[0]

                z_mean_batch, _ = model.gene_encode(genes_gt, genes_mask)
                all_gt_z_means.append(z_mean_batch.squeeze(0).cpu().numpy())

                n_patches = z_mean_batch.shape[1]
                label = split_df.loc[split_df["sample_id"] == sample_id, "oncotree_code_filled"].iloc[0]
                all_labels.extend([label] * n_patches)

        all_gt_z_means = np.concatenate(all_gt_z_means, axis=0)
        labels = np.array(all_labels)
        
        # Create and fit the UMAP reducer. This learns the 2D projection.
        print("[INFO_TRAJECTORY] Fitting new UMAP reducer...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=2025)
        embedding_map = reducer.fit_transform(all_gt_z_means)
        print("[INFO_TRAJECTORY] Fit complete.")

        # --- SAVE the reducer and the map for next time ---
        print(f"[INFO_TRAJECTORY] Saving UMAP reducer to {UMAP_REDUCER_PATH}...")
        joblib.dump(reducer, UMAP_REDUCER_PATH)
        print(f"[INFO_TRAJECTORY] Saving UMAP embedding map to {UMAP_EMBEDDING_PATH}...")
        np.save(UMAP_EMBEDDING_PATH, embedding_map)
        print("[INFO_TRAJECTORY] UMAP map and reducer saved successfully.")
        np.save(UMAP_LABELS_PATH, labels)
        print("[INFO_TRAJECTORY] UMAP map labels saved successfully.")

    # --- Step 2 & 3: Generate a trajectory for one sample and plot it ---
    print("[INFO_TRAJECTORY] Step 2: Generating and plotting a multi-patch trajectory...")
    with torch.no_grad():
        for sample_batch in test_loader:
            embeddings, _, genes_gt, genes_mask, oncotree_onehoted, sample_id = sample_batch
            # Add the batch dimension to each tensor
            sample_id = sample_id[0]


            # --- Check for static plot ---
            static_save_file = os.path.join(trajectory_log_dir, f"trajectory_snapshots_{sample_id}_epoch_{start_epoch}_PAPER_{PAPER}.png")
            
            # --- Check for animation ---
            animation_save_file = os.path.join(trajectory_log_dir, f"trajectory_animation_{sample_id}_epoch_{start_epoch}.gif")
            # animation_with_hero_save_file = os.path.join(trajectory_log_dir, f"trajectory_animation_hero_{sample_id}_epoch_{start_epoch}.gif")

            static_multi_trajecotry_save_file = os.path.join(trajectory_log_dir, f"trajectory_static_multi_{sample_id}_epoch_{start_epoch}_PAPER_{PAPER}.png")
            
            # save_file_name = os.path.join(trajectory_log_dir,f"trajectory_visualization_{sample_id}_epoch_{start_epoch}.png")
            if not os.path.isfile(static_save_file) or not os.path.isfile(animation_save_file) or not os.path.isfile(static_multi_trajecotry_save_file):
                
                # Move tensors to device
                embeddings, genes_gt, oncotree_onehoted, genes_mask = [t.to(device) for t in (embeddings, genes_gt, oncotree_onehoted, genes_mask)]

                # Generate the trajectory for all patches in the sample
                solver_config['return_intermediates'] = True
                if config["model"].lower() == "molf":
                    predicted_genes, solution_trajectory = model.sample(
                        emb=embeddings,
                        onco_onehot=oncotree_onehoted,
                        mask=genes_mask,
                        solver_config=solver_config,
                        guidance_scale=cfg_guidance,
                        return_intermediates=True,
                    )
                else:
                    raise ValueError(f"Unknown model type: {config['model']}")

                # Shape: [num_steps, 1, num_patches, latent_dim]
                print(f"DEBUG: Generated trajectory shape: {solution_trajectory.shape}") # n_time_step,B=1,P, latent_dim

                # --- Prepare data for plotting ---
                num_steps, _, num_patches, _ = solution_trajectory.shape
                
                # Get the ground truth destinations for all patches
                z_mean_gt, _ = model.gene_encode(genes_gt, genes_mask)
                destinations_high_dim = z_mean_gt.squeeze(0).cpu().numpy()
                destinations_2d = reducer.transform(destinations_high_dim)

                if not os.path.isfile(static_save_file):
                    # Define the time steps we want to visualize
                    if num_steps == 2:
                        snapshot_indices = [0, num_steps-1]
                    else:
                        snapshot_indices = [0, int(0.25*num_steps), int(0.5*num_steps), num_steps-1]
                    # --- Create the Visualization ---
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    axes = axes.flatten()

                    for i, step_idx in enumerate(snapshot_indices):
                        ax = axes[i]
                        time_t = step_idx / (num_steps - 1)
                        
                        # Get the high-dimensional positions of all patches at this time step
                        patches_high_dim_t = solution_trajectory[step_idx, 0, :, :].cpu().numpy()
                        
                        # Project them onto our 2D map
                        patches_2d_t = reducer.transform(patches_high_dim_t)
                        
                        # Plot the background map
                        ax.scatter(embedding_map[:, 0], embedding_map[:, 1], c='lightgray', s=1, alpha=0.3)
                        
                        # Plot the ground truth destinations in a faint color
                        ax.scatter(destinations_2d[:, 0], destinations_2d[:, 1], c='blue', s=15, marker='*',  alpha=0.7, label='Ground Truth')

                        # Plot the current positions of the patches
                        ax.scatter(patches_2d_t[:, 0], patches_2d_t[:, 1], c='red', s=10, marker='o', alpha=0.7, label='Prediction')
                        
                        ax.set_title(f"All Patches at Time t = {time_t:.2f}")
                        ax.set_xlabel("UMAP Dimension 1")
                        ax.set_ylabel("UMAP Dimension 2")
                        ax.legend(markerscale=4)
                        ax.grid(True, linestyle='--', alpha=0.6)
                    if not PAPER:
                        fig.suptitle(f'Latent Space Trajectory Snapshots for Sample {sample_id}', fontsize=20)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(static_save_file)
                    plt.show()
                    print("[INFO_TRAJECTORY] Trajectory visualization is saved at",static_save_file)
                else:
                    print(f"[INFO_TRAJECTORY] {sample_id} was already processed.")
                
                if not os.path.isfile(static_multi_trajecotry_save_file):
                    print("[INFO_TRAJECTORY] Creating static multi-trajectory plot for sample",sample_id)
                    # --- NEW: Call the static multi-trajectory plotting function ---
                    patch_indices_to_viz = [0, num_patches//2, num_patches - 200, num_patches-100]  # Select a few patches to visualize
                    plot_static_multi_trajectory(
                        solution_trajectory=solution_trajectory,
                        destinations_2d=destinations_2d,
                        reducer=reducer,
                        embedding_map=embedding_map,
                        sample_id=sample_id,
                        save_dir=trajectory_log_dir,
                        start_epoch=start_epoch,
                        patch_indices_to_viz=patch_indices_to_viz,
                        PAPER=PAPER,
                    )

            else:
                print(f"[INFO_TRAJECTORY] {sample_id} was already processed.")                
