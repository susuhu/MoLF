import os
import sys
import yaml
import json
import pickle
import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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


if __name__ == "__main__":
    print("[INFO_SAMPLING]Starting sampling...")
    seed_everything(seed=2025)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", type=str)  # logging dir
    parser.add_argument("--method", "-m", default="euler", type=str)  # dorpi5, euler, rk4
    parser.add_argument("--num_steps", "-n", default=2, type=int)  # logging dir
    parser.add_argument("--guidance_scale", "-g", type=float)  # guidance scale
    parser.add_argument("--extern_mouse", "-e", action="store_true")  # validate on external mouse data
    args = parser.parse_args()
    log_dir = args.log_dir  
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."

    cfg_guidance  = args.guidance_scale
    print(f"[INFO_SAMPLING] CFG guidance: {cfg_guidance}")
     
    solver_config = {"method": args.method, "num_steps": args.num_steps}
    print(f"[INFO_SAMPLING] Solver config: {solver_config}")

    extern_mouse = args.extern_mouse

    # load config
    with open(os.path.join(log_dir, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_SAMPLING] Using device: {device}")

    with open(config["paths_config"]["path_way_dict_path"], "rb") as f:
        pathway_dict = pickle.load(f)

    gene_list = load_gene_list(config)
    gene_dim = len(gene_list)

    if extern_mouse:
        split="test_mouse"
    else:
        split="test"
    split_path = os.path.join(config["paths_config"]["splits_path"], f"{split}_split.csv")

    print(f"[INFO_SAMPLING] Using standard dataloader for model {config['model']}")

    test_dataset = PrecomputedEmbeddingDataset(
        config,
        split=split,
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
    print(f"[INFO_SAMPLING] Flow model initialized")

    checkpoint_path = os.path.join(log_dir, "ckpt", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"[INFO_SAMPLING_FLOW_ST] Loading model from {checkpoint_path}")
        start_epoch, best_val_loss, extra_INFO_SAMPLING = load_checkpoint(
            path= checkpoint_path,
            model=model,
            # optimizer=optimizer,
            # scheduler=scheduler,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Please check the path.")
    model.eval()
    print(f"[INFO_SAMPLING] Loaded checkpoint: {checkpoint_path}")

    # ode solver parameters
    # A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5","rk4", "midpoint" and "heun3". For a complete list, see torchdiffeq.
    sample_log_dir = os.path.join(
        log_dir, f"sampling_" + str(args.num_steps) + args.method +"_cfg_guidance_"+ str(cfg_guidance) + "_epoch"+str(start_epoch)+"mouse_"+ str(extern_mouse)
    ) 
    os.makedirs(sample_log_dir, exist_ok=True)
    # Create a unique subfolder in sampling for each run
    log_path = os.path.join(sample_log_dir, "sampling_run.txt")
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout

    loss_fn = torch.nn.MSELoss()


    pred_df = pd.DataFrame(columns=["sample_id"] + gene_list)
    gene_pcc_across_samples = {gene: [] for gene in gene_list}
    gene_MSE_across_samples = {gene: [] for gene in gene_list}
    rows = []
    rows_gt = []
    all_gene_pcc_path = os.path.join(sample_log_dir, "ALL_GENES")
    os.makedirs(all_gene_pcc_path, exist_ok=True)
    with torch.no_grad():
        for batch in test_loader:
            # embeddings, coords, gene_tensor, gene_mask_tensor, oncotree_onehoted, sample_id
            embeddings, _, genes_gt, genes_mask, oncotree_onehoted, sample_id = batch
            sample_id = sample_id[0]  # it's a list
            # 2. Move each required tensor to the device individually
            embeddings = embeddings.to(device)
            genes_gt = genes_gt.to(device)
            # print("DEBUG: genes_gt shape:", genes_gt.shape)  # b x n_patches x n_genes
            oncotree_onehoted = oncotree_onehoted.to(device)
            genes_mask = genes_mask.to(device)

            if config["model"].lower() == "simplemlp":
                predicted_genes = model._get_predictions(x=embeddings)
                initial_gating_logits = None  # No gating logits for this model
            elif config["model"].lower() == "molf":
                # 2. Manually create the initial noise state
                B, P, _ = embeddings.shape
                x_init = torch.randn(B, P, model.gene_latent_dim, device=device)
                t_zero = torch.tensor(0.0, device=device)
                # 3. Call _get_predictions ONCE at t=0 to get the initial logits
                # This gives us the routing decision based on the image and pure noise
                _, initial_gating_logits = model._get_predictions(
                    z_t=x_init,
                    t=t_zero,
                    emb=embeddings,
                    onco_onehot=oncotree_onehoted
                )

                predicted_genes, latent_trajectory = model.sample(
                    emb=embeddings,
                    onco_onehot=oncotree_onehoted,
                    mask=genes_mask,
                    guidance_scale=cfg_guidance,
                    solver_config=solver_config,
                    x_init_override=x_init,
                )
                print("DEBUG: latent_trajectory.shape:", latent_trajectory.shape) # B=1, P, latent_dim

            else:   
                raise ValueError(f"Unknown model type: {config['model']}")

            # Compute MSE between generated and real gene expression
            test_loss = loss_fn(predicted_genes, genes_gt)

            predicted_genes_np = (
                predicted_genes.detach().cpu().numpy()
            )  # n_patches x n_genes
            genes_gt_np = genes_gt.detach().cpu().numpy()
            predicted_genes_np = reduce_batch_dim(predicted_genes_np)
            genes_gt_np = reduce_batch_dim(genes_gt_np)

            for n_patch in range(predicted_genes_np.shape[0]):
                rows.append([sample_id] + predicted_genes_np[n_patch].tolist())
                rows_gt.append([sample_id] + genes_gt_np[n_patch].tolist())

            # all trianing gene pcc
            all_gene_sample_json_path = os.path.join(
                all_gene_pcc_path, f"{sample_id}_gene_pcc.json"
            )
            
            if not os.path.isfile(all_gene_sample_json_path):
                print(f"[INFO_SAMPLING] Computing all-genes PCC for sample {sample_id}")
                pearson_all_gene_dict = {}
                pearson_values_all_genes = []
                for g, gene_name in enumerate(gene_list):
                    try:
                        # get the gene pcc across patches for each sample
                        r, _ = pearsonr(predicted_genes_np[:, g], genes_gt_np[:, g])
                        r = float(r)
                        pearson_all_gene_dict[gene_name] = r
                        gene_pcc_across_samples[gene_name].append(r)
                        pearson_values_all_genes.append(r)
                    except Exception:
                        pearson_all_gene_dict[gene_name] = None
                        gene_pcc_across_samples[gene_name].append(None)
                    avg_pearson_all_genes = (
                        float(np.nanmean([v for v in pearson_values_all_genes if v is not None]))
                        if pearson_values_all_genes
                        else float("nan")
                    )
                sample_all_gene_json_data = {
                    "pearson_corr": pearson_all_gene_dict,
                    "average_pcc": avg_pearson_all_genes,
                }
    
                with open(all_gene_sample_json_path, "w") as f:
                    json.dump(sample_all_gene_json_data, f, indent=2)


            # TODO loop up PCC instead of computing again
            # Sampling/generation loop on pathway level.
            for pathway_name, pathway_genes in pathway_dict.items():
                print(
                    f"[INFO_SAMPLING] Pathway: {pathway_name} | Number of genes in this pathway: {len(pathway_genes)}"
                )
                pathway_log_path = os.path.join(sample_log_dir, pathway_name)
                os.makedirs(pathway_log_path, exist_ok=True)
                
                # Only compute PCC for each pathway genes
                pearson_dict = {}
                pearson_values = []

                # Save per-sample Pearson to JSON
                sample_json_path = os.path.join(
                    pathway_log_path, f"{sample_id}_gene_pcc.json"
                )
                pathway_gene_indexes = [gene_list.index(x) for x in pathway_genes]
                # print(f"DEBUG: pathway_gene_indexes: {pathway_gene_indexes}")
                if not os.path.isfile(sample_json_path):
                    print(f"[INFO_SAMPLING] Computing {pathway_name} PCC for sample {sample_id}")
                    for g, gene_name in zip(pathway_gene_indexes, pathway_genes):
                        # PCC
                        try:
                            r, _ = pearsonr(predicted_genes_np[:, g], genes_gt_np[:, g])
                            r = float(r)
                            # print(f"DEBUG: PCC for gene {gene_name} (index {g}): {r}")
                            pearson_dict[gene_name] = r
                            pearson_values.append(r)
                        except Exception:
                            pearson_dict[gene_name] = None
                    avg_pearson = (
                        float(np.nanmean([v for v in pearson_values if v is not None]))
                        if pearson_values
                        else float("nan")
                    )
                    sample_json_data = {
                        "pearson_corr": pearson_dict,
                        "average_pcc": avg_pearson,
                    }
        
                    with open(sample_json_path, "w") as f:
                        json.dump(sample_json_data, f, indent=2)

                    try:
                        print(
                            f"[INFO_SAMPLING] sample_id={sample_id} | "
                            f"MSE={test_loss.item():.4f} | "
                            f"PCC={avg_pearson:.4f} | "
                            f"PCC saved: {sample_json_path}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"EXCEPTION in print block: {e}", flush=True)
                else:
                    print(f"[INFO_SAMPLING]{sample_id}'s pcc has already been processed.")

                # get pcc per oncotree for each pathway
                get_pcc(pathway_log_path, split_path)

            # --- NEW: Save the logits ---
            logits_save_path = os.path.join(sample_log_dir, f"{sample_id}_gating_logits.pt")
            if initial_gating_logits is not None and not os.path.exists(logits_save_path):
                # Squeeze batch dim before saving, shape becomes [num_patches, num_experts]
                torch.save(initial_gating_logits.squeeze(0), logits_save_path)
                print(f"[INFO_SAMPLING] Gating logits saved to: {logits_save_path}")

        # get oncotree_pcc for ALL_GENE sampling
        get_pcc(all_gene_pcc_path, split_path)

        # pred df is for all samples and all genes
        pred_df = pd.DataFrame(rows, columns=pred_df.columns)
        pred_df.to_csv(os.path.join(sample_log_dir, "predicted_genes.csv"), index=False)
        print(f"[INFO_SAMPLING] Predicted genes saved to: predicted_genes.csv")
        gt_df = pd.DataFrame(rows_gt, columns=pred_df.columns)
        gt_df.to_csv(os.path.join(sample_log_dir, "ground_truth_genes.csv"), index=False)
        print(f"[INFO_SAMPLING] Ground truth genes saved to: ground_truth_genes.csv")

        # Compute average PCC for each gene across all samples and overall average
        # TODO Maybe I don't need this information
        gene_pcc_summary = {}
        gene_averages = []
        for gene in gene_list:
            # Only use valid (not None, not nan) values
            gene_pccs = [
                v
                for v in gene_pcc_across_samples[gene]
                if v is not None and not np.isnan(v)
            ]
            avg_gene_pcc = float(np.mean(gene_pccs)) if gene_pccs else float("nan")
            gene_pcc_summary[gene] = avg_gene_pcc
            gene_averages.append(avg_gene_pcc)
        # Overall average across all genes (and all samples)
        avg_pcc_across_all_genes = (
            float(np.mean([v for v in gene_averages if not np.isnan(v)]))
            if gene_averages
            else float("nan")
        )

        # Save all samples' gene PCCs and summary to JSON
        all_json_path = os.path.join(sample_log_dir, f"all_samples_gene_pcc.json")
        with open(all_json_path, "w") as f:
            json.dump(
                {
                    "per_gene_average": gene_pcc_summary,  # Only one average per gene, plus overall average
                    "average_pcc_across_all_genes": avg_pcc_across_all_genes,
                },
                f,
                indent=2,
            )
        print(f"[INFO_SAMPLING] All samples' gene PCCs and summary saved to: {all_json_path}")

    print("[INFO_SAMPLING] Sampling complete!")
