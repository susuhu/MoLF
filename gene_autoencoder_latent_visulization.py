import os
import argparse
import joblib
import yaml

import numpy as np
import pandas as pd
import umap

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import torch
from torch.utils.data import DataLoader

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath


from models.flow_matching import GeneTransformerAutoencoder
from utils.custom_dataset import GeneExpressionDataset

from utils.other_utils import seed_everything
from utils.hest_utils import load_gene_list
from utils.training_utils import load_checkpoint, get_model, get_emb_dim

if __name__=="__main__":
    print("[INFO_SAMPLING]Starting sampling...")
    seed_everything(seed=2025)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", type=str)  # logging dir
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")  # clean base
    plt.style.use("ggplot")  # add ggplot color and background

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_GENE_VAE] Using device: {device}")

    # encoder_path = "/mnt/nct-zfs/ST_Gen/training_gene_autoencoder_log/run_20250908_145229"
    encoder_path = args.log_dir
    config_path = os.path.join(encoder_path, "config.yml")

    with open(config_path,"r") as f:
        config = yaml.safe_load(f)

    # Load gene list
    gene_list = load_gene_list(config)
    num_genes = len(gene_list)
    emb_dim = get_emb_dim(config)

    # Instantiate the model
    if config["model"].lower() == "genetransformerautoencoder":
        model = GeneTransformerAutoencoder(n_genes=num_genes,
                                        model_dim=config["gene_autoencoder"]["model_dim"],
                                        latent_dim=config["gene_autoencoder"]["latent_dim"],
                                        n_heads=config["gene_autoencoder"]["n_heads"],
                                        n_layers=config["gene_autoencoder"]["n_layers"],
                                        dropout=config["dropout_rate"])
        encoder_ckpt = os.path.join(encoder_path,"ckpt","best_autoencoder.pth")
    elif config["model"].lower() == "latentflow":
        flow_path = AffineProbPath(scheduler=CondOTScheduler())
        model = get_model(config, num_genes, emb_dim, device, flow_path)
        encoder_ckpt = os.path.join(encoder_path,"ckpt","best_model.pt")
    else:
        raise NotImplementedError
    
    if os.path.exists(encoder_ckpt):
        print(f"[INFO_GENE_VAE] Loading model from {encoder_ckpt}")
        start_epoch, _, _ = load_checkpoint(
            path= encoder_ckpt,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise FileNotFoundError(
            f"[ERROR] Checkpoint file {encoder_ckpt} does not exist. Cannot continue training."
        )


    # dataloader
    split = "test"
    split_path = os.path.join(config["paths_config"]["splits_path"],split+"_split.csv")
    split_df = pd.read_csv(split_path)

    test_dataset = GeneExpressionDataset(config=config, 
                                          split=split,
                                          gene_list=gene_list)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    num_test = len(test_dataset)
    print(f"[INFO_GENE_VAE] There are {num_test} test samples.")

    gene_latent_dim = model.latent_dim if config["model"] == "genetransformerautoencoder" else model.gene_latent_dim
    print("[INFO_GENE_VAE] gene_latent_dim", gene_latent_dim)

    # inference
    model.to(device)
    model.eval()
    all_latents = []
    all_labels = [] 
    with torch.no_grad():
        for data in test_loader:
            x, x_gt_mask,sample_id = data
            x = x.to(device)
            x_gt_mask = x_gt_mask.to(device)
            sample_id = sample_id[0]

            if config["model"].lower() == "genetransformerautoencoder":
                recon, z_mean, z_log_var = model(x, x_gt_mask)
            elif config["model"].lower() == "latentflow":
                z_mean, z_log_var = model.gene_encode(x, x_gt_mask)
                
            else:
                raise NotImplementedError
            
            z_mean_flat = z_mean.reshape(-1, gene_latent_dim)  # flatten patches
            n_patches = z_mean_flat.shape[0]

            label = split_df.loc[split_df["sample_id"] == sample_id, "oncotree_code_filled"].iloc[0]
            all_latents.append(z_mean_flat.cpu())
            all_labels.extend([label] * n_patches)

    latents = torch.cat(all_latents, dim=0).numpy()
    labels = np.array(all_labels)
    print("[INFO_GENE_VAE]  latents.shape, labels.shape", latents.shape, labels.shape)

    # plotting UMAP
    print("[INFO_GENE_VAE] Plotting UMAP")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=2025)
    embedding_2d = reducer.fit_transform(latents)
    
    # --- SAVE the reducer embedding_2d  for next time ---
    UMAP_REDUCER_PATH = os.path.join(encoder_path,"umap_reducer.joblib")
    UMAP_EMBEDDING_PATH =  os.path.join(encoder_path,"umap_embedding.npy")
    UMAP_LABELS_PATH =  os.path.join(encoder_path,"umap_labels.npy")
    print(f"[INFO_TRAJECTORY] Saving UMAP reducer to {UMAP_REDUCER_PATH}...")
    joblib.dump(reducer, UMAP_REDUCER_PATH)
    print(f"[INFO_TRAJECTORY] Saving UMAP embedding map to {UMAP_EMBEDDING_PATH}...")
    np.save(UMAP_EMBEDDING_PATH, embedding_2d)
    np.save(UMAP_LABELS_PATH, labels)
    print("[INFO_TRAJECTORY] UMAP map labels saved successfully.")
    
    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x=embedding_2d[:,0],
        y=embedding_2d[:,1],
        hue=labels,                # strings are fine here!
        palette="tab10",
        s=10,
        alpha=0.7
    )
    plt.title("Gene VAE Latent Space (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Increase legend marker size
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Oncotree code",
        markerscale=2.5,   # scale the markers in the legend
        fontsize=10
    )

    plt.tight_layout()
    save_file_name = os.path.join(encoder_path,split+"_umap_epoch_"+ str(start_epoch) +".png")
    plt.savefig(save_file_name,dpi=300)
    print("[INFO_GENE_VAE] UMAP saved to ",save_file_name)

    # TSNE
    print("[INFO_GENE_VAE] Plotting TSNE")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',   # new API: automatically picks a stable learning rate
        init='pca',             # better initialization
        metric='cosine',        # often works better for embeddings
        random_state=2025,
    )

    tsne_emb = tsne.fit_transform(latents)

    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x=tsne_emb[:,0],
        y=tsne_emb[:,1],
        hue=labels,             # strings supported
        palette='tab10',
        s=10,
        alpha=0.7
    )
    plt.title("Gene VAE Latent Space (t-SNE)", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Make legend markers bigger
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title="Oncotree code",
        markerscale=2.5,  # bigger dots
        fontsize=10
    )

    plt.tight_layout()
    save_file_name = os.path.join(encoder_path,split+"_tsne_epoch_"+ str(start_epoch) +".png")
    plt.savefig(save_file_name,dpi=300)
    print("[INFO_GENE_VAE] TSNE saved to ",save_file_name)
    


