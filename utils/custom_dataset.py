"""
Preprocess WSIs into patch-level embeddings using pathology foundation models (PFM)
with corresponding barcode in a python dictionary, which can be matched to gene expression data.

"""

import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

import pandas as pd
import numpy as np
import h5py
import scanpy as sc

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


from utils.hest_utils import (
    read_assets_from_h5,
    load_adata,
    create_spatial_knn_edges
)
from utils.other_utils import get_nested
from utils.hest_utils import load_gene_list,normalize_adata


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


def onehot_oncotree(code, all_codes):
    """
    Returns a one-hot torch tensor for a single code given the full list of all_codes.
    Example:
        all_codes = ['BRCA', 'LUAD', 'healthy', 'unknown']
        code = 'LUAD'
        onehot = onehot_oncotree_torch(all_codes, code)
        # onehot: tensor([0., 1., 0., 0.])
    """
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    idx = code_to_idx.get(code, -1)
    onehot = torch.zeros(len(all_codes), dtype=torch.float32)
    if idx >= 0:
        onehot[idx] = 1.0
    return onehot

    
    
class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, gene_list, single_oncotree_code=None, DEBUG=False):
        split_path = os.path.join(get_nested(config,["paths_config","splits_path"]), f"{split}_split.csv")
        df = pd.read_csv(split_path, header=0)
        if single_oncotree_code:
            print(f"[INFO custom datset] Load only oncotree {single_oncotree_code}.")
            df = df[df["oncotree_code"] == single_oncotree_code]
        self.sample_ids = df["sample_id"].tolist()
        self.oncotree_code = df["oncotree_code"].tolist()
        self.embeddings_path = os.path.join("/mnt/cluster/datasets/HEST1k/", config["PFM_name"])
        if DEBUG:
            self.sample_ids = self.sample_ids[:5]
            # self.oncotree_code = self.oncotree_code[:2]
        self.gene_list = gene_list
        self.patches_path = get_nested(config,["paths_config","patches_path"])
        self.st_path = get_nested(config,["paths_config","st_path"])
        self.oncotree_list_path = os.path.join(get_nested(config,["paths_config","splits_path"]), "oncotree_code_list.txt")
        with open(self.oncotree_list_path, "r") as f:
            self.all_codes = [line.strip() for line in f]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        oncotree_code = self.oncotree_code[idx]
        if oncotree_code not in self.all_codes:
            print("[WARNING] Oncotree code not found in all_codes list, using 'Unknown' instead.")
            oncotree_code = "Unknown"
        oncotree_onehoted = onehot_oncotree(oncotree_code, self.all_codes).unsqueeze(0)

        # Path to the pre-processed file
        embedding_file = os.path.join(self.embeddings_path, f"{sample_id}_embeddings.pt")
        
        # Load the pre-computed data (fast I/O)
        patches_transformed_dict = torch.load(embedding_file, weights_only=True)
        
        # Load the gene data
        try:
            gene_array, gene_mask_array = get_expression_and_mask(
                os.path.join(self.st_path, f"{sample_id}.h5ad"),
                patches_transformed_dict,
                self.gene_list,
            )
            gene_tensor = torch.tensor(gene_array, dtype=torch.float32)
            gene_mask_tensor = torch.tensor(gene_mask_array, dtype=torch.float32) # Pass mask as float
        except Exception as e:
            raise ValueError(f"Error processing sample {sample_id}: {e}")
        
        embeddings = torch.stack(list(patches_transformed_dict.values())).squeeze(1)

        # Load the asset from the HDF5 file
        asset_path = os.path.join(self.patches_path, f"{sample_id}.h5")
        if not os.path.exists(asset_path):
            print(asset_path)
            raise FileNotFoundError(
                f"[ERROR_dataset:]Asset file {asset_path} does not exist."
            )
        with h5py.File(asset_path, "r") as f:
            # Assuming the asset is stored in a dataset named 'asset'
            assets, _ = read_assets_from_h5(asset_path)
        coords = assets["coords"]

        return embeddings, coords, gene_tensor, gene_mask_tensor, oncotree_onehoted, sample_id


class HESTGraphDataset(PrecomputedEmbeddingDataset):
    def __init__(self, *args, knn=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn = knn

    def __getitem__(self, idx):
        # Use parent class to load base data
        embeddings, coords, gene_tensor, onco_onehot, sample_id = super().__getitem__(
            idx
        )

        # Create graph edges
        edge_index = create_spatial_knn_edges(coords, k=self.knn)
        edge_index = edge_index.detach().clone().long()

        # Wrap everything in a PyG Data object
        data = Data(
            x=embeddings,  # node features
            edge_index=edge_index,  # edges
            y=gene_tensor,  # target gene expression
        )
        data.sample_id = sample_id
        data.onco = onco_onehot  # additional global info

        return data


def get_gene_array(exp_path, embeddings_dict, gene_list):
    # Fix barcode decoding
    barcodes = []
    for b in embeddings_dict.keys():
        if isinstance(b, (list, np.ndarray)) and len(b) == 1:
            b = b[0]
        barcodes.append(decode_barcode(b))

    # Load AnnData
    gene_df = load_adata(
        exp_path, genes=None, barcodes=barcodes, normalize=True
    )  # Load all genes
    
    # Prepare output array: [n_barcodes, n_genes]
    n_barcodes = len(barcodes)
    n_genes = len(gene_list)

    # gene_array = np.zeros((n_barcodes, n_genes), dtype=np.float32)
    gene_array = np.full((n_barcodes, n_genes), np.nan, dtype=np.float32)

    # Map gene names to columns in AnnData
    adata_genes = list(gene_df.columns)
    gene_to_idx = {g: i for i, g in enumerate(adata_genes)}

    for j, gene in enumerate(gene_list):
        if gene in gene_to_idx:
            # Get the column index for the current gene from the loaded data
            source_idx = gene_to_idx[gene]
            # Copy the expression values for this gene into the correct column of our output array
            gene_array[:, j] = gene_df.iloc[:, source_idx].to_numpy(dtype=np.float32)
        # else: leave as np.nan
    return gene_array


def get_expression_and_mask(
    exp_path: str, 
    embeddings_dict: dict, 
    joint_gene_list: list, 
    pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads spatially-resolved gene expression for a sample and creates a padded
    expression array and a corresponding binary mask based on a joint gene list.

    Args:
        exp_path (str): Path to the AnnData (.h5ad) file for the sample.
        embeddings_dict (dict): Dictionary mapping spatial locations to barcodes.
        joint_gene_list (list): The master list of all possible genes (the union of all pathways).
                                This defines the final output dimension.
        pad_value (float): The numerical value to use for padding. Defaults to 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - expression_array (np.ndarray): Array of shape [n_patches, n_joint_genes]
              with non-existing genes filled with `pad_value`.
            - mask_array (np.ndarray): Array of shape [n_patches, n_joint_genes] with
              1.0 for measured genes and 0.0 for padded ones.
    """
    # --- 1. Barcode and Data Loading (same as before) ---
    barcodes = []
    for b in embeddings_dict.keys():
        if isinstance(b, (list, np.ndarray)) and len(b) == 1:
            b = b[0]
        # Assuming decode_barcode is defined
        barcodes.append(decode_barcode(b))

    # Load AnnData for the specified barcodes.
    # This DataFrame `gene_df` has shape (n_patches, n_genes_available_in_this_sample)
    gene_df = load_adata(
        exp_path, genes=None, barcodes=barcodes, normalize=True
    )
    
    # --- 2. Initialization ---
    n_patches = len(barcodes)
    n_joint_genes = len(joint_gene_list)

    # Initialize expression array with the padding value (e.g., 0.0)
    expression_array = np.full((n_patches, n_joint_genes), pad_value, dtype=np.float32)
    
    # Initialize mask array with 0s (all padded by default)
    mask_array = np.zeros((n_patches, n_joint_genes), dtype=np.float32) # Use float32 for consistency with PyTorch

    # --- 3. Efficiently Identify and Fill Data ---
    
    # Create a mapping from the joint list gene symbol to its final column index
    joint_gene_to_idx = {gene: i for i, gene in enumerate(joint_gene_list)}
    
    # Find the intersection of genes: which genes from our master list
    # are actually available in this sample's AnnData file?
    available_genes_in_sample = list(gene_df.columns)
    genes_to_fill = [gene for gene in joint_gene_list if gene in available_genes_in_sample]
    
    # If there are no genes to fill, return the zero-initialized arrays
    if not genes_to_fill:
        print(f"Warning: No genes from the joint list found in sample {exp_path}. Returning zero arrays.")
        return expression_array, mask_array

    # --- 4. Vectorized Data Transfer ---
    
    # Get the column indices in our final output arrays for the genes we need to fill
    output_indices = [joint_gene_to_idx[gene] for gene in genes_to_fill]
    
    # Select the data from the loaded gene_df in the correct order
    # This creates a NumPy array of shape (n_patches, n_genes_to_fill)
    expression_values_to_copy = gene_df[genes_to_fill].to_numpy(dtype=np.float32)
    
    # Place the real expression data into the correct columns of the output array
    expression_array[:, output_indices] = expression_values_to_copy
    
    # Set the corresponding positions in the mask to 1 (indicating real data)
    mask_array[:, output_indices] = 1.0

    return expression_array, mask_array


def load_adata_gene_only(expr_path, genes=None, barcodes=None, normalize=True):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        # drop duplicated genes
        adata = adata[:, genes]
    adata.var_names_make_unique()
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()


class GeneExpressionDataset(Dataset):
    def __init__(self, config, split, gene_list):
        self.st_path = config["paths_config"]["st_path"]
        self.gene_list = gene_list
        self.csv_path = os.path.join(config["paths_config"]["splits_path"], f"{split}_split.csv")
        self.embeddings_path = os.path.join("/mnt/cluster/datasets/HEST1k/", config["PFM_name"])
        self.csv = pd.read_csv(self.csv_path, header=0)
        self.sample_ids = self.csv["sample_id"].tolist()
 
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        # Path to the pre-processed file
        embedding_file = os.path.join(self.embeddings_path, f"{sample_id}_embeddings.pt")
        patches_transformed_dict = torch.load(embedding_file, weights_only=True)
        gene_array, gene_mask_array = get_expression_and_mask(
            os.path.join(self.st_path, f"{sample_id}.h5ad"),
            patches_transformed_dict,
            self.gene_list,
        )
        gene_tensor = torch.tensor(gene_array, dtype=torch.float32)
        gene_mask_tensor = torch.tensor(gene_mask_array, dtype=torch.float32) # Pass mask as float
        
        return gene_tensor, gene_mask_tensor, sample_id # [n_patches,num_genes]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str
    )  # ["uni","univ2", "hoptimus0", "virchow2"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_uni] Using device: {device}")


if __name__=="__main__":

    config_path = "/mnt/nct-zfs/ST_Gen/configs/training_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load gene list
    gene_list = load_gene_list(config)

    train_dataset = GeneExpressionDataset(config=config, 
                                          split="train",
                                          gene_list=gene_list)
    val_dataset = GeneExpressionDataset(config=config, 
                                          split="val",
                                          gene_list=gene_list)