"""
HEST1k data ujtils
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob

import h5py
import json
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
import torch

import scanpy as sc
from utils.other_utils import get_nested


# from HEST1K repository
def save_hdf5(
    output_fpath, asset_dict, attr_dict=None, mode="a", auto_chunk=True, chunk_size=None
):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:  # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_:
                    data_type = h5py.string_dtype(encoding="utf-8")
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        chunks=chunks,
                        maxshape=(None,) + data_shape[1:],
                        dtype=data_type,
                    )
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")

            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0] :] = va

    return output_fpath


# from HEST1K repository
def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs


# from HEST1K repository
def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    # print(adata.obs)
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row["array_row"])
            col = int(df_row["array_col"])
            neighbors_index = adata.obs[
                (
                    (adata.obs["array_row"] >= row - 1)
                    & (adata.obs["array_row"] <= row + 1)
                )
                & (
                    (adata.obs["array_col"] >= col - 1)
                    & (adata.obs["array_col"] <= col + 1)
                )
            ].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)

            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg

    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)

    return filtered_adata


# from HEST1K repository
def load_adata(expr_path, genes=None, barcodes=None, normalize=True):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        # drop duplicated genes
        adata.var_names_make_unique()
        adata = adata[:, genes]
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()



def get_target_genes_idx(train_gene_list, target_genes):
    """
    get target genes names, and gene ground truth from hest_bench, for metrics calculation
    """

    # get indexes
    target_genes_indices_in_genes_gt = [
        train_gene_list.index(gene) for gene in target_genes if gene in train_gene_list
    ]
    target_genes_filtered = [gene for gene in target_genes if gene in train_gene_list]
    return target_genes_filtered, target_genes_indices_in_genes_gt



def load_gene_list(config):
    with open(get_nested(config,["paths_config","gene_list_path"])) as f:
        gene_list = yaml.safe_load(f)["genes"]
    return gene_list


def load_metadata(config_path):
    # Load the metadata
    config = yaml.safe_load(open(config_path))
    return config


def get_all_oncotree_list(split_path):
    oncotree_codes_list = []
    for csv_path in glob.glob(f"{split_path}/*_split.csv"):
        df = pd.read_csv(csv_path)
        oncotree_codes_list.append(df["oncotree_code_filled"].tolist())
    
    flattened = [item for sublist in oncotree_codes_list for item in (sublist if isinstance(sublist, list) else [sublist])]

    oncotree_codes_list=list(set(flattened))
    with open(os.path.join(split_path,"oncotree_code_list.txt"), "w") as f:
        for item in oncotree_codes_list:
            f.write(f"{item}\n")


def plot_data_distribution(split_path, criterion= "oncotree_code_filled"):
    # Dictionary to store counts
    all_counts = {}

    for csv_path in glob.glob(f"{split_path}/*_split.csv"):
        df = pd.read_csv(csv_path)
        # Count occurrences of each oncotree_code
        counts = df[criterion].value_counts()
        # Use file name (without path and extension) as x label
        name = os.path.splitext(os.path.basename(csv_path))[0]
        all_counts[name] = counts

    # Combine into a dataframe where index=oncotree_code, columns=CSV names
    count_df = pd.DataFrame(all_counts).fillna(0).astype(int)

    # Plot stacked bar with oncotree_code on x-axis
    count_df.plot(kind="bar", stacked=True, figsize=(12, 6))

    plt.xlabel("Oncotree Code")
    plt.ylabel("Count")
    plt.title("Distribution of Oncotree Codes Across CSV Files")
    plt.legend(title="CSV Files", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_dir = os.path.join(split_path, "hest_data_distribution.png")
    plt.savefig(save_dir)
    print("Plot saved to ", save_dir)

def create_spatial_knn_edges(coords, k=8):
    """
    Spatial edges
    """
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)  # k+1 to include self

    # Remove self-loops (index 0)
    src_nodes = np.repeat(np.arange(coords.shape[0]), k)
    dst_nodes = indices[:, 1:].reshape(-1)

    edge_index = torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long)
    return edge_index

# def load_stpath_gene_dict():
#     with open("data/Hest_Bench/STPath_genes.json", "r") as f:
#         gene_dict = json.load(f)
#     return gene_dict


# def rename_genes(df_ori,stpath_gene_dict):
#     df_filtered = df_ori.loc[:, df_ori.columns.intersection(stpath_gene_dict.keys())]
#     # Rename columns to symbols
#     df_renamed = df_filtered.rename(columns=stpath_gene_dict)
#     return df_renamed

