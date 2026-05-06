import os
import random
import numpy as np
import torch
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm

import torch
from torch.amp import autocast

from utils.pfm_preprocesing import (
    load_uni_v2,
    load_uni_v1,
    load_hoptimus0,
    load_virchow2,
    load_gigapath,
)
from utils.hest_utils import load_gene_list
from models.flow_matching import LatentFlow,LatentFlow_CFG
from models.baseline_models import SimpleMLP 


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path, **kwargs):
    """
    Save training checkpoint.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to save (optional, can be None).
        epoch (int): Current training epoch.
        path (str): File path to save checkpoint.
        **kwargs: Any extra info you want to save (dict will be added to checkpoint).
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,  # Placeholder for best validation loss
    }
    checkpoint.update(kwargs)  # Add extra custom info if provided

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved to {path}")


# to add best_val_loss to the checkpoint
def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """
    Load training checkpoint.

    Args:
        path (str): Path to checkpoint file.
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to restore.
        map_location (str or torch.device): Where to map the loaded tensors.

    Returns:
        int: The epoch to resume from.
        dict: Any extra info saved in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if (
        scheduler is not None
        and "scheduler_state" in checkpoint
        and checkpoint["scheduler_state"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    extra_info = {
        k: v
        for k, v in checkpoint.items()
        if k not in ["epoch", "model_state", "optimizer_state", "scheduler_state"]
    }

    print(f"[INFO] Checkpoint loaded from {path}, resuming at epoch {start_epoch}")
    try:
        best_val_loss = checkpoint["best_val_loss"]
        return start_epoch, best_val_loss, extra_info
    except:
        return start_epoch, None, extra_info


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 50)
    print(f"[INFO] Total parameters   : {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print("=" * 50)


def get_emb_dim(config):
    if config["PFM_name"] in ["univ2", "hoptimus0", "gigapath"]:
        emb_dim = 1536
    elif config["PFM_name"] == "uni":
        emb_dim = 1024
    elif config["PFM_name"] == "virchow2":
        emb_dim = 1280
    else:
        raise ValueError(f"[FLOW_ST_ERROR] Unknown PFM name: {config['PFM_name']}")
    return emb_dim


def get_PFM_model(config):
    if config["PFM_name"].lower() in ["univ2"]:
        # Load the pretrained pathology foundation model and transform
        PFM_model, transform = load_uni_v2(FIRST_TIME=False)
    elif config["PFM_name"].lower() == "uni":
        PFM_model, transform = load_uni_v1(FIRST_TIME=False)
    elif config["PFM_name"].lower() == "virchow2":
        PFM_model, transform = load_virchow2()
    elif config["PFM_name"].lower() == "hoptimus0":
        PFM_model, transform = load_hoptimus0()
    elif config["PFM_name"].lower() == "gigapath":
        PFM_model, transform = load_gigapath()
    else:
        raise ValueError(f"[FLOW_ST_ERROR] Unknown PFM name: {config['PFM_name']}")
    return PFM_model, transform


def get_model(config, gene_dim, emb_dim, device, flow_path=None):
    # velocity field model init
    model_name = config["model"]
    print(f"[INFO_MDOEL] Initializing model: {model_name}")
    # if model_name.lower() == "latentflow":
    #     model = LatentFlow(
    #         config=config,
    #         gene_dim=gene_dim,
    #         time_dim=1,
    #         img_emb_dim=emb_dim,
    #         flow_path=flow_path,
    #     ).to(device)
    
    if model_name.lower() == "molf":
        model = LatentFlow_CFG(
            config=config,
            gene_dim=gene_dim,
            time_dim=1,
            img_emb_dim=emb_dim,
            flow_path=flow_path,
        ).to(device)
    elif model_name.lower() == "simplemlp":
        model = SimpleMLP(
            embedding_dim=emb_dim,
            hidden_features_1=config["hidden_dim"][0],
            hidden_features_2=config["hidden_dim"][1],
            output_features=gene_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    return model


def handle_tensor_batch(model, batch, model_name: str):
    """
    Batch handler for a non-graph model that uses padded tensors.
    """
    # Get the target device from the model itself.
    device = next(model.parameters()).device

    # 1. Unpack the batch tuple. This order MUST match the return order of your pad_collate_fn.
    #  embeddings, coords, gene_tensor, gene_mask_tensor, oncotree_onehoted, sample_id
    embeddings, _, genes, gene_mask_tensor, onco_onehot, sample_id = batch

    # 2. Move each TENSOR to the GPU individually.
    #    Note: sample_ids is a list of strings, it stays on the CPU.
    embeddings = embeddings.to(device)
    genes = genes.to(device)
    gene_mask_tensor = gene_mask_tensor.to(device)
    onco_onehot = onco_onehot.to(device)

    patch_validity = gene_mask_tensor.sum(dim=-1) > 0  # Shape: (B, P)
    if not torch.any(patch_validity):
        print("\n\n!!!!!!!!!! DEBUG TRIGGERED: BAD BATCH DETECTED !!!!!!!!!!")
        print(f"SAMPLE {sample_id} - This entire batch has a mask that will result in all patches being padded.")
        print(f"Shape of x_1: {genes.shape}")
        print(f"Shape of gene_mask_patches: {gene_mask_tensor.shape}")
        
        # Print statistics of the input data that's causing the high KL loss
        print(f"Stats for x_1: mean={genes.mean():.4f}, std={genes.std():.4f}, min={genes.min():.4f}, max={genes.max():.4f}")
        
        # Print the validity mask to confirm it's all False
        print("Patch validity (should be all False):")
        print(patch_validity)

        # It's useful to see the sum of the mask per patch
        print("Sum of mask per patch (should be all 0):")
        print(gene_mask_tensor.sum(dim=-1))

    # forward(self, x_1, emb,  gene_mask_patches, onco_onehot):
    if model_name == "molf":
        total_loss, loss_dict, moe_load_balancing_logits = model(
            x_1=genes,  # The padded input features
            emb=embeddings,
            gene_mask_patches=gene_mask_tensor,  # The padded ground truth targets
            onco_onehot=onco_onehot
        )
    elif model_name == "simplemlp":
        total_loss, loss_dict,= model(
            x=embeddings,
            y=genes,
        )
        moe_load_balancing_logits = None
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return total_loss, loss_dict, moe_load_balancing_logits, sample_id



def unified_train_epoch(
    model, dataloader, optimizer, scheduler, scaler, batch_handler_fn, epoch,  bad_batch_dir, model_name, save_bad_batches=True, 
):
    """
    This function now returns a dictionary containing the average loss values for the epoch.
    """
    model.train()

    # Initialize trackers for accumulating losses for this epoch
    epoch_total_loss = 0.0
    # Use a dictionary to flexibly track all individual loss components
    epoch_loss_components = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad(
            set_to_none=True
        )  # Use set_to_none=True for a small speedup

        try:
            # This tells PyTorch to automatically cast tensors to float16
            # for compatible operations like matrix multiplies.
            # with autocast(device_type="cuda"):
            # run in fp32 for stability
            with autocast(device_type="cuda", enabled=False):
                total_loss, loss_dict_for_logging, moe_load_balancing_logits, sample_id = batch_handler_fn(model, batch, model_name)

            # Check for NaN/Inf loss before backward
            if not torch.isfinite(total_loss):
                print(f"[WARNING NaN DETECTED] Epoch={epoch}, Batch={i}, Sample_id={sample_id} Loss={total_loss.item()}")
                if save_bad_batches:
                    torch.save(batch, os.path.join(bad_batch_dir, f"epoch{epoch}_iter{i}.pt"))
                continue  # Skip this batch


            # scaler.scale() multiplies the loss by a large factor to prevent
            # small float16 gradients from vanishing ("underflow").
            scaler.scale(total_loss).backward()

            # Optional: Gradient Clipping (unscale first)
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Drop any bad grads
            bad_grad_found = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad_found = True
                    p.grad = None  # Drop this param's grad
            if bad_grad_found:
                print(f"[WRNING BAD GRADIENT] Epoch={epoch}, Batch={i} Sample id={sample_id}→ grads reset for stability")

            # scaler.step() will automatically unscale the gradients
            # before the optimizer updates the weights.
            scaler.step(optimizer)
            # --- NEW: Update the scaler for the next iteration ---
            scaler.update()
            scheduler.step()

        except RuntimeError as e:
            print(f"[WARNING BACKWARD ERROR] Epoch={epoch}, Sample_id={sample_id}, Batch={i} → {e}")
            if save_bad_batches:
                torch.save(batch, os.path.join(bad_batch_dir, f"epoch{epoch}_iter{i}_error.pt"))
            continue  # Skip this batch

        # --- Batch-level Logging ---
        # --- Prepare data for this batch
        batch_log_data = {
            "batch/total_loss": total_loss.item(),
            **loss_dict_for_logging,
        }

        # --- Update the tqdm progress bar
        pbar.set_postfix(**batch_log_data)

        # --- Accumulate for Epoch-level Summary ---
        epoch_total_loss += total_loss.item()
        for key, value in loss_dict_for_logging.items():
            # If the key is new, add it. Otherwise, add to the existing value.
            epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value

    # --- Calculate Averages for the Epoch ---
    num_batches = len(dataloader)
    avg_epoch_total_loss = epoch_total_loss / num_batches
    # avg_epoch_components = {f"epoch/{key}": value / num_batches for key, value in epoch_loss_components.items()}
    avg_epoch_components = {
        key: value / num_batches for key, value in epoch_loss_components.items()
    }

    # Combine all epoch-level averages into one dictionary
    epoch_summary = {"train/total_loss": avg_epoch_total_loss}
    for key, value in avg_epoch_components.items():
        epoch_summary[f"train/{key}"] = value
    return epoch_summary


def evaluate_epoch(model, dataloader, batch_handler_fn, epoch, model_name, description="val"):
    """
    Runs one epoch of evaluation (validation or testing).
    This function is generic and can be used for both.

    Returns:
        A dictionary containing the average loss values for the epoch.
    """
    # --- Set the model to evaluation mode ---
    model.eval()

    # Initialize trackers for accumulating losses
    epoch_total_loss = 0.0
    epoch_loss_components = {}

   # We get num_experts and the device directly from the model for robustness.
    # This assumes your LatentFlow model has these attributes.
    if getattr(model, 'use_moe', False):
        num_experts = model.num_experts
        device = next(model.parameters()).device
        total_expert_counts = torch.zeros(num_experts, device=device)
    else:
        total_expert_counts = None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} {description}")

    # --- Disable gradient calculation for efficiency ---
    with torch.no_grad():
        for batch in pbar:
            # The batch handler works exactly the same way
            # with autocast(device_type="cuda"):
            total_loss_for_eval, loss_dict_for_logging, moe_load_balancing_logits, _ = batch_handler_fn(model, batch,model_name)

            # --- Local Logging & Accumulation ---
            log_data = {"total_loss": total_loss_for_eval.item(), **loss_dict_for_logging}
            pbar.set_postfix(**log_data)

            epoch_total_loss += total_loss_for_eval
            for key, value in loss_dict_for_logging.items():
                epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value

            # <<< NEW: Calculate and accumulate expert utilization >>>
            if total_expert_counts is not None and moe_load_balancing_logits is not None:
                # Get the indices of the top-k chosen experts
                k = model.velocity_predictor.gating.k
                _, top_indices = torch.topk(moe_load_balancing_logits, k=k, dim=-1)
                
                # Flatten the indices to count them easily
                top_indices_flat = top_indices.flatten()

                # Use bincount to efficiently count and add to the total
                batch_expert_counts = torch.bincount(top_indices_flat, minlength=num_experts)
                total_expert_counts += batch_expert_counts

    # --- Calculate and Return Averages ---
    num_batches = len(dataloader)
    avg_epoch_total_loss = epoch_total_loss / num_batches
    avg_epoch_components = {
        key: value / num_batches for key, value in epoch_loss_components.items()
    }

    # Use the 'description' to create prefixed keys (e.g., "val/total_loss")
    prefix = description.lower()
    epoch_summary = {f"{prefix}/total_loss": avg_epoch_total_loss}
    for key, value in avg_epoch_components.items():
        epoch_summary[f"{prefix}/{key}"] = value

    # <<< NEW: Process and add utilization stats to the summary >>>
    if total_expert_counts is not None:
        total_tokens_routed = total_expert_counts.sum()
        if total_tokens_routed > 0:
            expert_utilization = total_expert_counts / total_tokens_routed
            # Create a clean dictionary for logging utilization to wandb
            utilization_dict = {
                f"{prefix}/expert_{i}_utilization_pct": util.item() * 100
                for i, util in enumerate(expert_utilization)
            }
            epoch_summary.update(utilization_dict)

    return epoch_summary

