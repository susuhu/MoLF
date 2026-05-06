import argparse
import wandb
from ruamel.yaml import YAML
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
import datetime

import torch

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
# In your training loop
from torch.amp import autocast, GradScaler

from models.flow_matching import GeneTransformerAutoencoder
from utils.custom_dataset import GeneExpressionDataset
from utils.training_utils import save_checkpoint, load_checkpoint
from utils.hest_utils import load_gene_list
from utils.other_utils import seed_everything


def calculate_reconstruction_loss(predicted_genes, target_genes, mask):
    """Calculates MSE loss on only the valid (non-padded) genes."""
    error = (predicted_genes - target_genes)**2
    masked_error = error * mask
    loss = masked_error.sum() / mask.sum().clamp(min=1e-9)
    return loss

def calculate_kl_loss(z_mean, z_log_var):
    """Calculates the KL divergence between the predicted latent distribution and a standard normal."""
    # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # We sum over all dimensions and average over the batch.
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=[1, 2])
    return torch.mean(kl_loss)


if __name__ == '__main__':
    print("[INFO_GENE_VAE]Starting training scirpt...")
     # Set seed for reproducibility
    seed_everything(seed=2025) 

    # for better yaml logging
    yaml = YAML()
    yaml.preserve_quotes = True  # enables quote preservation

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Mutually exclusive group for config_path and log_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config_path", "-f", type=str, help="Path to config file")
    group.add_argument("--log_dir", "-l", type=str, help="Logging directory")

    # Other args
    parser.add_argument("--continue_training", "-c", action="store_true",
                        help="Flag to continue training or not")
    
    args = parser.parse_args()
    continuous_training = args.continue_training
    log_dir = args.log_dir

    # If not continuing training, create a new log directory
    if not continuous_training:
        # Create a unique subfolder in training_log for each run
        run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("gene_vae", f"run_{run_time}")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "training_run.txt")
        # Redirect stdout and stderr to the log file
        sys.stdout = open(log_path, "w")
        sys.stderr = sys.stdout

        # load the training configuration
        with open(args.config_path, "r") as f:
            config = yaml.load(f)

        # Save config to log_dir for reproducibility
        with open(os.path.join(log_dir, "config.yml"), "w") as f:
            yaml.dump(config, f)

    # If continuing training, use the existing log directory
    else:
        assert os.path.exists(
            log_dir
        ), f"[ERROR_GENE_VAE] Log directory {log_dir} does not exist. Please check the path."
        # load the training configuration
        with open(os.path.join(log_dir, "config.yml"), "r") as f:
            config = yaml.load(f)
        log_path = os.path.join(log_dir, "training_run.txt")
        # Redirect stdout and stderr to the log file (append mode)
        sys.stdout = open(log_path, "a")
        sys.stderr = sys.stdout

    ckpt_dir = os.path.join(log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load gene list
    gene_list = load_gene_list(config)
    num_genes = len(gene_list)
    
    # --- NEW: Early Stopping and Checkpointing Parameters ---
    EARLY_STOP_PATIENCE = config["early_stop_patience"]
    NUM_EPOCHS=config["num_epochs"]
    print(f"[INFO_GENE_VAE]Using early stopping with patience: {EARLY_STOP_PATIENCE}")

    # Initialize wandb
    # wandb project name
    wandb_project_name = config["wandb_project_name"]
    wandb.login()
    wandb.init(
        project=wandb_project_name,  # Change to your project name
        name=f"run_{wandb.util.generate_id()}",
        config=config,
    )
    # Save the run id for later
    with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)

    wandb.config.update({"gene_dim": num_genes})  # Log gene_dim to wandb

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = GeneTransformerAutoencoder(n_genes=num_genes,
                                    model_dim=config["gene_autoencoder"]["model_dim"],
                                    latent_dim=config["gene_autoencoder"]["latent_dim"],
                                    n_heads=config["gene_autoencoder"]["n_heads"],
                                    n_layers=config["gene_autoencoder"]["n_layers"],
                                    dropout=config["dropout_rate"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr= config["lr"],weight_decay=config["weight_decay"])
    criterion = torch.nn.L1Loss()
    train_dataset = GeneExpressionDataset(config=config, 
                                          split="train",
                                          gene_list=gene_list)
    val_dataset = GeneExpressionDataset(config=config, 
                                          split="val",
                                          gene_list=gene_list)
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    print(f"[INFO_GENE_VAE] There are {num_train} training samples, and {num_val} val samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    #  CREATE THE LEARNING RATE SCHEDULER ---
    # First, calculate the total number of training steps.
    num_training_steps = NUM_EPOCHS * len(train_loader)

    # Choose a warmup period. A common choice is 5-10% of total steps.
    warmup_steps = int(0.05 * num_training_steps)

    print(f"[INFO_GENE_VAE] Total training steps: {num_training_steps}")
    print(f"[INFO_GENE_VAE] Warmup steps: {warmup_steps}")

    # Create the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    # This is essential for preventing underflow with half-precision gradients.
    scaler = GradScaler(device="cuda")

    # load the model state for continuing training
    if continuous_training:
        # Load the model state if continuing training
        checkpoint_path = os.path.join(log_dir, "ckpt", "latest_autoencoder.pth")
        if os.path.exists(checkpoint_path):
            print(f"[INFO_GENE_VAE] Loading model from {checkpoint_path}")
            start_epoch, best_val_loss, extra_info = load_checkpoint(
                path= checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            raise FileNotFoundError(
                f"[ERROR_GENE_VAE] Checkpoint file {checkpoint_path} does not exist. Cannot continue training."
            )
    else:
        print(f"[INFO_GENE_VAE] Starting training from scratch.")
        start_epoch = 0
        best_val_loss = float("inf")

    # --- NEW: Variables for Tracking Best Model and Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    print("Starting pre-training for the Autoencoder...")
    # --- The Main Training Loop ---
    for epoch in range(config["num_epochs"]):
        train_loss = 0.
        train_gene_loss = 0.
        train_kl_loss = 0.
        val_loss = 0.
        val_gene_loss = 0.
        val_kl_loss = 0.

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for i, data in enumerate(pbar):
            x, x_gt_mask, sample_id = data
            x = x.to(device)
            x_gt_mask = x_gt_mask.to(device)

            # check all zero samples
            patch_validity = x_gt_mask.sum(dim=-1) > 0  # Shape: (B, P)
            if not torch.any(patch_validity):
                print("\n\n!!!!!!!!!! TRAIN DEBUG TRIGGERED: BAD BATCH DETECTED !!!!!!!!!!")
                print(f"SAMPLE {sample_id} - This entire batch has a mask that will result in all patches being padded.")
                print(f"Shape of x: {x.shape}")
                print(f"Shape of gene_mask_patches: {x_gt_mask.shape}")
                
                # Print statistics of the input data that's causing the high KL loss
                print(f"Stats for x: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

            optimizer.zero_grad()

            # Optional random masking
            mask = torch.rand_like(x).to(device) > 0.15  # mask 15% of genes
            x_masked = x.clone().to(device)
            x_masked[~mask] = 0.0

            # --- Forward pass with autocast ---
            with autocast("cuda"):
                recon, z_mean, z_log_var = model(x_masked, x_gt_mask)
                gene_recon_loss = calculate_reconstruction_loss(x,recon,x_gt_mask)
                kl_loss = calculate_kl_loss(z_mean,z_log_var)
                total_loss = gene_recon_loss + config["gene_autoencoder"]["kl_loss_weight"] * kl_loss
                train_gene_loss = train_gene_loss + gene_recon_loss.item()
                train_kl_loss = train_kl_loss + kl_loss.item()

            # Debug NaN
            if torch.isnan(total_loss):
                print("NaN detected in loss")
                print("Pred min/max:", recon.min().item(), recon.max().item())
                print("Target min/max:", x.min().item(), x.max().item())
                print("Mask sum:", mask.sum().item())
                break

           # scaler.scale() multiplies the loss by a large factor to prevent
            # small float16 gradients from vanishing ("underflow").
            scaler.scale(total_loss).backward()

            # Optional: Gradient Clipping (unscale first)
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step() will automatically unscale the gradients
            # before the optimizer updates the weights.
            scaler.step(optimizer)
            # --- NEW: Update the scaler for the next iteration ---
            scaler.update()
            scheduler.step()

            train_loss +=total_loss.item()

            batch_log_data = {"batch/loss_train": total_loss.item()}
            # --- Update the tqdm progress bar
            pbar.set_postfix(batch_log_data)

        model.eval()
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} val")
        with torch.no_grad():
            for i, data in enumerate(pbar):
                x, x_gt_mask, sample_id = data
                x = x.to(device)
                x_gt_mask = x_gt_mask.to(device)

                # check all zero samples
                patch_validity = x_gt_mask.sum(dim=-1) > 0  # Shape: (B, P)
                if not torch.any(patch_validity):
                    print("\n\n!!!!!!!!!! EVAL DEBUG TRIGGERED: BAD BATCH DETECTED !!!!!!!!!!")
                    print(f"SAMPLE {sample_id} - This entire batch has a mask that will result in all patches being padded.")
                    print(f"Shape of x: {x.shape}")
                    print(f"Shape of gene_mask_patches: {x_gt_mask.shape}")
                    
                    # Print statistics of the input data that's causing the high KL loss
                    print(f"Stats for x: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")
                      
                # --- Forward pass with autocast ---
                with autocast("cuda"):
                    recon, z_mean, z_log_var = model(x, x_gt_mask)
                    gene_recon_loss = calculate_reconstruction_loss(x,recon,x_gt_mask)
                    kl_loss = calculate_kl_loss(z_mean,z_log_var)
                    total_loss_nograd = gene_recon_loss + config["gene_autoencoder"]["kl_loss_weight"] * kl_loss
                    val_gene_loss = val_gene_loss + gene_recon_loss.item()
                    val_kl_loss = val_kl_loss + kl_loss.item()

                val_loss += total_loss_nograd.item()

                # --- Local Logging & Accumulation ---
                log_data = {"total_loss_val": total_loss_nograd.item()}
                pbar.set_postfix(log_data)
                
        average_train_loss = train_loss/num_train
        average_train_gene_loss = train_gene_loss/num_train
        average_train_kl_loss = train_kl_loss/num_train
        average_val_loss = val_loss/num_val
        average_val_gene_loss = val_gene_loss/num_val
        average_val_kl_loss = val_kl_loss/num_val
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {average_train_loss:.6f}, Val Loss: {average_val_loss:.6f}")
        checkpoint_path = os.path.join(ckpt_dir, "latest_autoencoder.pth")
        save_checkpoint(model=model,optimizer=optimizer, scheduler=None,epoch=epoch, best_val_loss=val_loss,path=checkpoint_path)

        wandb.log(
            {
                "train/epoch_loss": average_train_loss,
                "train/gene_loss":average_train_gene_loss,
                "train/kl_loss": average_train_kl_loss,
                "val/epoch_loss": average_val_loss,
                "val/gene_loss": average_val_gene_loss,
                "val/kl_loss": average_val_kl_loss,
                "epoch": epoch,
            }
        )

        # --- NEW: Checkpointing and Early Stopping Logic ---
        # Check if the current validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model's state dictionary to the specified path.
            checkpoint_path = os.path.join(ckpt_dir, "best_autoencoder.pth")
            save_checkpoint(model=model,optimizer=optimizer,scheduler=None,epoch=epoch, best_val_loss=best_val_loss,path=checkpoint_path)

            print(f"  -> New best model saved to {checkpoint_path} with validation loss: {best_val_loss/num_val:.6f}")
            # Reset the patience counter because we found a better model.
            epochs_no_improve = 0
            wandb.log(
                {
                    "best_epoch": epoch,
                }
            )

        else:
            # If the validation loss did not improve, increment the patience counter.
            epochs_no_improve += 1
            print(f"  -> Validation loss did not improve. Patience: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        # Check if we've exceeded our patience. If so, stop training.
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            print(f"The best model was saved at the epoch with validation loss: {best_val_loss/num_val:.6f}")
            break # Exit the training loop

    print("\nPre-training finished.")
    print(f"The best model is saved at '{log_dir}' with validation loss: {best_val_loss/num_val:.6f}")