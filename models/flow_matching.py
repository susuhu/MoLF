import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import math

import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from flow_matching.solver import ODESolver

    
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding. Adds positional information to the input embeddings.
    """
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is transposed from batch-first to seq-first for PE addition
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)].to(x.device)
        x = self.dropout(x)
        return x.permute(1, 0, 2) # Back to batch-first


class GeneTransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder for patch-level gene panels.
    Encoder: (B, P, G) -> (B, P, latent_dim)
    Decoder: (B, P, latent_dim) -> (B, P, G)
    """

    def __init__(self, n_genes: int, model_dim: int, latent_dim: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads=n_heads
        self.n_genes = n_genes
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout=dropout
        

        # --- ENCODER ---
        # Project each patch's gene panel into model_dim
        self.gene_input_proj = nn.Linear(self.n_genes, self.model_dim)
        self.input_norm = LayerNorm(self.model_dim)

        # Positional encoding across patches (WSI)
        self.pos_encoder = PositionalEncoding(max_len=21000, d_model=self.model_dim, dropout=self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=self.n_heads, batch_first=True, dropout=self.dropout, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Latent projection
        self.latent_head = nn.Linear(self.model_dim, 2 * self.latent_dim)
        # layernorm will only be applied to the mean latent
        self.latent_norm = LayerNorm(self.latent_dim)

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.model_dim),
            LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim * 2),
            LayerNorm(self.model_dim * 2),
            nn.GELU(),
            nn.Linear(self.model_dim * 2, n_genes), # Project back to the full gene panel
            nn.GELU()
        )

    def encode(self, gene_expr: torch.Tensor, mask: torch.Tensor ) -> torch.Tensor:
        """
        Encode patch-level gene panels.
        gene_expr: (B, P, G)
        Returns: (B, P, latent_dim)
        """
        B, P, G = gene_expr.shape

        # Apply mask: zero-out missing genes
        if mask is not None:
            gene_expr = gene_expr * mask.float()

        # Project each patch
        patch_emb = self.input_norm(self.gene_input_proj(gene_expr))  # (B, P, D)
        # Add positional encoding for patches
        patch_emb = self.pos_encoder(patch_emb)      # (B, P, D)

        # Optional: mask transformer input
        if mask is not None:
            # A patch is valid if it has any real gene
            patch_valid = (mask.sum(dim=-1) > 0)  # (B, P) bool
            padding_mask = ~patch_valid           # True = pad (PyTorch convention)
        else:
            padding_mask = None

        # Transformer over patches
        enc_out = self.transformer_encoder(patch_emb, src_key_padding_mask=padding_mask)  # (B, P, D)


        #    The shape will be (B, P, 2 * latent_dim)
        full_latent_output = self.latent_head(enc_out)

        #  Split the tensor into two halves along the last dimension.
        #    torch.chunk is a clean way to do this.
        z_mean, z_log_var = torch.chunk(full_latent_output, 2, dim=-1)

        #  Apply normalization ONLY to the mean component.
        #    It's standard practice not to normalize the log-variance.
        z_mean = self.latent_norm(z_mean)  # (B, P, latent_dim)

        return z_mean, z_log_var

    def decode(self, latent_z: torch.Tensor,mask: torch.Tensor ) -> torch.Tensor:
        """
        Decode latent patch embeddings back to gene panels.
        latent_z: (B, P, latent_dim)
        Returns: (B, P, G)
        """
        # B, P, D = latent_z.shape

        # latent_for_decoder = self.memory_norm(self.latent_to_model(latent_z))  # (B, P, model_dim)

        # # Expand learnable query per patch
        # queries = self.patch_queries.expand(B, P, -1)  # (B, P, D)

        # # Decoder attends to latent patch embeddings
        # dec_out = self.transformer_decoder(tgt=queries, memory=latent_for_decoder)  # (B, P, D)

        # # Project to gene panel
        # recon = self.out_norm(self.output_proj(dec_out))  # (B, P, G)
        
        # # Optional: zero-out missing genes in output
        # if mask is not None:
        #     recon = recon * mask.float()

        # return recon
                # Optional: zero-out missing genes in output
        # MLP decoder
        recon = self.decoder(latent_z)
        if mask is not None:
            recon = recon * mask.float()
        return recon

    def forward(self, gene_expr: torch.Tensor,mask: torch.Tensor):
        z_mean, z_log_var = self.encode(gene_expr,mask)

        std = torch.exp(0.5 * z_log_var)  # standard deviation
        epsilon = torch.randn_like(std)  # sample from N(0, I)
        latent_z = z_mean + std * epsilon       # scale and shift the noise

        if self.training:
            latent_z = latent_z + torch.randn_like(latent_z) * 0.1 # Add a small amount of noise

        recon = self.decode(latent_z,mask)
        return recon, z_mean, z_log_var



class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=4, dropout_rate=0.1):
        super().__init__()
        # PyTorch's built-in MultiheadAttention is perfect for this.
        # It handles all the complexity of Q, K, V projections and attention calculation.
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, # The dimension of the final output
            kdim=context_dim,    # The dimension of the keys (from context)
            vdim=context_dim,    # The dimension of the values (from context)
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True     # This is crucial for our data shapes
        )
        self.norm = LayerNorm(query_dim)
        
    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): The tensor that is "asking" the questions.
                                  Shape: [batch_size, num_queries, query_dim]
            context (torch.Tensor): The tensor providing the information.
                                    Shape: [batch_size, num_context_items, context_dim]
        
        Returns:
            torch.Tensor: The attended query tensor.
                          Shape: [batch_size, num_queries, query_dim]
        """
        # The MHA layer returns the attention output and the attention weights.
        # We only need the output.
        attn_output, _ = self.attention(
            query=query,
            key=context,
            value=context
        )
        # Add & Norm is a standard pattern in transformers
        return self.norm(query + attn_output)


class TransformerExpert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_heads, n_layers, dropout):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            activation = F.gelu
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (B, input_dim) → we add seq dimension for Transformer
        """
        x = self.input_norm(self.input_proj(x)).unsqueeze(1)  # (B, 1, hidden_dim)
        x = self.transformer(x)              # (B, 1, hidden_dim)
        x = self.output_proj(x.squeeze(1))   # (B, output_dim)
        return x


class TopKGate(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noise_epsilon = 1e-2
        self.linear = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: (B, input_dim)
        returns:
          topk_indices: (B, k)
          topk_scores: (B, k)  (normalized over k)
          all_logits: (B, num_experts) <-- NEW: For the loss
        """
        logits = self.linear(x)  # (B, num_experts)

        # <<< --- ADD NOISE ONLY DURING TRAINING --- >>>
        if self.training:
            # Add Gumbel noise for better exploration, but standard normal is fine too
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits += noise

        topk_scores_raw, topk_indices = torch.topk(logits, self.k, dim=-1)

        # Normalize scores across selected experts for the forward pass
        topk_scores = torch.softmax(topk_scores_raw, dim=-1)  # (B, k)
        
        # We return the raw logits for the loss calculation
        return topk_indices, topk_scores, logits


class MoEVelocityPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_experts,
                 n_heads, n_layers, dropout, k=2):
        super().__init__()
        self.num_experts = num_experts # Store this for later
        self.output_dim = output_dim   # Store this for later
        self.k = k
        self.experts = nn.ModuleList([
            TransformerExpert(input_dim, output_dim, hidden_dim,
                              n_heads, n_layers, dropout)
            for _ in range(num_experts)
        ])
        self.gating = TopKGate(input_dim, num_experts, k=self.k)

    def forward(self, x):
        N, _ = x.shape # n is B*P and B is always 1
        topk_indices, topk_scores, gating_logits = self.gating(x)  # (B,k),(B,k),(B,num_experts)

        # Initialize the final output tensor that will accumulate results.
        final_outputs = torch.zeros(N, self.output_dim, device=x.device, dtype=x.dtype)

        # --- Vectorized Dispatch Logic ---

        # Flatten the expert indices and scores to create a long list of routing tasks
        # Shape becomes (N * k)
        flat_topk_indices = topk_indices.flatten()

        # Create a mapping from the flattened list back to the original patch index (0 to N-1)
        # E.g., for N=3, k=2, this becomes [0, 0, 1, 1, 2, 2]
        # This tells us which original patch each routing task belongs to.
        batch_indices = torch.arange(N, device=x.device).repeat_interleave(self.k)

        # Instead of looping over patches, we loop over our small number of experts.
        # This loop is very fast (e.g., runs 8 times, not 10,000).
        for i in range(self.num_experts):
            # 1. GATHER: Find all patches that are routed to the current expert 'i'.
            # Create a boolean mask to identify these tasks.
            mask = (flat_topk_indices == i)

            # If no patches are sent to this expert, we can skip it.
            if not mask.any():
                continue

            # Use the mask to get the original indices of the patches for this expert.
            original_indices = batch_indices[mask]

            # 2. BATCH FORWARD PASS: Process all selected patches in a single batch.
            # This is the key to performance!
            inputs_for_expert = x[original_indices]
            expert_output = self.experts[i](inputs_for_expert)

            # 3. SCATTER: Weight the results and add them back to the correct positions.
            # Get the gating scores corresponding to these specific patches.
            scores_for_expert = topk_scores.flatten()[mask].unsqueeze(1)
            weighted_output = expert_output * scores_for_expert

            # Use `index_add_` to efficiently add the weighted outputs to the final tensor
            # at their original locations. This is a highly optimized "scatter-add" operation.
            final_outputs.index_add_(0, original_indices, weighted_output)

        return final_outputs, gating_logits


class AttentionVelocityHead(nn.Module):
    """
    A velocity predictor that uses a single Transformer Encoder layer for global context,
    without positional encoding, making it permutation-invariant.
    """
    def __init__(self, input_dim: int, output_dim: int, model_dim: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. An input projection layer to get to the model's dimension
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.input_norm = LayerNorm(model_dim)
        # 2. A single, powerful Transformer Encoder Layer
        # This is the core of the self-attention mechanism.
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_head, 
            batch_first=True,  # Crucial for our [batch, seq, feature] format
            dropout=dropout,
            dim_feedforward=model_dim * 4, # Standard practice
            activation=F.gelu
        )
        
        # 3. A final linear layer to project back to the latent velocity dimension
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): The concatenated input tensor [z_t, t, conditioned_emb].
                              Shape: [n_patches, input_dim]
        """
        # batch size is 1
        # Project to the Transformer's model dimension
        x = self.input_norm(self.input_proj(x)) # Shape: [1, n_patches, model_dim]
        
        # Pass through the self-attention layer. 
        # No positional encoding is added.
        # No mask is needed as we're not padding within a WSI.
        contextual_x = self.attn_layer(x) # Shape: [1, n_patches, model_dim]
        
        # Project back to the latent velocity dimension
        velocity = self.output_proj(contextual_x) # Shape: [1, n_patches, latent_dim]
        
        # Remove the temporary batch dimension
        return velocity
    

class LatentFlow(nn.Module):
    def __init__(
        self,
        config,
        gene_dim, 
        time_dim,
        img_emb_dim,
        flow_path,
    ):
        super().__init__()

        self.img_emb_dim = img_emb_dim
        self.gene_dim = gene_dim
        self.hidden_dim = config["hidden_dim"] # for velocity head hidden dim
        self.flow_loss_weight = config["loss_config"]["flow_loss_weight"]
        self.gene_loss_weight = config["loss_config"]["gene_loss_weight"]
        self.kl_loss_weight = config["loss_config"]["kl_loss_weight"]
        self.load_balancing_loss_weight = config["loss_config"]["load_balancing_loss_weight"]
        self.flow_path = flow_path
        self.dropout_rate =config["dropout_rate"]
        self.use_spatial_encoding =  config["spatial_encoding"]["use_spatial_encoding"]
        self.gene_vae_pretrained_path = config["gene_vae_pretrained_path"]

        if self.use_spatial_encoding:
            print("[INFO MODEL] Using spatial encoding for image embeddings.")
            ### NEW: A dedicated Transformer to process spatial patch embeddings
            spatial_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.img_emb_dim,
                nhead=config["spatial_encoding"]["spatial_transformer_heads"],
                batch_first=True,
                dropout=self.dropout_rate,
                activation=F.gelu
            )
            self.spatial_transformer = nn.TransformerEncoder(
                spatial_encoder_layer, 
                num_layers=config["spatial_encoding"]["spatial_transformer_layers"]
            )

            ### NEW: The 1D positional encoding module
            self.spatial_pos_encoder = PositionalEncoding(
                max_len=21000,
                d_model=self.img_emb_dim,
                dropout=self.dropout_rate,
            )


        # Instantiate the autoencoder as a single module
        if self.gene_vae_pretrained_path and os.path.exists(self.gene_vae_pretrained_path):
            self.gene_vae_config_path = os.path.join(self.gene_vae_pretrained_path,"config.yml")
            with open(self.gene_vae_config_path,"r") as f:
                self.gene_vae_config = yaml.safe_load(f)
            
            self.gene_latent_dim = self.gene_vae_config ["gene_autoencoder"]["latent_dim"]
            self.gene_autoencoder = GeneTransformerAutoencoder(n_genes=gene_dim,
                                            model_dim=self.gene_vae_config ["gene_autoencoder"]["model_dim"],
                                            latent_dim=self.gene_latent_dim,
                                            n_heads=self.gene_vae_config ["gene_autoencoder"]["n_heads"],
                                            n_layers=self.gene_vae_config ["gene_autoencoder"]["n_layers"],
                                            dropout=self.gene_vae_config ["dropout_rate"])

            self.gene_vae_pretrained_weights_path = os.path.join(self.gene_vae_pretrained_path,"ckpt", "best_autoencoder.pth")
            self.vae_checkpoint = torch.load(self.gene_vae_pretrained_weights_path, map_location="cpu")
            self.gene_autoencoder.load_state_dict(self.vae_checkpoint["model_state"])
            print(f"[INFO MODEL] Loading pre-trained Gene VAE from: {self.gene_vae_pretrained_weights_path} at epoch {self.vae_checkpoint['epoch']}")
            
            for param in self.gene_autoencoder.parameters():
                param.requires_grad = False
            print(f"[INFO MODEL] Freezing Gene VAE weights for Stage 2 training.")
            
        else:
            print("[INFO MODEL] Initializing new Gene VAE for end-to-end training.")
            self.gene_latent_dim = config['gene_autoencoder']['latent_dim']
            self.gene_autoencoder = GeneTransformerAutoencoder(
                n_genes=gene_dim,
                model_dim=config['gene_autoencoder']["model_dim"],
                latent_dim=self.gene_latent_dim,
                n_heads=config['gene_autoencoder']['n_heads'],
                n_layers=config['gene_autoencoder']['n_layers']
            )

        self.onco_onehot_dim = config.get("onco_onehot_dim", False)
        if self.onco_onehot_dim:
            self.onco_embedder = nn.Linear(self.onco_onehot_dim, self.img_emb_dim)
            self.cross_attn_heads = config["cross_attn_heads"]
            self.conditioner = CrossAttentionLayer(
                query_dim=self.img_emb_dim, # The image embedding is the query
                context_dim=self.img_emb_dim, # The onco embedding is the context
                dropout_rate=self.dropout_rate,
                n_heads=self.cross_attn_heads)
        else:
            self.conditioner = None


        time_dim = 1 # We pass time as a single scalar feature
        velocity_input_dim = self.gene_latent_dim + time_dim + self.img_emb_dim
        self.velocity_input_norm = LayerNorm(velocity_input_dim)

        self.use_moe = config["mixture_of_expert"]["use_moe"]
        if self.use_moe:
            self.num_experts = config["mixture_of_expert"]["num_experts"] # Store for loss calculation
            self.velocity_predictor = MoEVelocityPredictor(
                input_dim=velocity_input_dim,
                output_dim=self.gene_latent_dim,
                hidden_dim=config["mixture_of_expert"]["expert_dim"],
                num_experts=self.num_experts,
                n_heads=config["mixture_of_expert"]["expert_heads"],
                n_layers=config["mixture_of_expert"]["expert_layers"],
                dropout=config["mixture_of_expert"]["moe_dropout"],
                k=config["mixture_of_expert"]["top_k"],
            )
        else:  
            self.velocity_predictor = AttentionVelocityHead(
                input_dim=velocity_input_dim,
                output_dim=self.gene_latent_dim,
                model_dim=config['velocity_head']['model_dim'],     # Add this section to your config
                n_head=config['velocity_head']['n_head'],
                dropout=self.dropout_rate
            )

    # We can now define the encode/decode methods for the main model
    def gene_encode(self, gene_expr, mask):
        # We need to handle the pathway_id here if it's conditional.
        # For a joint gene list, the mask is sufficient.
        # If mask is (B, P, G), reduce to (B, P) indicating whether a patch has any real gene
        return self.gene_autoencoder.encode(gene_expr, mask)

    def gene_decode(self, latent_z, mask):
        return self.gene_autoencoder.decode(latent_z,mask)

    def _calculate_flow_loss(self, predicted_latent_velocity, target_latent_velocity):
        # We don't need NaN masking because latents should not contain NaNs.
        return F.mse_loss(predicted_latent_velocity, target_latent_velocity)

    ### CHANGE: The loss function now MUST use the mask, not isnan()
    def _calculate_reconstruction_loss(self, predicted_genes, target_genes, mask):
        """Calculates MSE loss on only the valid (non-padded) genes."""
        # Calculate element-wise squared error
        error = (predicted_genes - target_genes)**2
        
        # Apply the mask (mask should have 1s for real genes, 0s for pad)
        masked_error = error * mask
        
        # Calculate the mean loss only over the real elements
        loss = masked_error.sum() / mask.sum().clamp(min=1e-9)
        return loss

    def _calculate_load_balancing_loss(self, gating_logits: torch.Tensor):
        """
        Calculates the load balancing loss for the MoE.
        This loss encourages the gating network to distribute tokens evenly.
        
        Args:
            gating_logits (torch.Tensor): Raw logits from the gating network
                                          of shape (B, num_experts).
        
        Returns:
            torch.Tensor: A scalar loss value.
        """
        if gating_logits is None:
            return 0.0

        # gating_logits shape: (B * P, num_experts)
        # P is number of patches, B is batch size
        
        # Calculate the router probabilities over all experts
        router_probs = F.softmax(gating_logits, dim=-1)

        # Calculate f_i: Fraction of router probability allocated to expert i
        # This measures the "confidence" in each expert.
        # We take the mean over the batch dimension.
        f_i = router_probs.mean(dim=0) # Shape: (num_experts,)

        # Calculate P_i: Fraction of tokens dispatched to expert i
        # This measures the actual routing decisions.
        # First, get the chosen expert indices from the logits
        _, top_indices = torch.topk(gating_logits, k=self.velocity_predictor.gating.k, dim=-1)
        # Convert to a one-hot representation of which experts were chosen
        one_hot_choices = F.one_hot(top_indices, num_classes=self.num_experts).float()
        # Sum over the k choices for each token
        one_hot_choices = one_hot_choices.sum(dim=1) # Shape: (B*P, num_experts)
        # Now, calculate the mean over the batch
        P_i = one_hot_choices.mean(dim=0) # Shape: (num_experts,)

        # The loss is the scaled dot product of these two distributions
        # This encourages both distributions to be uniform
        load_balancing_loss = self.num_experts * (f_i * P_i).sum()
        
        return load_balancing_loss
    
    
    def _get_predictions(self, z_t, t, emb, onco_onehot):
        
        # spatial encoding of patches
        if self.use_spatial_encoding:
            emb_with_pos = self.spatial_pos_encoder(emb)
            context_aware_emb_batch = self.spatial_transformer(emb_with_pos)
        else:
            context_aware_emb_batch = emb

        # Conditioning for flobal information
        if self.conditioner is not None:
            onco_embedding = self.onco_embedder(onco_onehot)
            query = context_aware_emb_batch
            conditioned_emb = self.conditioner(query=query, context=onco_embedding) # key and value are both context
        else:
            # If no conditioning, just use the spatial embeddings
            conditioned_emb = context_aware_emb_batch

        B, P, _ = z_t.shape
        # Make t compatible: expand to (B, P, 1)
        t_expanded = t.view(B, 1, 1).expand(B, P, 1)
        h = self.velocity_input_norm(torch.cat([z_t, t_expanded, conditioned_emb], dim=-1))  # (n_batch*n_patches,combined_input_dim)
        # Predict velocity
        moe_gating_logits = None
        if self.use_moe:
            h_flat = h.view(B * P, -1)  # (B*P, combined_input_dim)
            predicted_latent_velocity, moe_gating_logits = self.velocity_predictor(h_flat)
            predicted_latent_velocity = predicted_latent_velocity.view(B, P, -1)  # (B, P, gene_latent_dim)
        else:
            predicted_latent_velocity = self.velocity_predictor(h)

        return predicted_latent_velocity, moe_gating_logits


    def get_velocity(self, t, x, emb, onco_onehot):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        predicted_latent_velocity, _  = self._get_predictions(
                                    z_t=x,
                                    t=t,
                                    emb=emb,
                                    onco_onehot=onco_onehot,
                                )
        # Directly return the velocity tensor
        return predicted_latent_velocity

    def sample(
        self,
        emb,
        onco_onehot,
        mask,
        solver_config: dict,
        return_intermediates=False,
        x_init_override=None,
    ):
        """
        Generates gene expression samples using the ODE solver.

        Args:
            emb (torch.Tensor): The image embeddings [num_nodes, emb_dim].
            onco_onehot (torch.Tensor): The per-graph conditioning [num_graphs, onco_dim].
            batch_vec (torch.Tensor): The node-to-graph mapping vector [num_nodes].
            solver_config (dict, optional): A dictionary with solver params like
                                            'method', 'step_size', 'time_grid'. Defaults to None.
        
        Returns:
            torch.Tensor: The predicted gene expression tensor [num_nodes, latent_gene_dim].
        """
        # --- 1. Set Up for Inference ---
        self.eval()  # Set model to evaluation mode
        device = next(self.parameters()).device
        
        # Define the time grid for the solver
        T = torch.linspace(0, 1, solver_config["num_steps"], device=device) # Example: 101 steps from t=0 to t=1
        step_size = 1.0 / solver_config["num_steps"]
        # --- 2. Prepare Inputs and the Velocity Function ---
        # Create the initial noise on the correct device
        # B,P,_ = emb.shape
        # x_init = torch.randn(B, P, self.gene_latent_dim, device=device) # 1 is batch size
        if x_init_override is not None:
            x_init = x_init_override
        else:
            B, P, _ = emb.shape
            x_init = torch.randn(B, P, self.gene_latent_dim, device=device)

        # --- 3. Run the Solver ---
        # We disable gradient calculation for a massive speedup and memory save.
        with torch.no_grad():
            # Initialize the solver by passing the model's method DIRECTLY.
            # This is the "velocity_model" that the solver will call.
            solver = ODESolver(velocity_model=self.get_velocity)
            
            # Run the sampling process
            solution_trajectory = solver.sample(
                x_init=x_init,
                time_grid=T,
                method=solver_config["method"],
                step_size=step_size,
                return_intermediates=return_intermediates,
                # These are the **model_extras**
                emb=emb,
                onco_onehot=onco_onehot,
            )

        # --- 4. Return the Final Result ---
        # The final prediction is the state at the last time step (t=1)
        if return_intermediates:
            predicted_genes_latent = solution_trajectory[-1] # B,P,latent_dim
            first_step_matches_init = torch.allclose(solution_trajectory[0], x_init, atol=1e-6)
            print(f"DEBUG MODEL SAMPLE: Does trajectory[0] match the original x_init? {first_step_matches_init}")
            if not first_step_matches_init:
                print("ERROR: The ODE solver is not correctly returning the initial state at t=0!")
        else:
            predicted_genes_latent = solution_trajectory # B,P,latent_dim
        predicted_genes = self.gene_autoencoder.decode(predicted_genes_latent,mask) # B,P,G
        return predicted_genes, solution_trajectory
    

    def forward(self, x_1, emb,  gene_mask_patches, onco_onehot):
        # Replace any NaN values in the input x_1 with 0 because autoencoder is trained without nan
        x_1 = torch.nan_to_num(x_1, nan=0.0)
        # taget latent vector
        z_1_mean, z_1_log_var = self.gene_encode(x_1,gene_mask_patches)
    
        # Use them for the reparameterization trick
        std = torch.exp(0.5 * z_1_log_var)
        epsilon = torch.randn_like(std)
        z_1 = z_1_mean + std * epsilon  # This is your new stochastic latent variable

        with torch.no_grad():
            # Calculate stdev across the patch dimension for each latent feature
            stdev_per_dim = torch.std(z_1.squeeze(0), dim=0) # Shape: [latent_dim]
            # Print the average standard deviation. This is the crucial number.
            print(f"DEBUG: Average StDev of z_1 target: {torch.mean(stdev_per_dim).item():.4f}")

        # And use them to calculate the KL loss
        kl_loss = -0.5 * torch.sum(1 + z_1_log_var - z_1_mean.pow(2) - z_1_log_var.exp(), dim=[1,2])
        kl_loss = torch.mean(kl_loss)

        # --- Stage 2: Perform Flow Matching in the Latent Space ---
        # Calculate the standard deviation of the target latents *in the batch*
        # Use keepdim=True to make broadcasting easy
        z_1_std = torch.std(z_1.flatten(start_dim=1), dim=1, keepdim=True).view(-1, 1, 1)
        # Sample initial noise and scale it to match the target's standard deviation
        z_0 = torch.randn_like(z_1) * z_1_std.clamp(min=1e-6) # clamp for stability

        t = torch.rand(z_1.size(0), device=z_1.device)

        path_sample = self.flow_path.sample(t=t, x_0=z_0, x_1=z_1)
        # The target is the velocity in the latent space
        target_latent_velocity = z_1 - z_0
        # Add Gaussian noise
        x_t = path_sample.x_t
        noise_std = 0.001  # small, relative to latent scale
        x_t_noisy = x_t + torch.randn_like(x_t) * noise_std

        # --- Stage 3: Get Model Predictions for LATENT Velocity ---
        predicted_latent_velocity, moe_gating_logits = self._get_predictions(
            z_t=x_t_noisy,
            t=t,
            emb=emb,
            onco_onehot=onco_onehot,
        )
    
        # --- Stage 4: Calculate Losses ---
        loss_dict_for_logging = {}
        # Initialize total_loss to 0
        total_loss = torch.tensor(0.0, device=x_1.device)

        # KL loss for gene autoencoder, still log it even pre-trained
        loss_dict_for_logging["kl_loss"] = kl_loss.item()
        # only add kl loss to total loss if it's not pretrained
        if not self.gene_vae_pretrained_path:
            total_loss = total_loss +  self.kl_loss_weight * kl_loss  # KL loss is always included

        # 2. Calculate and conditionally apply Load Balancing Loss
        if self.use_moe and moe_gating_logits is not None:
            bal_loss = self._calculate_load_balancing_loss(moe_gating_logits)
            loss_dict_for_logging["load_balancing_loss"] = bal_loss.item()
            
            # ONLY add the balancing loss for backpropagation during TRAINING
            if self.training:
                total_loss += self.load_balancing_loss_weight * bal_loss

        # The primary loss is now the flow loss in the latent space
        flow_loss = self._calculate_flow_loss(predicted_latent_velocity, target_latent_velocity)
        total_loss = total_loss + self.flow_loss_weight * flow_loss
        loss_dict_for_logging["flow_loss"] = flow_loss.item()

        # Final reconstruction loss for logging since the autoeoncoder is pretrained.
        if self.gene_loss_weight > 0:
            # Reconstruct the final latent state from the predicted velocity
            predicted_z_1 = path_sample.x_t + (1 - path_sample.t.unsqueeze(1)) * predicted_latent_velocity
            # Decode the latent prediction back to gene space
            predicted_genes = self.gene_decode(predicted_z_1,gene_mask_patches)
            # Calculate loss using the MASK on the original (non-NaN-replaced) data
            recon_loss = self._calculate_reconstruction_loss(predicted_genes, x_1, gene_mask_patches)
            total_loss = total_loss + self.gene_loss_weight * recon_loss
            loss_dict_for_logging["gene_loss"] = recon_loss.item()

        return total_loss, loss_dict_for_logging, moe_gating_logits
    

class LatentFlow_CFG(LatentFlow):
    def __init__(
        self,
        config,
        gene_dim, 
        time_dim,
        img_emb_dim,
        flow_path,
    ):
        super().__init__(config, gene_dim, time_dim, img_emb_dim, flow_path)

        self.cond_drop_prob = config["CFG_params"]["CFG_dropout"]
        # to remove this from model parameters
        # self.cfg_guidance_scale = config["CFG_params"]["CFG_scale"]
        self.unconditional_embedding = nn.Parameter(torch.randn(1, 1, self.img_emb_dim))


    def _get_predictions(self, z_t, t, emb, onco_onehot):
        
        # spatial encoding of patches
        if self.use_spatial_encoding:
            emb_with_pos = self.spatial_pos_encoder(emb)
            context_aware_emb_batch = self.spatial_transformer(emb_with_pos)
        else:
            context_aware_emb_batch = emb

        # Conditioning for flobal information
        if self.conditioner is not None:
            onco_embedding = self.onco_embedder(onco_onehot)
            query = context_aware_emb_batch
            conditioned_emb = self.conditioner(query=query, context=onco_embedding) # key and value are both context
        else:
            # If no conditioning, just use the spatial embeddings
            conditioned_emb = context_aware_emb_batch

        B, P, _ = z_t.shape
        # Make t compatible: expand to (B, P, 1)
        t_expanded = t.view(B, 1, 1).expand(B, P, 1)
        h = self.velocity_input_norm(torch.cat([z_t, t_expanded, conditioned_emb], dim=-1))  # (n_batch*n_patches,combined_input_dim)
        # Predict velocity
        moe_gating_logits = None
        if self.use_moe:
            h_flat = h.view(B * P, -1)  # (B*P, combined_input_dim)
            predicted_latent_velocity, moe_gating_logits = self.velocity_predictor(h_flat)
            predicted_latent_velocity = predicted_latent_velocity.view(B, P, -1)  # (B, P, gene_latent_dim)
        else:
            predicted_latent_velocity = self.velocity_predictor(h)

        return predicted_latent_velocity, moe_gating_logits


    def get_velocity(self, t, x, emb, onco_onehot,guidance_scale):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if guidance_scale <= 1.0:
            predicted_latent_velocity, _ = self._get_predictions(
                z_t=x, t=t, emb=emb, onco_onehot=onco_onehot
            )
            return predicted_latent_velocity
    
        # --- CFG Inference Logic ---
        # 1. Get the conditional prediction (using the real image embedding)
        v_cond, _ = self._get_predictions(
            z_t=x, t=t, emb=emb, onco_onehot=onco_onehot
        )

        # 2. Get the unconditional prediction
        B, P, _ = emb.shape
        uncond_emb = self.unconditional_embedding.expand(B, P, -1)
        v_uncond, _ = self._get_predictions(
            z_t=x, t=t, emb=uncond_emb, onco_onehot=onco_onehot
        )
        
        # 3. Combine them using the CFG formula
        # v_guided = v_uncond + w * (v_cond - v_uncond)
        guided_velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        return guided_velocity

    def sample(
        self,
        emb,
        onco_onehot,
        mask,
        solver_config: dict,
        guidance_scale,
        return_intermediates=False,
        x_init_override =None,
    ):
        """
        Generates gene expression samples using the ODE solver.

        Args:
            emb (torch.Tensor): The image embeddings [num_nodes, emb_dim].
            onco_onehot (torch.Tensor): The per-graph conditioning [num_graphs, onco_dim].
            batch_vec (torch.Tensor): The node-to-graph mapping vector [num_nodes].
            solver_config (dict, optional): A dictionary with solver params like
                                            'method', 'step_size', 'time_grid'. Defaults to None.
        
        Returns:
            torch.Tensor: The predicted gene expression tensor [num_nodes, latent_gene_dim].
        """
            
        # --- 1. Set Up for Inference ---
        self.eval()  # Set model to evaluation mode
        device = next(self.parameters()).device
        
        # Define the time grid for the solver
        T = torch.linspace(0, 1, solver_config["num_steps"], device=device) # Example: 101 steps from t=0 to t=1
        step_size = 1.0 / solver_config["num_steps"]
        # --- 2. Prepare Inputs and the Velocity Function ---
        # Create the initial noise on the correct device
        # B,P,_ = emb.shape
        # x_init = torch.randn(B, P, self.gene_latent_dim, device=device) # 1 is batch size
        if x_init_override is not None:
            x_init = x_init_override
        else:
            B, P, _ = emb.shape
            x_init = torch.randn(B, P, self.gene_latent_dim, device=device)

        # --- 3. Run the Solver ---
        # We disable gradient calculation for a massive speedup and memory save.
        with torch.no_grad():
            # Initialize the solver by passing the model's method DIRECTLY.
            # This is the "velocity_model" that the solver will call.
            solver = ODESolver(velocity_model=self.get_velocity)
            
            # Run the sampling process
            solution_trajectory = solver.sample(
                x_init=x_init,
                time_grid=T,
                method=solver_config["method"],
                step_size=step_size,
                return_intermediates=return_intermediates,
                # These are the **model_extras**
                emb=emb,
                onco_onehot=onco_onehot,
                guidance_scale=guidance_scale, # NEW: Pass guidance_scale
            )

        # --- 4. Return the Final Result ---
        # The final prediction is the state at the last time step (t=1)
        if return_intermediates:
            predicted_genes_latent = solution_trajectory[-1] # B,P,latent_dim
            first_step_matches_init = torch.allclose(solution_trajectory[0], x_init, atol=1e-6)
            print(f"DEBUG MODEL SAMPLE: Does trajectory[0] match the original x_init? {first_step_matches_init}")
            if not first_step_matches_init:
                print("ERROR: The ODE solver is not correctly returning the initial state at t=0!")
        else:
            predicted_genes_latent = solution_trajectory # B,P,latent_dim
        predicted_genes = self.gene_autoencoder.decode(predicted_genes_latent,mask) # B,P,G
        return predicted_genes, solution_trajectory
    

    def forward(self, x_1, emb,  gene_mask_patches, onco_onehot):
        # Replace any NaN values in the input x_1 with 0 because autoencoder is trained without nan
        x_1 = torch.nan_to_num(x_1, nan=0.0)

        # taget latent vector
        z_1_mean, z_1_log_var = self.gene_encode(x_1,gene_mask_patches)
    
        # Use them for the reparameterization trick
        std = torch.exp(0.5 * z_1_log_var)
        epsilon = torch.randn_like(std)
        z_1 = z_1_mean + std * epsilon  # This is your new stochastic latent variable

        with torch.no_grad():
            # Calculate stdev across the patch dimension for each latent feature
            stdev_per_dim = torch.std(z_1.squeeze(0), dim=0) # Shape: [latent_dim]
            # Print the average standard deviation. This is the crucial number.
            print(f"DEBUG: Average StDev of z_1 target: {torch.mean(stdev_per_dim).item():.4f}")

        # And use them to calculate the KL loss
        kl_loss = -0.5 * torch.sum(1 + z_1_log_var - z_1_mean.pow(2) - z_1_log_var.exp(), dim=[1,2])
        kl_loss = torch.mean(kl_loss)

        # --- Stage 2: Perform Flow Matching in the Latent Space ---
        # Calculate the standard deviation of the target latents *in the batch*
        # Use keepdim=True to make broadcasting easy
        z_1_std = torch.std(z_1.flatten(start_dim=1), dim=1, keepdim=True).view(-1, 1, 1)
        # Sample initial noise and scale it to match the target's standard deviation
        z_0 = torch.randn_like(z_1) * z_1_std.clamp(min=1e-6) # clamp for stability

        t = torch.rand(z_1.size(0), device=z_1.device)

        path_sample = self.flow_path.sample(t=t, x_0=z_0, x_1=z_1)
        # The target is the velocity in the latent space
        target_latent_velocity = z_1 - z_0
        # Add Gaussian noise
        x_t = path_sample.x_t
        noise_std = 0.001  # small, relative to latent scale
        x_t_noisy = x_t + torch.randn_like(x_t) * noise_std

        # --- CONDITIONAL DROPOUT FOR CFG ---
        if self.training and (self.cond_drop_prob > 0):
            # --- NEW: CFG Training Logic for Image Embeddings ---
            B, P, _ = emb.shape # Get Batch size and Patch count
            
            # Create a mask for dropping the condition on a per-sample basis in the batch
            # Shape: (B, 1, 1) to allow broadcasting over the patch and embedding dimensions
            cond_mask = (torch.rand(B, device=emb.device) > self.cond_drop_prob).view(-1, 1, 1)
            
            # Create the unconditional embedding by expanding our learnable parameter
            uncond_emb = self.unconditional_embedding.expand(B, P, -1)
            
            # Choose between the real image embedding and the unconditional one
            effective_emb = torch.where(cond_mask, emb, uncond_emb)
        else:
            effective_emb = emb

        # --- Stage 3: Get Model Predictions for LATENT Velocity ---
        predicted_latent_velocity, moe_gating_logits = self._get_predictions(
            z_t=x_t_noisy,
            t=t,
            emb=effective_emb,
            onco_onehot=onco_onehot,
        )
    
        # --- Stage 4: Calculate Losses ---
        loss_dict_for_logging = {}
        # Initialize total_loss to 0
        total_loss = torch.tensor(0.0, device=x_1.device)

        # KL loss for gene autoencoder, still log it even pre-trained
        loss_dict_for_logging["kl_loss"] = kl_loss.item()
        # only add kl loss to total loss if it's not pretrained
        if not self.gene_vae_pretrained_path:
            total_loss = total_loss +  self.kl_loss_weight * kl_loss  # KL loss is always included

        # 2. Calculate and conditionally apply Load Balancing Loss
        if self.use_moe and moe_gating_logits is not None:
            bal_loss = self._calculate_load_balancing_loss(moe_gating_logits)
            loss_dict_for_logging["load_balancing_loss"] = bal_loss.item()
            
            # ONLY add the balancing loss for backpropagation during TRAINING
            if self.training:
                total_loss += self.load_balancing_loss_weight * bal_loss

        # The primary loss is now the flow loss in the latent space
        flow_loss = self._calculate_flow_loss(predicted_latent_velocity, target_latent_velocity)
        total_loss = total_loss + self.flow_loss_weight * flow_loss
        loss_dict_for_logging["flow_loss"] = flow_loss.item()

        # Final reconstruction loss for logging since the autoeoncoder is pretrained.
        if self.gene_loss_weight > 0:
            # Reconstruct the final latent state from the predicted velocity
            predicted_z_1 = path_sample.x_t + (1 - path_sample.t.unsqueeze(1)) * predicted_latent_velocity
            # Decode the latent prediction back to gene space
            predicted_genes = self.gene_decode(predicted_z_1,gene_mask_patches)
            # Calculate loss using the MASK on the original (non-NaN-replaced) data
            recon_loss = self._calculate_reconstruction_loss(predicted_genes, x_1, gene_mask_patches)
            total_loss = total_loss + self.gene_loss_weight * recon_loss
            loss_dict_for_logging["gene_loss"] = recon_loss.item()

        return total_loss, loss_dict_for_logging, moe_gating_logits



class LatentFlow_CFG_toy(nn.Module):
    def __init__(self, config, time_dim, flow_path):
        super().__init__()     # or remove this line completely

        self.flow_loss_weight = config["loss_config"]["flow_loss_weight"]
        self.load_balancing_loss_weight = config["loss_config"]["load_balancing_loss_weight"]
        self.recon_loss_weight = config["loss_config"]["recon_loss_weight"]

        self.flow_path = flow_path
        self.dropout_rate =config["dropout_rate"]

        self.latent_dim = config['latent_dim']         # This is 2 for our toy problem
        self.condition_dim = config['condition_dim']   # This is also 2
        self.cond_drop_prob = config["CFG_params"]["CFG_dropout"]
        self.unconditional_embedding = nn.Parameter(torch.randn(1, 1, self.condition_dim))

        velocity_input_dim = self.latent_dim + time_dim + self.condition_dim
        self.velocity_input_norm = LayerNorm(velocity_input_dim)

        self.use_moe = config["mixture_of_expert"]["use_moe"]
        if self.use_moe:
            self.num_experts = config["mixture_of_expert"]["num_experts"] # Store for loss calculation
            self.velocity_predictor = MoEVelocityPredictor(
                input_dim=velocity_input_dim,
                output_dim=self.latent_dim,
                hidden_dim=config["mixture_of_expert"]["expert_dim"],
                num_experts=self.num_experts,
                n_heads=config["mixture_of_expert"]["expert_heads"],
                n_layers=config["mixture_of_expert"]["expert_layers"],
                dropout=config["mixture_of_expert"]["moe_dropout"],
                k=config["mixture_of_expert"]["top_k"],
            )
        else:  
            self.velocity_predictor = AttentionVelocityHead(
                input_dim=velocity_input_dim,
                output_dim=self.latent_dim,
                model_dim=config['velocity_head']['model_dim'],     # Add this section to your config
                n_head=config['velocity_head']['n_head'],
                dropout=self.dropout_rate
            )

    def _get_predictions(self, z_t, t, c):
        B, P, _ = z_t.shape

        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=z_t.device, dtype=z_t.dtype)

        # --- FIXED T BROADCAST LOGIC ---
        # Make t compatible: expand to (B, P, 1)
        if t.numel() == 1:
            # scalar t
            t_expanded = t.view(1, 1, 1).expand(B, P, 1)
        elif t.numel() == B:
            # batch t
            t_expanded = t.view(B, 1, 1).expand(B, P, 1)
        elif t.numel() == B * P:
            # node–level t
            t_expanded = t.view(B, P, 1)
        else:
            raise ValueError(
                f"Invalid t shape {t.shape} for z_t shape {z_t.shape}"
        )
            
        # `c` is the condition, equivalent to `emb`
        h = self.velocity_input_norm(torch.cat([z_t, t_expanded, c], dim=-1))    

        # Predict velocity
        moe_gating_logits = None
        if self.use_moe:
            h_flat = h.view(B * P, -1)  # (B*P, combined_input_dim)
            predicted_latent_velocity, moe_gating_logits = self.velocity_predictor(h_flat)
            predicted_latent_velocity = predicted_latent_velocity.view(B, P, -1)  # (B, P, gene_latent_dim)
            assert predicted_latent_velocity.shape[0] == B * P, (
                f"Expected predictor first dim == B*P ({B*P}), got {predicted_latent_velocity.shape}"
    )
        else:
            predicted_latent_velocity = self.velocity_predictor(h)

        return predicted_latent_velocity, moe_gating_logits


    def get_velocity(self, t, x, c,guidance_scale):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        
        if guidance_scale <= 1.0:
            predicted_latent_velocity, _ = self._get_predictions(
                z_t=x, t=t, c=c
            )
            return predicted_latent_velocity
    
        # --- CFG Inference Logic ---
        # 1. Get the conditional prediction (using the real image embedding)
        v_cond, _ = self._get_predictions(
            z_t=x, t=t, c=c
        )

        # 2. Get the unconditional prediction
        B, P, _ = c.shape
        uncond_emb = self.unconditional_embedding.expand(B, P, -1)
        v_uncond, _ = self._get_predictions(
            z_t=x, t=t, c=uncond_emb
        )
        
        # 3. Combine them using the CFG formula
        # v_guided = v_uncond + w * (v_cond - v_uncond)
        guided_velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        return guided_velocity

    def sample(
        self,
        c,
        solver_config: dict,
        guidance_scale,
        return_intermediates=False,
        x_init_override =None,
    ):
        """
        Generates gene expression samples using the ODE solver.

        Args:
            emb (torch.Tensor): The image embeddings [num_nodes, emb_dim].
            onco_onehot (torch.Tensor): The per-graph conditioning [num_graphs, onco_dim].
            batch_vec (torch.Tensor): The node-to-graph mapping vector [num_nodes].
            solver_config (dict, optional): A dictionary with solver params like
                                            'method', 'step_size', 'time_grid'. Defaults to None.
        
        Returns:
            torch.Tensor: The predicted gene expression tensor [num_nodes, latent_gene_dim].
        """
            
        # --- 1. Set Up for Inference ---
        self.eval()  # Set model to evaluation mode
        device = next(self.parameters()).device
        
        # Define the time grid for the solver
        T = torch.linspace(0, 1, solver_config["num_steps"], device=device) # Example: 101 steps from t=0 to t=1
        step_size = 1.0 / solver_config["num_steps"]
        # --- 2. Prepare Inputs and the Velocity Function ---

        if x_init_override is not None:
            x_init = x_init_override
        else:
            B, P, _ = c.shape
            x_init = torch.randn(B, P, self.latent_dim, device=device)

        # --- 3. Run the Solver ---
        # We disable gradient calculation for a massive speedup and memory save.
        with torch.no_grad():
            # Initialize the solver by passing the model's method DIRECTLY.
            # This is the "velocity_model" that the solver will call.
            solver = ODESolver(velocity_model=self.get_velocity)
            
            # Run the sampling process
            solution_trajectory = solver.sample(
                x_init=x_init,
                time_grid=T,
                method=solver_config["method"],
                step_size=step_size,
                return_intermediates=return_intermediates,
                # These are the **model_extras**
                c=c,
                guidance_scale=guidance_scale, # NEW: Pass guidance_scale
            )

        # --- 4. Return the Final Result ---
        # The final prediction is the state at the last time step (t=1)
        if return_intermediates:
            predicted_latent = solution_trajectory[-1] # B,P,latent_dim
            first_step_matches_init = torch.allclose(solution_trajectory[0], x_init, atol=1e-6)
            print(f"DEBUG MODEL SAMPLE: Does trajectory[0] match the original x_init? {first_step_matches_init}")
            if not first_step_matches_init:
                print("ERROR: The ODE solver is not correctly returning the initial state at t=0!")
        else:
            predicted_latent = solution_trajectory # B,P,latent_dim
        # predicted_genes = self.gene_autoencoder.decode(predicted_genes_latent) # B,P,G
        return predicted_latent, solution_trajectory

    def _calculate_load_balancing_loss(self, gating_logits: torch.Tensor):
        """
        Calculates the load balancing loss for the MoE.
        This loss encourages the gating network to distribute tokens evenly.
        
        Args:
            gating_logits (torch.Tensor): Raw logits from the gating network
                                          of shape (B, num_experts).
        
        Returns:
            torch.Tensor: A scalar loss value.
        """
        if gating_logits is None:
            return 0.0

        # gating_logits shape: (B * P, num_experts)
        # P is number of patches, B is batch size
        
        # Calculate the router probabilities over all experts
        router_probs = F.softmax(gating_logits, dim=-1)

        # Calculate f_i: Fraction of router probability allocated to expert i
        # This measures the "confidence" in each expert.
        # We take the mean over the batch dimension.
        f_i = router_probs.mean(dim=0) # Shape: (num_experts,)

        # Calculate P_i: Fraction of tokens dispatched to expert i
        # This measures the actual routing decisions.
        # First, get the chosen expert indices from the logits
        _, top_indices = torch.topk(gating_logits, k=self.velocity_predictor.gating.k, dim=-1)
        # Convert to a one-hot representation of which experts were chosen
        one_hot_choices = F.one_hot(top_indices, num_classes=self.num_experts).float()
        # Sum over the k choices for each token
        one_hot_choices = one_hot_choices.sum(dim=1) # Shape: (B*P, num_experts)
        # Now, calculate the mean over the batch
        P_i = one_hot_choices.mean(dim=0) # Shape: (num_experts,)

        # The loss is the scaled dot product of these two distributions
        # This encourages both distributions to be uniform
        load_balancing_loss = self.num_experts * (f_i * P_i).sum()
        
        return load_balancing_loss
    

    def forward(self, z_1, c):
        
        # generave a new random z_0 at each forward
        z_0 = torch.randn_like(z_1)
        t = torch.rand(z_1.size(0), device=z_1.device)

        path_sample = self.flow_path.sample(t=t, x_0=z_0, x_1=z_1)
        x_t = path_sample.x_t

        target_latent_velocity = z_1 - z_0

        # --- CONDITIONAL DROPOUT FOR CFG ---
        if self.training and (self.cond_drop_prob > 0):
            # --- NEW: CFG Training Logic for Image Embeddings ---
            B, P, _ = c.shape # Get Batch size and Patch count
            
            # Create a mask for dropping the condition on a per-sample basis in the batch
            # Shape: (B, 1, 1) to allow broadcasting over the patch and embedding dimensions
            cond_mask = (torch.rand(B, device=c.device) > self.cond_drop_prob).view(-1, 1, 1)
            
            # Create the unconditional embedding by expanding our learnable parameter
            uncond_emb = self.unconditional_embedding.expand(B, P, -1)
            
            # Choose between the real image embedding and the unconditional one
            c_con = torch.where(cond_mask, c, uncond_emb)
        else:
            c_con = c
       
        # make t explicit shape (B,1,1) for clarity:
        t_b11 = path_sample.t.view(-1, 1, 1)   # <- (B,1,1)

        # --- Stage 3: Get Model Predictions for LATENT Velocity ---
        predicted_latent_velocity, moe_gating_logits = self._get_predictions(
            z_t=x_t,
            t=t,
            c=c_con,
        )
      
        # --- Stage 4: Calculate Losses ---
        loss_dict_for_logging = {}
        # Initialize total_loss to 0
        total_loss = torch.tensor(0.0, device=z_1.device)

        # 2. Calculate and conditionally apply Load Balancing Loss
        if self.use_moe and moe_gating_logits is not None:
            bal_loss = self._calculate_load_balancing_loss(moe_gating_logits)
            loss_dict_for_logging["load_balancing_loss"] = bal_loss.item()
            
            # ONLY add the balancing loss for backpropagation during TRAINING
            if self.training:
                total_loss += self.load_balancing_loss_weight * bal_loss

        # The primary loss is now the flow loss in the latent space
        flow_loss = F.mse_loss(predicted_latent_velocity, target_latent_velocity)
        total_loss = total_loss + self.flow_loss_weight * flow_loss
        loss_dict_for_logging["flow_loss"] = flow_loss.item()
        # Reconstruct the final latent state from the predicted velocity
        predicted_z_1 = path_sample.x_t + (1 - t_b11) * predicted_latent_velocity
        # predicted_z_1 = x_t + (1 - path_sample.t) * predicted_latent_velocity
        # Calculate loss using the MASK on the original (non-NaN-replaced) data
        recon_loss = F.mse_loss(z_1, predicted_z_1)
        total_loss = total_loss + self.recon_loss_weight * recon_loss
        loss_dict_for_logging["recon_loss"] = recon_loss.item()

        return total_loss, loss_dict_for_logging, moe_gating_logits