import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_features_1, hidden_features_2, output_features):
        """
        A 3-layer MLP that predicts gene expression for each patch individually.

        Args:
            embedding_dim (int): The dimension of each input patch embedding.
            hidden_features_1 (int): The number of neurons in the first hidden layer.
            hidden_features_2 (int): The number of neurons in the second hidden layer.
            output_features (int): The number of output features (e.g., number of genes).
        """
        super(SimpleMLP, self).__init__()
        
        # No pooling layer is needed anymore.
        
        # --- MLP Layers ---
        # The MLP operates on each patch embedding of size `embedding_dim`.
        self.fc1 = nn.Linear(embedding_dim, hidden_features_1)
        self.ln1 = nn.LayerNorm(hidden_features_1)
        
        self.fc2 = nn.Linear(hidden_features_1, hidden_features_2)
        self.ln2 = nn.LayerNorm(hidden_features_2)
        
        # The output layer predicts the gene expression vector.
        self.fc3 = nn.Linear(hidden_features_2, output_features)
        
        # --- Loss Function ---
        # Mean Squared Error is suitable for comparing predicted and actual expression vectors.
        self.loss_fn = nn.MSELoss()

    def _calculate_gene_loss(self, predictions, targets):

        """
        The sole loss function for this model. Calculates MSE on valid targets.
        """
        # Create a mask to ignore NaN values in the ground truth
        mask = ~torch.isnan(targets)
        # Handle the edge case where a batch has no valid labels
        if not mask.any():
            # Return a zero-loss tensor that is still connected to the graph
            return predictions.sum() * 0.0
            
        # Calculate MSE loss only on the valid (non-NaN) data points
        return F.mse_loss(predictions[mask], targets[mask])
    
    def _get_predictions(self,x):
        """
        Forward pass of the MLP for per-patch prediction.

        Args:
            x (torch.Tensor): Input image embeddings, shape [n_patches, embedding_dim].
            y (torch.Tensor): Ground truth gene expression, shape [num_genes].
        
        Returns:
            tuple:
                - torch.Tensor: The computed total loss (averaged over all patches).
                - dict: A dictionary with the named scalar loss for logging.
        """
        # --- Forward Pass through MLP Layers ---
        # We pass the entire batch of patches through the MLP.
        # `x` is already in the correct shape: [n_batch=1, n_patches, embedding_dim].
        x = x.squeeze(0) #remove batch dim
        x_out = self.fc1(x)
        x_out = self.ln1(x_out)
        x_out = F.relu(x_out)

        x_out = self.fc2(x_out)
        x_out = self.ln2(x_out)
        x_out = F.relu(x_out)
        
        # `logits` will have shape [n_patches, output_features]
        logits = self.fc3(x_out)
        return logits


    def forward(self, x, y):
        """
        Forward pass of the MLP for per-patch prediction.

        Args:
            x (torch.Tensor): Input image embeddings, shape [n_patches, embedding_dim].
            y (torch.Tensor): Ground truth gene expression, shape [num_genes].
        
        Returns:
            tuple:
                - torch.Tensor: The computed total loss (averaged over all patches).
                - dict: A dictionary with the named scalar loss for logging.
        """
        
        # `logits` will have shape [n_patches, output_features]
        logits = self._get_predictions(x)

        # --- Calculate Loss ---
        # The loss is calculated between the per-patch predictions and the single ground truth vector.
        gene_loss = self._calculate_gene_loss(logits, y.squeeze(0))
        total_loss = gene_loss
        
        # --- Prepare Logging Dictionary ---
        loss_dict_for_logging = {"gene_loss": gene_loss.item()}
        
        return total_loss, loss_dict_for_logging

