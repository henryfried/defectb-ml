from typing import List

import torch
from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F




class Conv1DSelfAtten(pl.LightningModule):
    def __init__(self, input_size: int, conv_layer: List[int], num_heads: int, head_dim: int, output_dims: List[int],
                 learning_rate: float, alpha: float, target_size, dropout: float, dr: float, weights=None):
        """
        Transformer model implemented using PyTorch Lightning.

        Parameters:
        - input_size: The size of the input features.
        - conv_layer: Parameters for the convolutional layers.
        - num_heads: The number of attention heads in the SelfAttention module.
        - head_dim: The dimension of each attention head.
        - output_dims: Dimensions for the fully connected layers.
        - learning_rate: The learning rate for training.
        - alpha: A parameter for mixing losses (e.g., for Laplace distribution).
        - target_size: The size of the output layer.
        - dropout: The dropout rate.
        - dr: Decay rate for learning rate.
        - weights: Custom loss weights, if any.
        """
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()
        self.model = Net(input_size=input_size,
                         conv_layer=conv_layer,
                         num_heads=num_heads,
                         head_dim=head_dim,
                         output_dims=output_dims,
                         target_size=target_size,
                         dropout=dropout)
        self.alpha = alpha
        self.l2_lambda = 1e-5
        self.dr = dr
        self.training_losses = []
        self.validation_losses = []
        self.loss_weights = torch.tensor(weights) if weights is not None else None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the  model. """
        return self.model(data.view(-1, self.input_size))

    def training_step(self, batch, batch_idx):
        """ Perform one training step. """
        loss, scores, y = self._common_step(batch, batch_idx)
        self.training_losses.append({"loss": loss})
        self.log_dict({"train_loss": loss}, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        """ Perform one validation step. """
        loss, scores, y = self._common_step(batch, batch_idx)
        self.validation_losses.append({"val_loss": loss})
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """ Perform one test step. """
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        """ Make predictions using the model. """
        x = batch
        scores = self.forward(x)
        return scores

    def on_train_epoch_end(self):
        """ Log average training loss at the end of each epoch. """
        avg_train_loss = torch.stack([x['loss'] for x in self.training_losses]).mean()
        self.logger.experiment.add_scalars('losses', {'train': avg_train_loss}, self.current_epoch)
        self.training_losses.clear()

    def on_validation_epoch_end(self):
        """ Log average validation loss at the end of each epoch. """
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_losses]).mean()
        self.logger.experiment.add_scalars('losses', {'valid': avg_val_loss}, self.current_epoch)
        self.validation_losses.clear()

    def _common_step(self, batch, batch_idx):
        """ Common step for training, validation, and testing. """
        x, y = batch
        scores = self.forward(x)
        loss = self.weighted_parameter_loss(scores, y, self.loss_weights)
        return loss, scores, y

    def weighted_parameter_loss(self, predictions, targets, weights=None):
        """
        Custom weighted loss function for parameters.

        Args:
        - predictions: Predicted tensor.
        - targets: Ground truth tensor.
        - weights: Weights for each parameter in the loss calculation.

        Returns:
        - Weighted loss as a tensor.
        """
        if weights is None:
            loss = torch.mean((predictions - targets) ** 2)
        else:
            loss = torch.mean(weights * (predictions - targets) ** 2)
        return loss

    def configure_optimizers(self):
        """ Configure optimizer and scheduler for training. """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.dr)
        return [optimizer], [scheduler]
    
class Net(nn.Module):
    def __init__(self, input_size: int, conv_layer: List[int], num_heads: int, head_dim: int, output_dims: List[int],
                 target_size: int, dropout: float) -> None:
        """
        Initialize the network consisting of convolutional layers, attention mechanism, and multi-layer perceptron.

        Parameters:
        - input_size: The size of the input features.
        - conv_layer: List containing parameters for the 1D convolution layer.
        - num_heads: The number of attention heads in the SelfAttention mechanism.
        - head_dim: The dimension of each attention head.
        - output_dims: The dimensions for the fully connected layers.
        - target_size: The size of the target/output layer.
        - dropout: Dropout rate to prevent overfitting.
        - train: A flag indicating whether the model is in training mode or not.
        """
        super().__init__()
        self.dropout_rate = dropout
        #self.training = train
        layers: List[nn.Module] = []
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Define a 1D convolutional layer
        out_chan, kernel_size, stride, padding = conv_layer[0]
        self.conv1D = nn.Conv1d(in_channels=1,
                                out_channels=out_chan,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode='reflect')

        # Calculate the output dimension of the convolution layer
        conv_out_dim = (input_size - kernel_size + 2 * padding) // stride + 1

        # Define a max pooling layer, small kernel to keep the features of the input
        pool_kernel_size = 4
        pool_stride = 2
        pool_padding = 0
        self.pool = nn.MaxPool1d(pool_kernel_size, pool_stride, pool_padding)

        # Calculate the output dimension after pooling
        pool_out_dim = ((conv_out_dim - pool_kernel_size + 2 * pool_padding) // pool_stride + 1) * out_chan
        out_conv_1_dim = (input_size - kernel_size + 2 * padding) // stride + 1
        # Define the attention layer
        # self.attention = SelfAttentionWeights(input_dim=out_conv_1_dim, num_heads=self.num_heads, head_dim=self.head_dim) 
       
        # Define the fully connected layers after attention
       
        
        self.attention = SelfAttention(
            input_dim=1,
            qk_dim=num_heads,              # 1 dim per head
            v_dim=num_heads * head_dim,
            num_heads=num_heads,
            out_dim=1,
            dropout=dropout,
        )
        out_atten = out_conv_1_dim
        input_dim = out_atten  #SelfAttention
         #input_dim = num_heads * head_dim  #SelfAttentionWeights
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = output_dim

        # Output layer
        layers.append(nn.Linear(input_dim, target_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters:
        - data: The input tensor (batch_size, input_size).

        Returns:
        - out: The output tensor after passing through convolution, attention, and MLP.
        """
        batch_size, _ = data.size()
        # Apply 1D convolution
        conv1D_out = self.conv1D(data.unsqueeze(1))

        # Apply max pooling
        # pooled_out = self.pool(conv1D_out)
        pooled_out = conv1D_out.mean(dim=1)
        # Flatten the pooled output and pass through the attention mechanism
        # flattened_output = pooled_out.view(batch_size, -1)
        # attn_result = self.attention(flattened_output)
        attn_result = self.attention(pooled_out)

        # Apply fully connected layers with dropout and ReLU activation
        out = self.layers(attn_result)
        out[:, 1] = nn.ReLU()(out[:, 1])
        return out

class SelfAttention(nn.Module):
    """
    Self-attention with asymmetric dimensions:
    - Q/K dimension = qk_dim (default 1)
    - V dimension   = v_dim (default 128)

    Note:
        qk_dim must be divisible by num_heads.
        v_dim must be divisible by num_heads.
        If qk_dim = 1, then num_heads must be 1.
    """
    def __init__(
        self,
        input_dim=1,
        qk_dim=1,
        v_dim=128,
        num_heads=1,
        out_dim=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        if qk_dim % num_heads != 0:
            raise ValueError(f"qk_dim ({qk_dim}) must be divisible by num_heads ({num_heads}).")
        if v_dim % num_heads != 0:
            raise ValueError(f"v_dim ({v_dim}) must be divisible by num_heads ({num_heads}).")

        self.out_dim = out_dim or input_dim

        # Separate projections for asymmetric attention
        self.proj_q = nn.Linear(input_dim, qk_dim, bias=bias)
        self.proj_k = nn.Linear(input_dim, qk_dim, bias=bias)
        self.proj_v = nn.Linear(input_dim, v_dim, bias=bias)

        self.mha = nn.MultiheadAttention(
            embed_dim=qk_dim,   # query dimension
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            kdim=qk_dim,
            vdim=v_dim,
        )

        # Output is in query space (qk_dim)
        self.proj_out = nn.Linear(qk_dim, self.out_dim, bias=bias)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L) or (B, L, C)
        key_padding_mask: (B, L) with True where positions are PAD
        attn_mask: (L, L) or (B*num_heads, L, L)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        h, _ = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )

        y = self.proj_out(h)

        return y.squeeze(-1) if y.shape[-1] == 1 else y
    
    

class SelfAttentionWeights(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        """
        Initialize the SelfAttention module that calculates self attention between Q and K of length head_dim.

        Parameters:
        - input_dim: The input feature size.
        - num_heads: Number of attention heads in the multi-head attention mechanism.
        - head_dim: The dimension of each attention head.
        """
        super(SelfAttentionWeights, self).__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = input_dim

        # Inner dimension is the product of the number of heads and the dimension of each head
        inner_dim = num_heads * head_dim

        # Linear transformations for queries, keys, and values
        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(input_dim, inner_dim, bias=False)

        # Scaling factor for the attention scores (1 / sqrt(head_dim))
        self.scale = head_dim ** -0.5

        # Variable to hold the attention scores (for debugging/monitoring)
        self.attn_scores = None

    def forward(self, x):
        """
        Perform a forward pass through the SelfAttention module.

        Parameters:
        - x: Input tensor (batch_size, input_dim).

        Returns:
        - attn_output: The final attention-weighted output tensor.
        """
        batch_size, dim = x.size()
 
        query = self.to_q(x).view(batch_size, self.num_heads, self.head_dim).unsqueeze(-1)
        key = self.to_k(x).view(batch_size, self.num_heads, self.head_dim).unsqueeze(-1)
        value = self.to_v(x).view(batch_size, self.num_heads, self.head_dim).unsqueeze(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        attn_weights = F.softmax(scores, dim=-1)
        self.attn_scores = attn_weights

        attn_output = torch.matmul(attn_weights, value).contiguous()

        # Flatten the output and return it
        attn_output = attn_output.view(batch_size, -1)
        return attn_output