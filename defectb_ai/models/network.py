from typing import List

import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl

import torch.nn.functional as F



class SelfAttention(nn.Module):
    """
    Thin wrapper around PyTorch MultiheadAttention for 1D densities of states.
    Keeps the projections lightweight and leaves positional encodings out.
    """

    def __init__(self, input_dim=1, embed_dim=128, num_heads=1, out_dim=None, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(embed_dim, out_dim or input_dim)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L) or (B, L, C)
        key_padding_mask: (B, L) with True where positions are PAD (optional)
        attn_mask: (L, L) or (B*num_heads, L, L) (optional; causal or custom masks)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)

        h = self.proj_in(x)  # (B, L, E)
        attn_output, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        y = self.proj_out(attn_output)  # (B, L, out_dim)

        return y.squeeze(-1) if y.shape[-1] == 1 else y


class Net(nn.Module):
    def __init__(self, input_size: int, conv_layer: List[int], num_heads: int, head_dim: int, output_dims: List[int],
                 target_size: int, dropout: float, train=True) -> None:
        super().__init__()
        self.dropout_rate = dropout
        self.training = train
        layers: List[nn.Module] = []

        self.num_heads = num_heads
        self.head_dim = head_dim
        # Attention mechanism
        out_chan, kernel_size, stride, padding = conv_layer[0]
        self.conv1D = nn.Conv1d(in_channels=1,
                                out_channels=out_chan,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode='reflect')

        out_conv_1_dim = (input_size - kernel_size + 2 * padding) // stride + 1

        #self.attention = SelfAttention(input_dim=out_conv_1_dim, num_heads=self.num_heads, embed_dim=self.head_dim)
        self.attention = SelfAttention(input_dim=1,
                                       embed_dim=num_heads * head_dim,
                                       num_heads=num_heads,
                                       out_dim=1,
                                       dropout=dropout)
       # out_atten = num_heads * head_dim
        out_atten = out_conv_1_dim
        input_dim = out_atten
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, target_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Split data into chunks
        conv1D_out = self.conv1D(data.unsqueeze(1))
        # conv1D_out = F.relu(self.conv1D(data.unsqueeze(1)))
        pooled_out = conv1D_out.mean(dim=1)
        attn_result = self.attention(pooled_out)

        # print(attn_result.shape)
        # Dropout layers already respect module training state, so no manual rescaling needed
        out = self.layers(attn_result)
        out[:, 1] = nn.ReLU()(out[:, 1])
        return out


class Transformer(pl.LightningModule):
    def __init__(self, input_size: int, conv_layer: List[int], num_heads: int, head_dim: int, output_dims: List[int],
                 learning_rate: float, alpha: float,  # target_min_max: List[int],
                 target_size, dropout, dr: float, train=True, weights=None):
        # super(NeuralNetwork, self).__init__()
        super().__init__()
        self.input_size = input_size
        self.automatic_optimization = True
        self.dropout_rate = dropout
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()
        self.save_path = 'saved_nn/'
        self.model = Net(input_size=input_size,
                         conv_layer=conv_layer,
                         num_heads=num_heads,
                         head_dim=head_dim,
                         output_dims=output_dims,
                         target_size=target_size,
                         dropout=dropout,
                         train=train)
        self.alpha = alpha
        #   self.target_min_max = target_min_max
        self.l2_lambda = 1e-5
        self.dr = dr
        self.training_losses = []
        self.validation_losses = []
        if weights is not None:
            self.loss_weights = torch.tensor(weights)
        else:
            self.loss_weights = weights

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, self.input_size))

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.training_losses.append({"loss": loss})
        # l2_reg = self.l2_regularization()
        # loss += self.l2_lambda * l2_reg
        self.log_dict(
            {
                "train_loss": loss,
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # self.training_losses.append(loss)
        return {"loss": loss, "scores": scores, "y": y}

    def l2_regularization(self) -> torch.Tensor:
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return l2_reg

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.validation_losses.append({"val_loss": loss})
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # self.validation_losses.append(loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        x = batch
        scores = self.forward(x)
        return scores

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x['loss'] for x in self.training_losses]).mean()
        self.logger.experiment.add_scalars('losses', {'train': avg_train_loss}, self.current_epoch)
        self.training_losses.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_losses]).mean()
        self.logger.experiment.add_scalars('losses', {'valid': avg_val_loss}, self.current_epoch)
        self.validation_losses.clear()
    # def on_train_end(self):
    #     np.savetxt('val_loss.dat', np.array(self.validation_losses))
    #     np.savetxt('train_loss.dat', np.array(self.training_losses))
    def laplace_dist(self, scores):
        U = scores[:, 0]
        sigma = scores[:, 1]
        u_min, u_max = self.target_min_max[0]
        sigma_min, sigma_max = self.target_min_max[1]
        scaled_U = U * (u_max - u_min) + u_min
        scaled_sigma = sigma * (sigma_max - sigma_min) + sigma_min

        b = torch.sqrt(scaled_sigma / 2)
        eps_prist = 2.37
        d = .2512
        return eps_prist - (eps_prist - scaled_U) * torch.exp(-d / (b))

    def weighted_parameter_loss(self, predictions, targets, weights=None):
        """
        Custom loss function with weighted importance for parameters.

        Args:
            predictions (torch.Tensor): The predicted parameters.
            targets (torch.Tensor): The true parameters.
            weights (torch.Tensor): The weights for each parameter.

        Returns:
            torch.Tensor: The weighted loss.
        """
        if weights == None:
            loss = torch.mean((predictions - targets) ** 2)
        else:
            loss = torch.mean(weights * (predictions - targets) ** 2)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        # loss = self.loss_fn(scores, y)
        loss = self.weighted_parameter_loss(scores, y, self.loss_weights)
        # scores_lp = self.laplace_dist(scores)

        #  y_lp = self.laplace_dist(y)

        #  loss_lp = self.loss_fn(scores_lp, y_lp)
        # total_loss = (1 - self.alpha) * loss + self.alpha * loss_lp
        # middle_lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        # print('>>> middle_lr  ', middle_lr)
        return loss, scores, y

    # def configure_optimizer(self):
    #     optimizer = torch.optim.Adam(...)
    #     lr_scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
    #         'name': 'my_logging_name'
    #     }
    #     return [optimizer], [lr_scheduler]
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.dr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=1-self.dr)
        return [optimizer], [scheduler]


class Conv1DSelfAtten(Transformer):
    """
    Backwards-compatible alias for the LightningModule used in training scripts.
    """

    def __init__(self, input_size: int, conv_layer: List[int], num_heads: int, head_dim: int, output_dims: List[int],
                 learning_rate: float, alpha: float, target_size, dropout: float, dr: float, weights=None):
        super().__init__(
            input_size=input_size,
            conv_layer=conv_layer,
            num_heads=num_heads,
            head_dim=head_dim,
            output_dims=output_dims,
            target_size=target_size,
            learning_rate=learning_rate,
            alpha=alpha,
            dropout=dropout,
            dr=dr,
            train=True,
            weights=weights,
        )
