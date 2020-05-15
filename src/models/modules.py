import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
# from .attention import Attention as PtAttention


class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0,
                 batchnorm=False,
                 bias=False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


class BatchMLP(nn.Module):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined 
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=2,
                 dropout=0,
                 batchnorm=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        # First layer
        self.initial = NPBlockRelu2d(input_size,
                                     hidden_size,
                                     dropout=dropout,
                                     batchnorm=batchnorm)
        self.encoder = nn.Sequential(*[
            NPBlockRelu2d(
                hidden_size, hidden_size, dropout=dropout, batchnorm=batchnorm)
            for _ in range(num_layers - 1)
        ])
        # Final transformation (not really a layer) without ReLU
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)


class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        torch.nn.init.normal_(self.linear.weight, std=in_channels**-0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    """From KurochkinAlexey/Recurrent-neural-processes."""
    def __init__(
            self,
            hidden_dim,
            attention_type,
            attention_layers=2,
            n_heads=8,
            x_dim=1,
            rep="mlp",
            dropout=0,
            batchnorm=False,
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.batch_mlp_k = BatchMLP(
                x_dim,
                hidden_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            self.batch_mlp_q = BatchMLP(
                x_dim,
                hidden_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )

        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        elif attention_type == "ptmultihead":
            self._W = torch.nn.MultiheadAttention(hidden_dim,
                                                  n_heads,
                                                  bias=False,
                                                  dropout=dropout)
            self._attention_func = self._pytorch_multihead_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.batch_mlp_k(k)
            q = self.batch_mlp_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_) * scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1]**0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        o = self._W(q, k, v)[0]
        return o.permute(1, 0, 2)


class LatentEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim=32,
            latent_dim=32,
            self_attention_type="dot",
            n_latent_encoder_layers=3,
            min_std=0.01,
            batchnorm=False,
            dropout=0,
            attention_dropout=0,
            use_lvar=False,
            use_self_attn=False,
            attention_layers=2,
    ):
        super().__init__()
        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self._encoder = nn.ModuleList([
            NPBlockRelu2d(hidden_dim,
                          hidden_dim,
                          batchnorm=batchnorm,
                          dropout=dropout)
            for _ in range(n_latent_encoder_layers)
        ])
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self._min_std = min_std
        self._use_lvar = use_lvar
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        """Accept x and y, provided as [B, x_dim] and [B, y_dim].
        Defined in collate_fns_GP
        """
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        encoded = self._input_layer(encoder_input)
        for layer in self._encoder:
            encoded = layer(encoded)

        # Aggregator: take the mean over all points
        if self._use_self_attn:
            attention_output = self._self_attention(encoded, encoded, encoded)
            mean_repr = attention_output.mean(dim=1)
        else:
            mean_repr = encoded.mean(dim=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))

        # Then apply further linear layers to output latent mu and log sigma
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        if self._use_lvar:
            # Clip it in the log domain, so it can only approach self.min_std, this helps avoid mode collapase
            # 2 ways, a better but untested way using the more stable log domain, and the way from the deepmind repo
            log_var = F.logsigmoid(log_var)
            log_var = torch.clamp(log_var, np.log(self._min_std),
                                  -np.log(self._min_std))
            sigma = torch.exp(0.5 * log_var)
        else:
            sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(
                log_var * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var


class DeterministicEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            x_dim,
            hidden_dim=32,
            n_det_encoder_layers=3,
            self_attention_type="dot",
            cross_attention_type="dot",
            use_self_attn=False,
            attention_layers=2,
            batchnorm=False,
            dropout=0,
            attention_dropout=0,
    ):
        super().__init__()
        self._use_self_attn = use_self_attn
        # First layer
        self._input_layer = NPBlockRelu2d(input_dim, hidden_dim)
        self._d_encoder = nn.ModuleList([
            NPBlockRelu2d(
                hidden_dim,
                hidden_dim,
                batchnorm=batchnorm,
                dropout=attention_dropout,
            ) for _ in range(n_det_encoder_layers - 1)
        ])
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            attention_layers=attention_layers,
        )

    def forward(self, context_x, context_y, target_x):
        """Returns [B, total_points |C|, rep_summary_dim/hidden_dim] i.e. all r_i's, need to be averaged later to form r_C.
        """
        # Concatenate x and y along the filter axes
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        d_encoded = self._input_layer(d_encoder_input)
        for layer in self._d_encoder:
            d_encoded = layer(d_encoded)

        if self._use_self_attn:
            d_encoded = self._self_attention(d_encoded, d_encoded, d_encoded)

        # Apply attention(k, v, q), on (x_c's, r_i's, and x_T)
        print("SHAPE OF all r_i's to feed into cross ATTENTION")
        print(d_encoded.shape)
        h = self._cross_attention(context_x, d_encoded, target_x)
        print("SHAPE OF OUTPUT FROM CROSS ATTENTION, r_C")
        print(h.shape)

        return h


class Decoder(nn.Module):
    def __init__(
            self,
            x_dim,
            y_dim,
            hidden_dim=32,
            latent_dim=32,
            n_decoder_layers=3,
            use_deterministic_path=True,
            min_std=0.01,
            use_lvar=False,
            batchnorm=False,
            dropout=0,
    ):
        super(Decoder, self).__init__()
        # Basically the first layer.
        self._target_transform = NPBlockRelu2d(x_dim, hidden_dim)
        if use_deterministic_path:
            input_dim = 2 * hidden_dim + latent_dim
        else:
            input_dim = hidden_dim + latent_dim
        output_dim = 2 * y_dim
        self._y_dim = y_dim
        self._decoder = BatchMLP(input_dim,
                                 hidden_dim,
                                 output_dim,
                                 num_layers=n_decoder_layers,
                                 dropout=dropout,
                                 batchnorm=batchnorm)
        # self._mean = nn.Linear(input_dim, y_dim)
        # self._std = nn.Linear(input_dim, y_dim)
        self._use_deterministic_path = use_deterministic_path
        self._min_std = min_std
        self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        x = self._target_transform(target_x)

        # concatenate target_x and representation
        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)

        representation = torch.cat([z, x], dim=-1)

        # Pass final axis through MLP
        representation = self._decoder(representation)

        # Get the mean and the variance, each of shape [B, |T|, y_dim]
        mean, log_sigma = torch.split(representation, self._y_dim,
                                      dim=-1)  # it returns a tuple

        # Bound or clamp the variance
        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std),
                                    -math.log(1e-5))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_sigma
