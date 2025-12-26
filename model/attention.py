
import math
import torch
import torch.nn as nn
from copy import deepcopy

class MuModuleList(nn.ModuleList):
    def forward(self,x,mu):
        for layer in self:
            if type(layer) == DynamicLinear:
                x=layer(x,mu)
            else:
                x=layer(x)
        return x

class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim, need_out_dim, k=30):
        super(ParamDecoder, self).__init__()
        self.need_in_dim = need_in_dim
        self.need_out_dim = need_out_dim
        self.k = k
        self.decoder = nn.Linear(mu_dim, need_in_dim * k)
        self.V = nn.parameter.Parameter(torch.zeros(k, need_out_dim))

    def forward(self, t_feat):
        B = t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B, self.need_in_dim, self.k)  # B x need_in_dim x k
        param = torch.einsum('bik,kj->bij', U, self.V).reshape(B, -1)
        return param

class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias = bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim)

    def forward(self, x, mu):
        param = rearrange(self.decoder(mu), 'B (dim_A dim_B) -> B dim_A dim_B', dim_A=self.in_dim + 1,
                          dim_B=self.out_dim)
        weight = param[:, :-1, :]
        bias = param[:, -1, :]
        x = torch.einsum('b...d,bde->b...e', x, weight)
        if self.bias:
            bias = bias.view(((bias.shape[0],) + (1,) * (len(x.size()) - 2) + (bias.shape[-1],)))
            x = x + bias
        return x

class SentimentModulated(nn.Module):
    def __init__(self, gate_channels,text_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(SentimentModulated, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,text_dim)
        ])
        self.pool_types = pool_types
    def forward(self, x ,mu):
        B=x.shape[0] # batchsize
        D=x.shape[-1] # dimension
        channel_att_sum = None
        for pool_type in self.pool_types:
            pre_pool=x.view(B,-1,D)
            if pool_type=='avg':
                avg_pool=torch.mean(pre_pool,dim=1)
                channel_att_raw = self.mlp( avg_pool ,mu)
            elif pool_type=='max':
                max_pool=torch.max(pre_pool,dim=1).values
                channel_att_raw = self.mlp( max_pool ,mu)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum )
        scale = scale.view(((scale.shape[0],)+(1,)*(len(x.size())-2)+(scale.shape[-1],)))

        return x + (x * scale)

class Attention(nn.Module):
    def __init__(self, num_attention_head, hidden_size, dropout_prob) -> None:
        super().__init__()
        self.num_attention_head = num_attention_head
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_head)
        self.query = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.key = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.value = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None, encoder_attention_mask=None, output_attention=False):
        query_layer = self.transpose_for_scores(self.query(query))
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))

        if encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attention else (context_layer, )
        return outputs


class LayerNorm(nn.Module):
    def __init__(self, x_size, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.ones_tensor = nn.Parameter(torch.ones(x_size))
        self.zeros_tensor = nn.Parameter(torch.zeros(x_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout, layer_norm_eps) -> None:
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size
        )
        self.w_2 = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=hidden_size,
            eps=layer_norm_eps
        )
        self.dropout_1 = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(inter))


class SublayerConnecttion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1) -> None:
        super(SublayerConnecttion, self).__init__()
        self.layer_norm = LayerNorm(x_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))

def clone_module_to_modulelist(module, module_num):
    return nn.ModuleList([deepcopy(module) for _ in range(module_num)])

class Crossmodal_Attention(nn.Module):
    def __init__(self, num_attention_head, hidden_size, intermediate_size, dropout_prob, layer_norm_eps) -> None:
        super(Crossmodal_Attention, self).__init__()
        self.attn = Attention(
            num_attention_head=num_attention_head,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        self.sublayer_connection_list = clone_module_to_modulelist(
            module=SublayerConnecttion(
                hidden_size=hidden_size,
                dropout=dropout_prob
            ),
            module_num=2
        )

    def forward(self, query, key, value, encoder_attention_mask=None):
        x = self.sublayer_connection_list[0](
            query, lambda query: self.attn(
                query=query,
                key=key,
                value=value,
                encoder_attention_mask=encoder_attention_mask)[0]
        )
        return self.sublayer_connection_list[1](x, self.feed_forward)
