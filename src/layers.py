import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from typing import Union
import numpy as np
# 导入 TRLinear
from tednet.tnn.tensor_ring import TRLinear

# --- 辅助函数：因子分解 (用于 TRLinear 的 in_shape/out_shape) ---
def _factorize_dim(dim: int, num_factors: int = 2) -> list:
    """
    将维度分解为 num_factors 个因子。
    例如：_factorize_dim(64, 2) -> [8, 8]。
    这是使用 TRLinear 的关键。
    """
    if dim < 1: return [1]
    
    # 尽可能分解为两个接近的因子，简化处理
    if num_factors == 2:
        i = int(round(math.sqrt(dim)))
        while dim % i != 0 and i > 1:
            i -= 1
        if i > 1:
            return [i, dim // i]
        else:
            return [dim, 1]
    
    return [dim] # 无法合理分解，返回原维度


def Attn_tem(heads=8, layers=1, channels=64, tr_ranks=None):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", tr_ranks=tr_ranks
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", tr_ranks: Union[list, np.ndarray] = None):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        if tr_ranks is not None and len(tr_ranks) >= 2:
            # Linear1: d_model -> dim_feedforward
            in_shape_1 = _factorize_dim(d_model)
            out_shape_1 = _factorize_dim(dim_feedforward)
            self.linear1 = TRLinear(in_shape_1, out_shape_1, tr_ranks[0])

            # Linear2: dim_feedforward -> d_model
            in_shape_2 = out_shape_1
            out_shape_2 = in_shape_1
            self.linear2 = TRLinear(in_shape_2, out_shape_2, tr_ranks[1])
        else:
            # 使用标准 Linear 层
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True, device=None, is_adp=True):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels
        self.support_len = 2
        self.is_adp = is_adp
        if is_adp:
            self.support_len += 1

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape
        if K == 1:
            return x
        if self.is_adp:
            nodevec1 = support_adp[-1][0]
            nodevec2 = support_adp[-1][1]
            support = support_adp[:-1]
        else:
            support = support_adp
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        if self.is_adp:
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            support = support + [adp]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out


class TemporalLearning(nn.Module):
    def __init__(self, channels, nheads, is_cross=True, tr_ranks=None):
        super().__init__()
        self.is_cross = is_cross
        # Attn_tem 内部的 TransformerEncoderLayer_QKV 会使用 tr_ranks
        self.time_layer = Attn_tem(heads=nheads, layers=1, channels=channels, tr_ranks=tr_ranks)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)
        if self.is_cross:
            itp_y = itp_y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


class SpatialLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, proj_t, is_cross, tr_ranks=None):
        super().__init__()
        self.is_cross = is_cross
        # tr_ranks 传入 SpaDependLearning
        self.feature_layer = SpaDependLearning(channels, nheads=nheads, order=order, target_dim=target_dim,
                                             include_self=include_self, device=device, is_adp=is_adp, 
                                             proj_t=proj_t, is_cross=is_cross, tr_ranks=tr_ranks)
        
    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, proj_t, is_cross=True, tr_ranks: Union[list, np.ndarray] = None):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp)
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
    # ------------------- 替换 nn.Linear 为 TRLinear -------------------
        if tr_ranks is not None and len(tr_ranks) >= 2:
            in_shape = _factorize_dim(channels)
            out_shape = _factorize_dim(channels * 2)
            
            # Linear1: channels -> channels * 2
            self.ff_linear1 = TRLinear(in_shape, out_shape, tr_ranks[0])
            
            # Linear2: channels * 2 -> channels
            self.ff_linear2 = TRLinear(out_shape, in_shape, tr_ranks[1])
        else:
            self.ff_linear1 = nn.Linear(channels, channels * 2)
            self.ff_linear2 = nn.Linear(channels * 2, channels)

        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)       # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        
        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_local + y_attn
        # Permute to (K*L, B, C) for TRLinear
        y_permuted = y_in2.permute(2, 0, 1).contiguous()
        y = F.relu(self.ff_linear1(y_permuted))
        y = self.ff_linear2(y)
        # Permute back to (B, C, K*L)
        y = y.permute(1, 2, 0).contiguous()
        y = y + y_in2

        y = self.norm2(y)
        return y


class GuidanceConstruct(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, proj_t, tr_ranks: Union[list, np.ndarray] = None):
        super().__init__()
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp)
        self.attn_s = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.attn_t = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn_s = nn.GroupNorm(4, channels)
        self.norm1_attn_t = nn.GroupNorm(4, channels)
        # ------------------- 替换 nn.Linear 为 TRLinear -------------------
        if tr_ranks is not None and len(tr_ranks) >= 2:
            in_shape = _factorize_dim(channels)
            out_shape = _factorize_dim(channels * 2)

            # Linear1: channels -> channels * 2
            self.ff_linear1 = TRLinear(in_shape, out_shape, tr_ranks[0])
            
            # Linear2: channels * 2 -> channels
            self.ff_linear2 = TRLinear(out_shape, in_shape, tr_ranks[1])
        else:
            self.ff_linear1 = nn.Linear(channels, channels * 2)
            self.ff_linear2 = nn.Linear(channels * 2, channels)
            
        self.norm2 = nn.GroupNorm(4, channels)


    def forward(self, y, base_shape, support):
        B, channel, K, L = base_shape
        y_in1 = y
        K_L = K * L # 定义 K*L，便于后续 Reshape
        
        y_local = self.GCN(y, base_shape, support)       # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)

        y_attn_s1 = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y_attn_s = self.attn_s(y_attn_s1.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn_s = y_attn_s.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn_s = y_in1 + y_attn_s
        y_attn_s = self.norm1_attn_s(y_attn_s)

        y_attn_t1 = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y_attn_t1.permute(2, 0, 1)
        y_attn_t = self.attn_t(v, v, v).permute(1, 2, 0)
        y_attn_t = y_attn_t.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        y_attn_t = y_in1 + y_attn_t
        y_attn_t = self.norm1_attn_t(y_attn_t)

        y_in2 = y_local + y_attn_s + y_attn_t
        # y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        # y = self.ff_linear2(y).permute(0, 2, 1)
        # y = y + y_in2
         # 1. 维度交换：[B, C, K*L] -> [B, K*L, C] (L是序列长度维度，C是特征维度)
        y_permuted = y_in2.permute(0, 2, 1)

        # 2. **关键修复：** 强制内存连续，并展平为 2D 矩阵 [N, C]
        #    N = B * K*L，C = channel
        y_2d = y_permuted.contiguous().reshape(-1, channel)

        # 3. FFN L1: [N, C] -> [N, 2C]
        y_out = F.relu(self.ff_linear1(y_2d))

        # 4. FFN L2: [N, 2C] -> [N, C]
        y_out = self.ff_linear2(y_out)

        # 5. 重塑回 3D 形状 [B, K*L, C]
        y_out = y_out.reshape(B, K_L, channel)

        # 6. 逆转维度：[B, K*L, C] -> [B, C, K*L] (用于残差连接和 GroupNorm)
        y_out = y_out.permute(0, 2, 1) 
        
        y = y_out + y_in2 # 残差连接

        y = self.norm2(y)
        return y


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
