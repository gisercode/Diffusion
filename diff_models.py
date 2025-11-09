import os
import pickle
import numpy as np
from layers import *
from gwnet import GWNet
from generate_adj import *
from typing import Union


class Guide_diff(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.layer = config["layers"]
        self.is_itp = is_itp
        self.ranks_for_ffn = [[config["rank"], config["rank"], config["rank"], config["rank"]],
                                [config["rank"], config["rank"], config["rank"], config["rank"]]
                                                                                    ]
        total_tr_ffn_layers = (1 if self.is_itp else 0) + self.layer
        tr_linear_ranks_config = [self.ranks_for_ffn] * total_tr_ffn_layers

        self.tr_linear_ranks = tr_linear_ranks_config
        
        self.tr_idx = 0 # 用于跟踪当前使用的 ranks 索引
        self.itp_channels = None

        # 提前加载邻接矩阵
        dataset_name = config["name"]
        if dataset_name == 'pm25':
            self.adj = get_adj_AQI36()
        elif dataset_name == 'metr_la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif dataset_name == 'pems_bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        elif dataset_name == 'pre':
            # 为 'pre' 数据集加载邻接矩阵
            adj_path = os.path.join('./data', dataset_name, 'adj_pre.pkl')
            try:
                with open(adj_path, 'rb') as f:
                    # pkl 文件可能包含多个对象，具体取决于其保存方式
                    # 假设它直接是邻接矩阵或在一个元组/列表/字典中
                    adj_data = pickle.load(f)
                    if isinstance(adj_data, (list, tuple)) and len(adj_data) > 2:
                        self.adj = adj_data[2] # 根据 pkl 文件结构调整索引
                    else:
                        self.adj = adj_data
                # 确保邻接矩阵是 numpy array
                if not isinstance(self.adj, np.ndarray):
                    self.adj = np.array(self.adj)
            except FileNotFoundError:
                raise FileNotFoundError(f"邻接矩阵文件未找到: {adj_path}")
            except Exception as e:
                raise RuntimeError(f"加载或处理邻接矩阵失败: {e}")
        else:
            raise ValueError(f"未知的数据集名称: {dataset_name}")

        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            # 取出用于 GuidanceConstruct FFN 的 ranks
            gc_ranks = self.tr_linear_ranks[self.tr_idx] if self.tr_idx < len(self.tr_linear_ranks) else None
            self.tr_idx += 1

            self.itp_modeling = GWNet(
                device=config["device"],
                num_nodes=self.adj.shape[0],
                in_dim=self.itp_channels,
                out_dim=self.itp_channels,
                residual_channels=self.itp_channels,
                dilation_channels=self.itp_channels,
                skip_channels=self.itp_channels * 4,
                end_channels=self.itp_channels * 8,
                supports=compute_support_gwn(self.adj, device=config["device"]),
                addaptadj=config["is_adp"],
                aptinit=self.adj,
                tr_ranks=config["tr_ranks"]
            )
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.device = config["device"]
        self.support = compute_support_gwn(self.adj, device=config["device"])
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
            self.support.append((self.nodevec1, self.nodevec2))
        print("Support in diff_models.py:")
        for i, s in enumerate(self.support):
            if isinstance(s, tuple):
                print(f"  Support {i} (tuple): {[t.shape for t in s]}")
            else:
                print(f"  Support {i} shape: {s.shape}")

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        if self.tr_idx > len(self.tr_linear_ranks):
             print(f"Warning: tr_linear_ranks length ({len(self.tr_linear_ranks)}) is less than required ({self.tr_idx}). Some layers used standard Linear.")


    def train(self, mode=True):
        """
        重载 train 方法，确保子模型的状态与父模型同步。
        """
        super(Guide_diff, self).train(mode)
        if self.is_itp:
            self.itp_modeling.train(mode)
        return self

    def eval(self):
        """
        重载 eval 方法，确保子模型的状态与父模型同步。
        """
        super(Guide_diff, self).eval()
        if self.is_itp:
            self.itp_modeling.eval()
        return self

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
     if self.is_itp:
         x = torch.cat([x, itp_x], dim=1)
     B, inputdim, K, L = x.shape

     x = x.reshape(B, inputdim, K * L)
     x = self.input_projection(x)
     x = F.silu(x)
     x = x.reshape(B, self.channels, K, L)

     if self.is_itp:
         itp_x = itp_x.reshape(B, inputdim-1, K * L)
         itp_x = self.itp_projection(itp_x)
         itp_cond_info = side_info.reshape(B, -1, K * L)
         itp_cond_info = self.cond_projection(itp_cond_info)
         itp_x = itp_x + itp_cond_info
         itp_x = itp_x.reshape(B, self.itp_channels, K, L)
         itp_x = self.itp_modeling(itp_x)
         # itp_x = F.silu(itp_x) # 移除 SiLU 激活，让引导信号直接传递

     diffusion_emb = self.diffusion_embedding(diffusion_step)

     skip = []
     for i in range(len(self.residual_layers)):
         x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
         skip.append(skip_connection)

     x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
     x = x.reshape(B, self.channels, K * L)
     x = self.output_projection1(x)  # (B,channel,K*L)
     x = F.silu(x)
     x = self.output_projection2(x)  # (B,1,K*L)
     x = x.reshape(B, K, L)
     return x


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, is_cross_t=False, is_cross_s=True, tr_linear_ranks: Union[list, np.ndarray] = None):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # TemporalLearning 内部的 TransformerEncoderLayer_QKV FFN 使用 tr_ranks
        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t, tr_ranks=tr_linear_ranks)
        
        # SpatialLearning/SpaDependLearning 内部的 FFN 使用 tr_ranks
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               proj_t=proj_t, is_cross=is_cross_s, tr_ranks=tr_linear_ranks) # 传入 ranks
        


    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, itp_info)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

