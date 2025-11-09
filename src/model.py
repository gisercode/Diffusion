import numpy as np
import torch
import torch.nn as nn
import torchcde
from tqdm import tqdm
# 假设 diff_models.py 存在于同一目录下或可通过 PYTHONPATH 访问
from src.diff_models import Guide_diff


class DiffuSDH(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.config = config # 保存 config 为实例属性
        self.device = device
        self.target_dim = target_dim # N (节点数)
        self.seq_len = seq_len       # L (序列长度)

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        config_diff["name"] = config["dataset"]["name"]
    
        self.device = device

        input_dim = 2 # 扩散模型的输入维度，通常是 (noisy_data, cond_obs)
        self.diffmodel = Guide_diff(config_diff, input_dim, target_dim, self.use_guide)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        else:
            raise ValueError(f"未知的调度策略: {config_diff['schedule']}")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        """
        生成时间位置编码。
        pos: (B, L)
        返回: (B, L, d_model)
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2) # (B, L, 1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        ) # (d_model/2,)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        """
        生成 side_info，包含时间嵌入和特征嵌入。
        observed_tp: (B, L)
        cond_mask: (B, L, K)
        返回: (B, emb_total_dim, K, L)
        """
        # 获取批次大小 B
        B = cond_mask.shape[0]
        # N 是节点数 (self.target_dim), L 是序列长度 (self.seq_len)

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B, L_seq, emb_time_dim)
        # 扩展 time_embed 以匹配节点维度 (N)
        time_embed = time_embed.unsqueeze(2).expand(B, self.seq_len, self.target_dim, -1) # (B, L_seq, N, emb_time_dim)
        
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (N, emb_feature_dim)
        # 扩展 feature_embed 以匹配批次维度 (B) 和序列长度维度 (L_seq)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, self.seq_len, self.target_dim, -1) # (B, L_seq, N, emb_feature_dim)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B, L_seq, N, emb_total_dim)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb_total_dim, N, L_seq)

        return side_info

    #这里是为了提高效率修改的
    # def calc_loss_valid(
    #     self, observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train
    # ):
    #     loss_sum = 0
    #     for t in range(self.num_steps):  # calculate loss for all t
    #         loss = self.calc_loss(
    #             observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train, set_t=t
    #         )
    #         loss_sum += loss.detach()
    #     return loss_sum / self.num_steps
    

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train
    ):
        # 验证时不再遍历所有t，而是随机选择一个t来计算损失，以大幅提升速度
        # 这种单样本蒙特卡洛估计在实践中足够准确且高效
        B, K, L = observed_data.shape
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        
        loss = self.calc_loss(
            observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train, set_t=t
        )
        return loss
    
    

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = set_t
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        if not self.use_guide:
            itp_info = cond_mask * observed_data
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)

        # target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                # 错误修复：确保时间步 t 的形状与批次大小 B 匹配
                # 之前使用 torch.tensor([t]) 会创建一个单元素张量，可能导致不正确的广播和数值问题
                # 现在我们创建一个形状为 [B] 的张量，其中所有元素都是 t
                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
                predicted = self.diffmodel(diff_input, side_info, t_tensor, itp_info, cond_mask)
                #predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), itp_info, cond_mask) 20251109

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        # 从 DataLoader 获取数据
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            coeffs,
            cond_mask, 
            target_mask
        ) = self.process_data(batch)

        # 2. 准备 side_info
        side_info = self.get_side_info(observed_tp, cond_mask)
        
        # 3. 准备 itp_info
        itp_info = None
        if self.use_guide:
            itp_info = coeffs.unsqueeze(1)

        # 4. 计算损失
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(observed_data, cond_mask, observed_mask, target_mask, side_info, itp_info, is_train)
        return output
    
    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()
        target_mask = batch["target_mask"].to(self.device).float()

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            coeffs,
            cond_mask,
            target_mask
        )

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            coeffs,
            cond_mask,
            target_mask
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = 1.0 - gt_mask
            #target_mask = observed_mask - gt_mask

            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = coeffs.unsqueeze(1)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)

        return samples, observed_data, gt_mask, observed_mask, observed_tp

