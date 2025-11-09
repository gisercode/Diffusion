import os
import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yaml
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, mode, config):
        """
        自定义时间序列数据集，实现三层缺失机制。
        
        三层缺失：
        1. 原始数据缺失 (observed_mask): 数据本身的零值
        2. 预设缺失 (eval_mask & gt_mask): 在 __init__ 中创建的固定缺失模式
        3. 条件缺失 (cond_mask): 在 __getitem__ 中为训练样本动态创建的缺失
        
        Args:
            data_path (str): 预处理后的 .npz 文件路径。
            mode (str): 'train', 'val', or 'test'。
            config (dict): 全局配置文件。
        """
        self.mode = mode
        self.config = config
        
        # 加载预处理数据
        processed_data = np.load(data_path)
        raw_data = torch.from_numpy(processed_data[f'{mode}_data']).float()
        
        self.N, self.T = raw_data.shape
        
        # 读取滑动窗口参数
        self.L_input = config['dataloader']['L_input']
        self.L_pred = config['dataloader']['L_pred']
        self.L = self.L_input + self.L_pred
        if self.mode == 'train':
            self.stride = config['dataloader']['stride']
        else:
            if self.L_pred:
                self.stride = self.L_pred
            else:
                self.stride = self.L_input
        self.is_interpolate = config['model']['use_guide']
        
        # ============================================================
        # 第一层：原始数据缺失 (observed_mask)
        # ============================================================
        # observed_mask: 1 表示有观测值，0 表示原始数据中的缺失
        # self.observed_mask = (raw_data.detach().cpu().numpy() != 0.).astype('uint8')
        self.observed_mask = 1.0 - torch.from_numpy(processed_data[f'{mode}_mask']).float()
        
        # # 标准化处理：只对有观测值的位置进行标准化
        # mean = torch.from_numpy(processed_data['mean']).float()
        # std = torch.from_numpy(processed_data['std']).float()
        
        # # 扩展维度以匹配数据形状 (N, T)
        # if mean.dim() == 1:
        #     mean = mean.unsqueeze(1)  # (N, 1)
        #     std = std.unsqueeze(1)    # (N, 1)
        
        # 标准化并应用原始缺失掩码
        self.observed_data = raw_data * self.observed_mask
        
        # ============================================================
        # 第二层：预设缺失 (eval_mask & gt_mask)
        # ============================================================
        # eval_mask: 预设的缺失模式，1 表示被人为设置为缺失的位置
        # gt_mask: 1 表示既有观测值又不在预设缺失中的位置（可用于评估的真值）
        
        if mode in ['val', 'test']:
            # 为验证集和测试集生成固定的预设缺失模式
            preset_cfg = config['dataloader']['preset_masking']
            random_seed = config['global']['random_seed']
            self.eval_mask = self._generate_preset_mask(preset_cfg, random_seed) #1为预设缺失位置，0为非预设缺失
        else:
            # 训练集不需要预设缺失
            self.eval_mask = torch.zeros_like(self.observed_mask)
        
        # gt_mask = observed_mask AND eval_mask
        # 即：有观测值且预设缺失中的位置
        self.gt_mask = self.observed_mask * self.eval_mask # 1 表示需要评估的位置
        
        # 计算所有可能的样本起始点
        self.indices = self._generate_indices()

        # 如果配置中启用了保存数据集，则执行保存操作
        # if self.config.get('global', {}).get('save_datasets', False):
        #     # 1. 根据 preset_masking 配置创建文件夹名
        #     preset_cfg = self.config['dataloader']['preset_masking']
        #     folder_name_parts = []
        #     if preset_cfg.get('full_node', {}).get('enabled', False):
        #         folder_name_parts.append(f"fm_{preset_cfg['full_node']['ratio']}")
        #     if preset_cfg.get('block', {}).get('enabled', False):
        #         lr = preset_cfg['block']['length_range']
        #         folder_name_parts.append(f"bm_{preset_cfg['block']['target_missing_ratio']}")
        #     if preset_cfg.get('random', {}).get('enabled', False):
        #         folder_name_parts.append(f"rm_{preset_cfg['random']['ratio']}")
            
        #     if not folder_name_parts:
        #         folder_name = f"p_{self.L_input}_{self.L_pred}"
        #     else:
        #         folder_name = "_".join(folder_name_parts)

        #     # 2. 创建保存目录
        #     dataset_name = self.config['dataset']['name']
        #     save_dir = os.path.join('./saved_datasets', dataset_name, folder_name)
        #     os.makedirs(save_dir, exist_ok=True)

        #     # 3. 复制配置文件
        #     config_source_path = 'config.yaml'
        #     if os.path.exists(config_source_path):
        #         shutil.copy(config_source_path, os.path.join(save_dir, 'config.yaml'))

        #     # 4. 保存相关数据文件
        #     print(f"Saving processed dataset for mode '{self.mode}' to: {save_dir}")
        #     if self.mode == 'train':
        #         np.save(os.path.join(save_dir, 'train_data.npy'), self.observed_data.cpu().numpy())
        #         np.save(os.path.join(save_dir, 'train_mask.npy'), self.observed_mask.cpu().numpy()) #1 表示有观测值，0 表示原始数据中的缺失
        #     elif self.mode == 'val':
        #         np.save(os.path.join(save_dir, 'val_data.npy'), self.observed_data.cpu().numpy())
        #         np.save(os.path.join(save_dir, 'val_eval_mask.npy'), self.eval_mask.cpu().numpy()) # 1 表示预设缺失的位置
        #         np.save(os.path.join(save_dir, 'val_gt_mask.npy'), self.gt_mask.cpu().numpy()) # 1 表示有观测值+预设缺失的位置，参与模型评估的位置
        #     elif self.mode == 'test':
        #         np.save(os.path.join(save_dir, 'test_data.npy'), self.observed_data.cpu().numpy())
        #         np.save(os.path.join(save_dir, 'test_eval_mask.npy'), self.eval_mask.cpu().numpy())
        #         np.save(os.path.join(save_dir, 'test_gt_mask.npy'), self.gt_mask.cpu().numpy())

    def _generate_preset_mask(self, preset_cfg, random_seed):
        """
        生成预设缺失掩码 (第二层缺失)。
        这个缺失模式在整个验证/测试集中是固定的。
        
        Args:
            preset_cfg: config['dataloader']['preset_masking'] 配置
            random_seed (int): 用于设置随机种子的整数，确保可复现性。
            
        Returns:
            eval_mask: (N, T) 的 torch.Tensor，1 表示预设缺失位置
        """
        eval_mask = torch.zeros((self.N, self.T))
        
        # 设置随机种子以确保可重复性
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 策略1: 全节点掩码
        if preset_cfg['full_node']['enabled']:
            node_ratio = preset_cfg['full_node']['ratio']
            n_nodes_to_mask = int(self.N * node_ratio)
            if n_nodes_to_mask > 0:
                masked_node_indices = torch.randperm(self.N)[:n_nodes_to_mask]
                eval_mask[masked_node_indices, :] = 1
        
        # 策略2: 块状缺失
        if preset_cfg['block']['enabled']:
            min_len, max_len = preset_cfg['block']['length_range']
            target_ratio = preset_cfg['block']['target_missing_ratio']
            
            n_total_points = self.N * self.T
            n_block_masked_target = int(n_total_points * target_ratio)
            
            n_masked_count = 0
            max_attempts = n_total_points * 10  # 防止死循环
            attempts = 0
            
            while n_masked_count < n_block_masked_target and attempts < max_attempts:
                attempts += 1
                
                # 随机选择块长度
                block_len = torch.randint(min_len, max_len + 1, (1,)).item()
                
                # 找到可用的行（未被 full_node 完全掩盖）
                available_rows_mask = (eval_mask.sum(dim=1) < self.T)
                if not available_rows_mask.any():
                    break
                
                available_rows_indices = available_rows_mask.nonzero().squeeze()
                if available_rows_indices.dim() == 0:
                    available_rows_indices = available_rows_indices.unsqueeze(0)
                
                # 随机选择节点和起始时间
                node_idx = available_rows_indices[torch.randint(0, len(available_rows_indices), (1,))]
                start_t = torch.randint(0, self.T - block_len + 1, (1,)).item()
                
                # 检查是否重叠
                if eval_mask[node_idx, start_t : start_t + block_len].sum() == 0:
                    eval_mask[node_idx, start_t : start_t + block_len] = 1
                    n_masked_count += block_len
        
        # 策略3: 随机缺失
        if preset_cfg['random']['enabled']:
            random_ratio = preset_cfg['random']['ratio']
            n_random_masked_target = int(self.N * self.T * random_ratio)
            
            if n_random_masked_target > 0:
                available_indices = (eval_mask == 0).nonzero(as_tuple=False)
                
                if available_indices.shape[0] > 0:
                    n_to_mask = min(n_random_masked_target, available_indices.shape[0])
                    perm = torch.randperm(available_indices.shape[0])
                    selected_indices = available_indices[perm[:n_to_mask]]
                    eval_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
        
        return eval_mask

    def _generate_indices(self):
        """根据总时长、窗口长度和步长，生成所有样本的起始索引"""
        indices = []
        for i in range(0, self.T - self.L + 1, self.stride):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        获取单个样本并动态生成条件缺失掩码（第三层缺失）。

        Returns:
            dict: 包含单个样本数据的字典，键值对含义如下：
            - 'observed_data': (N, L) tensor, 标准化后的原始观测数据。
            - 'observed_mask': (N, L) tensor, 原始数据掩码 (1=观测到, 0=原始缺失)。
                                 代表数据的最底层、最真实的状态。
            - 'gt_mask':       (N, L) tensor, 真值掩码 (1=可用于评估的真值)。
                                 它排除了原始缺失和预设缺失的位置 (gt_mask = observed_mask AND eval_mask)。
                                 在 val/test 模式下作为损失计算的依据。
            - 'cond_mask':     (N, L) tensor, 条件掩码 (1=模型可见的条件，0=动态掩码掉的值)。
                                 在训练时动态生成，在 val/test 时等于 gt_mask (未来部分除外)。
                                 模型根据这部分信息来预测 target_mask 的位置。
            - 'target_mask':   (N, L) tensor, 目标掩码 (1=模型需要预测/插补的位置)。
                                 它与 cond_mask 互补，但只包含有实际观测值的位置 (target_mask = (1 - cond_mask) AND observed_mask)。
            - 'timepoints':    (L,) tensor, 当前样本的时间步索引 (0, 1, ..., L-1)。
            - 'coeffs':        (N, L, 2) tensor, (可选) CDE 模型的线性插值系数。
        """
        # 1. 根据索引切分窗口
        start_idx = self.indices[idx]
        end_idx = start_idx + self.L
        
        # 切分各层掩码和数据
        observed_data = self.observed_data[:, start_idx:end_idx]
        observed_mask = self.observed_mask[:, start_idx:end_idx]
        gt_mask = self.gt_mask[:, start_idx:end_idx]
        
        # ============================================================
        # 第三层：条件缺失 (cond_mask)
        # ============================================================
        # cond_mask: 1 表示模型可以看到的数据（条件），0 表示模型需要预测的数据
        
        if self.mode == 'train':
            # 训练模式：动态生成条件缺失
            cond_cfg = self.config['dataloader']['conditional_masking']
            cond_mask = self._generate_dynamic_mask(observed_mask, cond_cfg)
        else:
            # 验证/测试模式：条件掩码是观察到但非评估点的位置
            # cond_mask = observed_mask AND (NOT eval_mask)
            cond_mask = observed_mask - gt_mask
            
            if self.L_pred > 0:
                # 对于预测任务，条件不能包含预测窗口
                cond_mask[:, self.L_input:] = 0
        
        # ============================================================
        # 计算辅助信息
        # ============================================================
        
        # target_mask: 模型需要计算损失的位置
        target_mask = (1 - cond_mask) * observed_mask # 动态掩码掉的位置
        # target_mask = observed_mask # 全部可观测的位置
        
        # 返回数据字典
        data_dict = {
            'observed_data': observed_data,     
            'observed_mask': observed_mask,     
            'gt_mask': gt_mask,          
            'cond_mask': cond_mask,       
            'target_mask': target_mask, 
            'timepoints': torch.arange(self.L), 
        }
        if self.is_interpolate:
            # 计算 itp_data (线性插值系数，用于 CDE 模型)
            tmp_data = observed_data.to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            # torchcde 需要 (time, channel) 格式
            itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).cpu()).float()
            itp_data = itp_data.permute(1, 0)
            data_dict['coeffs'] = itp_data         # 插值信息
        return data_dict

    def _generate_dynamic_mask(self, base_mask, mask_cfg):
        """
        动态生成掩码（用于条件缺失或预设缺失）。
        
        Args:
            base_mask: 基础掩码，通常是 observed_mask
            mask_cfg: 掩码配置（来自 config）
            
        Returns:
            mask: (N, L) 的 torch.Tensor
        """
        N, L = base_mask.shape
        target_mask = torch.zeros_like(base_mask)
        
        # 任务1: 预测 (Prediction) - 未来时间步需要预测
        if self.L_pred > 0:
            target_mask[:, self.L_input:] = 1
        
        # 任务2: 插值与重构 - 在历史窗口内进行
        hist_mask_view = target_mask[:, :self.L_input]
        
        # 策略1: 全节点掩码 (最高优先级)
        if mask_cfg['full_node']['enabled']:
            node_ratio = mask_cfg['full_node']['ratio']
            n_nodes_to_mask = int(N * node_ratio)
            if n_nodes_to_mask > 0:
                masked_node_indices = torch.randperm(N)[:n_nodes_to_mask]
                hist_mask_view[masked_node_indices, :] = 1
        
        # 策略2: 块状缺失 (中等优先级)
        if mask_cfg['block']['enabled']:
            min_len, max_len = mask_cfg['block']['length_range']
            target_ratio = mask_cfg['block']['target_missing_ratio']
            
            n_total_hist_points = N * self.L_input
            n_block_masked_target = int(n_total_hist_points * target_ratio)
            
            n_masked_count = 0
            max_attempts = n_total_hist_points * 10
            attempts = 0
            
            while n_masked_count < n_block_masked_target and attempts < max_attempts:
                attempts += 1
                
                block_len = torch.randint(min_len, max_len + 1, (1,)).item()
                
                available_rows_mask = (hist_mask_view.sum(dim=1) == 0)
                if not available_rows_mask.any():
                    break
                
                available_rows_indices = available_rows_mask.nonzero().squeeze()
                if available_rows_indices.dim() == 0:
                    available_rows_indices = available_rows_indices.unsqueeze(0)
                
                node_idx = available_rows_indices[torch.randint(0, len(available_rows_indices), (1,))]
                start_t = torch.randint(0, self.L_input - block_len + 1, (1,)).item()
                
                if hist_mask_view[node_idx, start_t : start_t + block_len].sum() == 0:
                    hist_mask_view[node_idx, start_t : start_t + block_len] = 1
                    n_masked_count += block_len
        
        # 策略3: 随机缺失 (最低优先级)
        if mask_cfg['random']['enabled']:
            random_ratio = mask_cfg['random']['ratio']
            n_random_masked_target = int(N * self.L_input * random_ratio)
            
            if n_random_masked_target > 0:
                available_indices = (hist_mask_view == 0).nonzero(as_tuple=False)
                
                if available_indices.shape[0] > 0:
                    n_to_mask = min(n_random_masked_target, available_indices.shape[0])
                    perm = torch.randperm(available_indices.shape[0])
                    selected_indices = available_indices[perm[:n_to_mask]]
                    hist_mask_view[selected_indices[:, 0], selected_indices[:, 1]] = 1
        
        # 返回的掩码：1=需要预测的位置，0=可以看到的位置
        # 因此需要取反才能得到 cond_mask
        return 1 - target_mask


def get_dataloader(config, mode, num_workers=None):
    """
    获取指定模式的 DataLoader。
    """
    dataset_name = config['dataset']['name']
    output_dir = config['paths']['output_dir']
    data_path = os.path.join(output_dir, f'{dataset_name}_processed.npz')
    
    dataset = TimeSeriesDataset(data_path, mode, config)
    
    shuffle = True if mode == 'train' else False
    
    if mode in ['val', 'test']:
        effective_num_workers = 0
    else:
        effective_num_workers = config['global']['num_workers'] if num_workers is None else num_workers
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=shuffle,
        num_workers=effective_num_workers
    )
    
    return dataloader


if __name__ == '__main__':
    # ================== 测试代码 ==================
    # 解决 OMP: Error #15, 允许多个 OpenMP 运行时库存在
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    config = load_config('config.yaml')
    
    print("=" * 80)
    print("测试三层缺失机制的 DataLoader")
    print("=" * 80)
    
    for mode in ['train', 'val', 'test']:
        print(f"\n{'='*80}")
        print(f"测试 {mode.upper()} 模式")
        print(f"{'='*80}")
        
        loader = get_dataloader(config, mode)
        batch = next(iter(loader))
        
        print("\n批次数据维度:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:20s}: {tuple(value.shape)}")
        
        # 验证掩码关系
        print("\n掩码关系验证:")
        observed_mask = batch['observed_mask']
        gt_mask = batch['gt_mask']
        cond_mask = batch['cond_mask']
        target_mask = batch['target_mask']
        observed_data = batch['observed_data']
        
        # 1. gt_mask 应该 <= observed_mask
        check1 = torch.all(gt_mask <= observed_mask)
        print(f"  ✓ gt_mask ≤ observed_mask: {check1}")
        
        # 2. cond_mask 和 target_mask 应该互补
        # check2 = torch.all((cond_mask + target_mask) == 1)
        # print(f"  ✓ cond_mask + target_mask = 1: {check2}")
        check2 = torch.all((cond_mask + target_mask)[observed_mask == 1] == 1)
        print(f"   ✓ (cond + target)[observed=1] = 1: {check2}")
        
        # 3. observed_data 在 observed_mask=0 的位置应该为 0
        check3 = torch.all(observed_data[observed_mask == 0] == 0)
        print(f"  ✓ observed_data[observed_mask=0] = 0: {check3}")
        
        # 4. 训练模式下，cond_mask 应该与 gt_mask 不同（因为动态生成）
        if mode == 'train':
            check4 = not torch.all(cond_mask == gt_mask)
            print(f"  ✓ cond_mask ≠ gt_mask (动态生成): {check4}")
        else:
            check4 = torch.all(cond_mask == gt_mask)
            print(f"  ✓ cond_mask = gt_mask (使用真值): {check4}")
        
        # 统计信息
        print(f"\n掩码统计:")
        B, N, L = observed_mask.shape
        total_points = B * N * L
        print(f"  总数据点: {total_points}")
        print(f"  observed_mask=1: {observed_mask.sum().item()} ({observed_mask.sum().item()/total_points*100:.2f}%)")
        print(f"  gt_mask=1: {gt_mask.sum().item()} ({gt_mask.sum().item()/total_points*100:.2f}%)")
        print(f"  cond_mask=1: {cond_mask.sum().item()} ({cond_mask.sum().item()/total_points*100:.2f}%)")
        print(f"  target_mask=1: {target_mask.sum().item()} ({target_mask.sum().item()/total_points*100:.2f}%)")
    
    print("\n" + "="*80)
    print("DataLoader 测试完成！")
    print("="*80)


    # ================== 可视化代码 ==================
    def visualize_mask(mask, title, ax):
        """
        使用 seaborn.heatmap 可视化掩码 (0/1)。
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        sns.heatmap(mask, cmap='viridis', cbar=False, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Nodes")

    def visualize_heatmap_with_missing(data, mask, title, ax):
        """
        将缺失值显示为黑色的热力图。
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # 将掩码为0的位置设置为NaN
        plot_data = np.where(mask == 1, data, np.nan)
        
        sns.heatmap(plot_data, cmap='viridis', cbar=True, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Nodes")

    print("\n" + "="*80)
    print("开始可视化掩码")
    print("="*80)

    # --- 辅助函数：用于生成特定类型的预设缺失 ---
    def generate_specific_mask(dataset, strategy):
        """为测试集生成特定策略的缺失掩码"""
        import copy
        temp_cfg = copy.deepcopy(config['dataloader']['preset_masking'])
        
        for key, value in temp_cfg.items():
            if not isinstance(value, dict):
                continue
            
            if key != strategy:
                value['enabled'] = False
            else:
                value['enabled'] = True
        
        return dataset._generate_preset_mask(
            temp_cfg,
            config['global']['random_seed']
        )

    # --- 加载所有数据集 ---
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    test_loader = get_dataloader(config, 'test')
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    # # 获取一个训练批次以展示动态缺失
    # train_batch = next(iter(train_loader))

    # # --- 开始绘图 --
    # # 设置全局字体为 Times New Roman
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.serif": ["Times New Roman"],
    #     "mathtext.fontset": "stix"  # 保证数学公式字体协调（可选）
    # })

    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle('Mask Visualization Comprehensive Overview', fontsize=20)

    # # ==================================================
    # # 第一行: 数据集整体的原始缺失
    # # ==================================================
    # visualize_heatmap_with_missing(train_dataset.observed_data,
    #                                train_dataset.observed_mask,
    #                                'Full Train Set: Data Heatmap (Missing=Black)',
    #                                axes[0, 0])
    # visualize_heatmap_with_missing(val_dataset.observed_data,
    #                                val_dataset.observed_mask,
    #                                'Full Validation Set: Data Heatmap (Missing=Black)',
    #                                axes[0, 1])
    # visualize_heatmap_with_missing(test_dataset.observed_data,
    #                                test_dataset.observed_mask,
    #                                'Full Test Set: Data Heatmap (Missing=Black)',
    #                                axes[0, 2])

    # ==================================================
    # 第二行: 单个训练样本的动态缺失过程
    # ==================================================
    # sample_idx = 0
    # visualize_mask(train_batch['observed_mask'][sample_idx],
    #                f'Train Sample {sample_idx}: Original Missing',
    #                axes[2, 0])
    # visualize_mask(train_batch['cond_mask'][sample_idx],
    #                f'Train Sample {sample_idx}: Condition (Model Input)',
    #                axes[2, 1])
    # visualize_mask(train_batch['target_mask'][sample_idx],
    #                f'Train Sample {sample_idx}: Target (To Predict)',
    #                axes[2, 2])


    # ==================================================
    # 第三行: 测试集中不同策略缺失后的可视化图
    # ==================================================
    # full_node_mask = generate_specific_mask(test_dataset, 'full_node')
    # visualize_mask(full_node_mask,
    #                'Test Preset Strategy: Full Node',
    #                axes[1, 0])

    # block_mask = generate_specific_mask(test_dataset, 'block')
    # visualize_mask(block_mask,
    #                'Test Preset Strategy: Block',
    #                axes[1, 1])

    # random_mask = generate_specific_mask(test_dataset, 'random')
    # visualize_mask(random_mask,
    #                'Test Preset Strategy: Random',
    #                axes[1, 2])

    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # plt.show()

    # ==================================================
    # 第三行: 测试集不同策略的预设缺失
    # ==================================================
    # full_node_mask = generate_specific_mask(test_dataset, 'full_node')
    # visualize_mask(full_node_mask,
    #                'Test Preset Strategy: Full Node',
    #                axes[1, 0])

    # block_mask = generate_specific_mask(test_dataset, 'block')
    # visualize_mask(block_mask,
    #                'Test Preset Strategy: Block',
    #                axes[1, 1])

    # random_mask = generate_specific_mask(test_dataset, 'random')
    # visualize_mask(random_mask,
    #                'Test Preset Strategy: Random',
    #                axes[1, 2])

    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # plt.show()