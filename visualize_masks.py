import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import get_dataloader

# 解决 OMP: Error #15, 允许多个 OpenMP 运行时库存在
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def visualize_mask(mask, title, ax):
    """使用 seaborn.heatmap 可视化掩码 (0/1)。"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 使用 'gray' colormap，0为黑，1为白
    sns.heatmap(mask, cmap='gray', cbar=False, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Nodes")

def visualize_heatmap_with_missing(data, mask, title, ax):
    """将缺失值（mask=0）显示为黑色的热力图。"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # 将掩码为0的位置设置为NaN，这样heatmap会将其留空（或根据bad_color设置颜色）
    plot_data = np.where(mask == 1, data, np.nan)
    
    # 获取 viridis colormap
    cmap = plt.get_cmap('viridis').copy()
    # 将 'bad' 值的颜色（用于 NaN）设置为黑色
    cmap.set_bad(color='black')
    
    # 使用修改后的 colormap
    sns.heatmap(plot_data, cmap=cmap, cbar=True, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Nodes")

def main():
    """主函数，执行可视化"""
    config = load_config()
    
    print("正在加载测试数据集...")
    test_loader = get_dataloader(config, 'test')
    test_dataset = test_loader.dataset
    print("测试数据集加载完毕。")

    # --- 准备数据和掩码 ---
    # 1. 原始观测数据和其自身的缺失掩码
    original_data = test_dataset.observed_data
    original_mask = test_dataset.observed_mask

    # 2. 在测试集上生成的预设评估掩码 (eval_mask)
    # 这个掩码标记了哪些点要被人为地“挖掉”用于评估
    # 值为1表示要被挖掉
    preset_eval_mask = test_dataset.eval_mask

    # 3. 最终用于显示的掩码
    # 只有在 original_mask 中为1 (有数据) 且在 preset_eval_mask 中为0 (不被挖掉) 的点才保留
    final_mask_to_show = original_mask * (1 - preset_eval_mask)

    # --- 开始绘图 ---
    print("开始生成可视化图像...")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix"
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Visualization of Preset Masking Strategy on Test Set', fontsize=16, y=1.02)

    # 图1: 原始测试数据
    visualize_heatmap_with_missing(original_data, 
                                   original_mask, 
                                   '(A) Original Test Data\n(Black = Natively Missing)', 
                                   axes[0])

    # 图2: 预设评估掩码
    visualize_mask(preset_eval_mask, 
                   '(B) Preset Evaluation Mask\n(White = Points to be Masked Out)', 
                   axes[1])

    # 图3: 应用掩码后的数据
    visualize_heatmap_with_missing(original_data, 
                                   final_mask_to_show, 
                                   '(C) Test Data After Masking\n(Black = Natively Missing or Masked Out)', 
                                   axes[2])

    plt.tight_layout()
    plt.savefig("test_set_masking_visualization.png", dpi=300, bbox_inches='tight')
    print("\n可视化图像已保存为 'test_set_masking_visualization.png'")
    plt.show()


if __name__ == '__main__':
    main()