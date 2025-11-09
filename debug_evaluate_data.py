import torch
import yaml
import numpy as np
import os
from model import DiffuSDH
from dataloader import get_dataloader

def debug_data_pipeline():
    """
    执行一次评估数据管道，并保存一个批次的数据用于调试。
    """
    # --- 1. 加载配置 ---
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 '{config_path}' 不存在。请确保路径正确。")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 3. 加载模型 ---
    model_path = r"D:\lxy\diffusion baseline\Diffusion\models\metr_la\best_model_20251109164123.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在。请检查 config.yaml 中的路径。")
        return

    # 从 config 中获取 target_dim 和 seq_len
    # 从 config 中获取 target_dim 和 seq_len
    # HACK: 调试时硬编码 target_dim，因为它不在 config 文件中
    dataset_name = config['dataset']['name']
    if dataset_name == 'metr_la':
        target_dim = 207
    elif dataset_name == 'pm25':
        target_dim = 36
    else:
        # 如果有其他数据集，可以在这里添加，或者抛出错误
        raise ValueError(f"未知的 target_dim for dataset: {dataset_name}")
    seq_len = config['dataloader']['L_input']
    
    model = DiffuSDH(target_dim, seq_len, config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已从 '{model_path}' 加载。")

    # --- 4. 获取测试数据加载器 ---
    # 修正：get_dataloader 不接受 device 参数，也不返回 scaler 和 mean_scaler
    test_loader = get_dataloader(
        config=config,
        mode='test' # 需要明确指定 mode
    )
    print("测试数据加载器已创建。")

    # --- 5. 获取一个批次的数据并执行模型评估 ---
    try:
        test_batch = next(iter(test_loader))
        print("成功从 test_loader 获取一个批次的数据。")
    except StopIteration:
        print("错误: test_loader 为空，无法获取数据。")
        return

    with torch.no_grad():
        output = model.evaluate(test_batch, n_samples=20) # 使用 n_samples=20 获取一些样本
        samples, c_target, eval_points, _, _ = output
    
    print("模型评估方法已执行。")

    # --- 6. 保存数据到文件 ---
    save_path = 'debug_data.npz'
    np.savez_compressed(
        save_path,
        c_target=c_target.cpu().numpy(),
        samples=samples.cpu().numpy(),
        eval_points=eval_points.cpu().numpy()
    )
    print(f"调试数据已成功保存到: {save_path}")
    print("\n现在可以检查 'debug_data.npz' 文件来分析输入和输出数据。")

if __name__ == "__main__":
    debug_data_pipeline()