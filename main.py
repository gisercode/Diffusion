import os
import sys
import yaml
import argparse
import torch
import numpy as np
import pickle
import time
import glob
import re
from tqdm import tqdm

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataloader import get_dataloader
from model import DiffuSDH
from evaluate import evaluate

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config, evaluate_only=False, quick_eval=False):
    """
    执行一次完整的模型训练、验证和评估流程，并返回评估结果。
    """
    print("======================================================")
    print("             DiffuSDH 模型实验")
    print("======================================================")

    # 1. 检查并创建模型保存目录
    dataset_name = config['dataset']['name']
    model_save_dir = os.path.join('models', dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)

    run_timestamp = time.strftime('%Y%m%d%H%M%S')
    print(f"本次运行ID (时间戳): {run_timestamp}")
    model_filename = f'best_model_{run_timestamp}.pth'
    model_save_path = os.path.join(model_save_dir, model_filename)
    
    print(f"\n模型将保存至: {model_save_path}")

    # 结果保存路径
    results_save_dir = os.path.join('results', dataset_name)
    os.makedirs(results_save_dir, exist_ok=True)
    results_filename = f"{dataset_name}_{run_timestamp}.npz"
    results_save_path = os.path.join(results_save_dir, results_filename)

    # 2. 加载数据和 scaler
    output_dir = config['paths']['output_dir']
    processed_data_path = os.path.join(output_dir, f'{dataset_name}_processed.npz')
    scaler_path = os.path.join(config['paths']['raw_data_dir'], dataset_name, f'{dataset_name.split("_")[0]}_meanstd.pk')

    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"预处理数据文件未找到: {processed_data_path}")

    try:
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            mean_scaler = scaler_data[0]
            scaler = scaler_data[1]
        print(f"成功加载 Scaler: {scaler_path}")
    except FileNotFoundError:
        print(f"[警告] Scaler 文件未找到: {scaler_path}。评估指标将基于归一化数据。")
        scaler, mean_scaler = 1.0, 0.0

    # 3. 创建 DataLoader
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    test_loader = get_dataloader(config, 'test')

    # 4. 实例化模型
    first_batch = next(iter(train_loader))
    _, N, L = first_batch['observed_data'].shape
    device = torch.device(config["global"]["device"] if torch.cuda.is_available() else "cpu")
    model = DiffuSDH(target_dim=N, seq_len=L, config=config, device=device).to(device)
    print(f"使用设备: {device}")

    if not evaluate_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        epochs = config['train']['epochs']
        best_val_loss = float('inf')

        print(f"\n开始训练，共 {epochs} 个 epochs...")
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [训练]", ncols=100, unit="batch") as pbar:
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = model(batch, is_train=1)
                    loss.backward()
                    # --- 解决梯度爆炸问题 ---
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    pbar.update(1)
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    val_loss = model(batch, is_train=0)
                    total_val_loss += val_loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | 平均训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> 验证损失提升，已保存最佳模型至 {model_save_path}")
        
        print("\n训练完成！")

    else:
        print("\n跳过训练过程。")
        # 在仅评估模式下，需要一个预训练模型路径
        # 为简化调优脚本，我们假设每次都从头训练
        print("[警告] 仅评估模式需要预训练模型，但调优流程会从头训练。")


    # 7. 最终评估
    print("\n开始在测试集上进行最终评估...")

    # 在 evaluate_only 模式下，查找最新的模型
    if evaluate_only:
        model_dir = os.path.join('models', dataset_name)
        list_of_files = glob.glob(os.path.join(model_dir, 'best_model_*.pth'))
        if not list_of_files:
            print(f"[错误] 在 {model_dir} 中没有找到任何模型文件。")
            return -1, -1, -1
        
        # 解析文件名中的时间戳并找到最新的文件
        latest_file = max(list_of_files, key=lambda f: re.search(r'best_model_(\d+).pth', f).group(1))
        model_load_path = latest_file
        print(f"检测到 --evaluate_only 模式，将加载最新的模型: {model_load_path}")
    else:
        # 在训练模式下，加载本次训练保存的最佳模型
        model_load_path = model_save_path
        if not os.path.exists(model_load_path):
            print(f"[错误] 模型文件 {model_load_path} 不存在。无法进行评估。")
            return -1, -1, -1
        print(f"加载本次训练的最佳模型: {model_load_path}")

    model.load_state_dict(torch.load(model_load_path, map_location=device))
    
    eval_config = config.get('evaluation', {})
    probabilistic_eval = eval_config.get('probabilistic_eval', False)
    nsample = eval_config.get('nsample', 1)

    if quick_eval:
        print("\n[提示] 已启用快速评估模式，将覆盖配置文件的评估设置。")
        probabilistic_eval = False
        nsample = 1

    # 在调优脚本中，我们不保存评估结果文件
    # evaluate函数现在可能返回3个或5个值，为安全起见，进行切片处理
    # 根据配置决定是否保存结果
    should_save_results = config.get('results', {}).get('save_final_results', False)
    final_results_save_path = results_save_path if should_save_results else None

    eval_results = evaluate(
        model,
        test_loader,
        device=device,
        probabilistic_eval=probabilistic_eval,
        nsample=nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        results_save_path=final_results_save_path,
        dataset_name=dataset_name
    )
    final_rmse, final_mae, final_mse = eval_results[:3]
    
    print("\n评估完成！")
    
    # 仅在非仅评估模式下删除本次训练的临时模型
    if not evaluate_only and os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"已删除临时模型: {model_save_path}")

    return final_rmse, final_mae, final_mse


def main():
    """主函数，封装实验流程"""
    parser = argparse.ArgumentParser(description="DiffuSDH 模型训练与评估")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--evaluate_only',
        action='store_true',
        help='跳过训练，直接在测试集上评估现有模型 (在调优脚本中不推荐使用)'
    )
    parser.add_argument(
        '--quick_eval',
        action='store_true',
        help='启用快速评估模式，强制 nsample=1 并禁用概率性评估，覆盖配置文件中的设置。'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 运行实验
    rmse, mae, mse = run_experiment(config, args.evaluate_only, args.quick_eval)

    # 为 hyperparam_tuning.py 提供可捕获的输出
    print("\n--- 实验结果 ---")
    print(f"Final RMSE: {rmse}")
    print(f"Final MAE: {mae}")
    print(f"Final MSE: {mse}")

if __name__ == "__main__":
    main()