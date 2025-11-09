import os
import sys
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 假设你的模块在 src/ 目录下
from src.dataloader import get_dataloader
from src.model import DiffuSDH


def inverse_transform(data, mean, std):
    """
    反标准化函数 (TENSOR version)
    假设 data 是 (..., N, L) 维的张量，mean/std 是 (N,) 维的张量。
    """
    # 将 mean 和 std 移动到 data 所在的设备
    mean = mean.to(data.device)
    std = std.to(data.device)
    
    # 创建 (1, ..., 1, N, 1) 的形状用于广播
    shape = [1] * data.dim()
    shape[-2] = -1 # 对应 N (features) 维度
    
    mean_expanded = mean.view(shape)
    std_expanded = std.view(shape)
    
    return data * std_expanded + mean_expanded


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(config, mode='full', model_path=None):
    """
    执行一次完整的模型训练、验证和评估流程。

    :param config: 配置字典
    :param mode: 运行模式 ('train', 'test', 'full')
    :param model_path: 'test' 模式下要加载的模型路径
    """
    print("======================================================")
    print(f"             DiffuSDH 模型实验 (模式: {mode.upper()})")
    print("======================================================")

    # --- 1. 设备和 Dataloader 设置 ---
    device = torch.device(config["global"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 在 'train' 或 'full' 模式下，需要 train/val loader
    if mode in ['train', 'full']:
        train_loader, _, _ = get_dataloader(config, 'train')
        val_loader, _, _ = get_dataloader(config, 'val')
        # 从 train_loader 获取 N, L
        first_batch = next(iter(train_loader))
        _, N, L = first_batch['observed_data'].shape
    
    # 总是需要 test_loader 来获取 mean/std
    # 在 'test' 模式下，也用它来获取 N, L
    test_loader, mean_tensor, std_tensor = get_dataloader(config, 'test')
    
    if mode == 'test':
        # 如果是 'test' 模式，从 test_loader 获取 N, L
        first_batch = next(iter(test_loader))
        _, N, L = first_batch['observed_data'].shape

    # --- 2. 实例化模型 ---
    model = DiffuSDH(target_dim=N, seq_len=L, config=config, device=device).to(device)

    # --- 3. 模式特定逻辑 ---

    # =========================
    # 模式: 仅测试 (TEST)
    # =========================
    if mode == 'test':
        print(f"\n正在从 {model_path} 加载模型...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功。")
        print("开始在测试集上评估 (不记录 TensorBoard)...")
        
        model.eval()
        all_rmse, all_mse, all_mae = [], [], []
        
        with tqdm(total=len(test_loader), desc="[测试]", ncols=100, unit="batch") as pbar:
            with torch.no_grad():
                for batch_no, test_batch in enumerate(test_loader, start=1):
                    output = model.evaluate(test_batch, config['evaluation']['nsample'])
                    
                    # output 是 Tensors
                    samples, observed_data, eval_points, _, _ = output
                    
                    # 反标准化 (使用 Tensors)
                    samples_original_tensor = inverse_transform(samples, mean_tensor, std_tensor)
                    observed_original_tensor = inverse_transform(observed_data, mean_tensor, std_tensor)
                    
                    # 转换为 Numpy 用于评估
                    samples_original = samples_original_tensor.detach().cpu().numpy()
                    observed_original = observed_original_tensor.detach().cpu().numpy()
                    eval_points = eval_points.detach().cpu().numpy()
                    
                    # 只选择需要评估的位置
                    eval_mask = eval_points.astype(bool)
                    pred_eval = samples_original[eval_mask]
                    true_eval = observed_original[eval_mask]
                    
                    # 计算指标
                    mse = mean_squared_error(true_eval, pred_eval)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(true_eval, pred_eval)
                    
                    all_rmse.append(rmse); all_mse.append(mse); all_mae.append(mae)
                    pbar.set_postfix({
                        'Batch': f'{batch_no}/{len(test_loader)}',
                        'RMSE': f'{np.mean(all_rmse):.4f}', 
                        'MAE': f'{np.mean(all_mae):.4f}'
                    })
                    pbar.update(1)
        
        final_rmse = np.mean(all_rmse)
        final_mse = np.mean(all_mse)
        final_mae = np.mean(all_mae)

        print("\n--- 测试结果 (Test-Only Mode) ---")
        print(f"Final RMSE: {final_rmse:.4f}")
        print(f"Final MAE: {final_mae:.4f}")
        print(f"Final MSE: {final_mse:.4f}")
        
        return # 结束函数

    # =========================
    # 模式: 训练+验证 / 完整
    # =========================
    
    # --- 4. 训练设置 (仅 'train' 和 'full' 模式) ---
    dataset_name = config['dataset']['name']
    model_save_dir = config['results']['save_dir']
    os.makedirs(model_save_dir, exist_ok=True)
    
    run_timestamp = time.strftime('%Y%m%d%H%M%S')
    print(f"本次运行ID (时间戳): {run_timestamp}")
    model_filename = f'{dataset_name}_{run_timestamp}.pth'
    model_save_path = os.path.join(model_save_dir, model_filename)
    print(f"\n模型将保存至: {model_save_path}")

    # 5. 日志记录器
    log_dir = os.path.join(config['results']['log_dir'], dataset_name, run_timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存至: {log_dir}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    epochs = config['train']['epochs']
    best_val_loss = float('inf')
    
    print(f"\n开始训练，共 {epochs} 个 epochs...")
    
    # 调整 pbar 的 total_step
    steps_per_epoch = len(train_loader) + len(val_loader)
    if mode == 'full':
        steps_per_epoch += len(test_loader)
    total_step = steps_per_epoch * epochs

    final_metrics_log = {}

    with tqdm(total=total_step, desc=f"[训练]", ncols=150, unit="batch") as pbar:
        for epoch in range(epochs):
            
            # --- 训练 ---
            model.train()
            total_train_loss = 0
            pbar.set_description(f"[训练 Epoch {epoch+1}/{epochs}]")
            for (j, batch) in enumerate(train_loader, 1):
                optimizer.zero_grad()
                train_loss = model(batch, is_train=1)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += train_loss.item()
                avg_train_loss = total_train_loss / j
                writer.add_scalar('Train/Batch_Loss', train_loss.item(), epoch * len(train_loader) + j)
                pbar.set_postfix({'Loss': f'{avg_train_loss:.4f}'})
                pbar.update(1)
            writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # --- 验证 ---
            model.eval()
            total_val_loss = 0
            pbar.set_description(f"[验证 Epoch {epoch+1}/{epochs}]")
            with torch.no_grad():
                for j, batch in enumerate(val_loader, 1):
                    val_loss = model(batch, is_train=0)
                    total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss / j
                    writer.add_scalar('Validation/Batch_Loss', val_loss.item(), epoch * len(val_loader) + j)
                    pbar.set_postfix({'Val_Loss': f'{avg_val_loss:.4f}'})
                    pbar.update(1)
            writer.add_scalar('Validation/Epoch_Loss', avg_val_loss, epoch + 1)
            scheduler.step(avg_val_loss)
            
            # --- 保存最佳模型 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
                pbar.set_description(f"[验证 Epoch {epoch+1}] (新最佳)")
            
            # --- 测试 (仅 'full' 模式) ---
            if mode == 'full':
                model.eval()
                all_rmse, all_mse, all_mae = [], [], []
                pbar.set_description(f"[测试 Epoch {epoch+1}/{epochs}]")
                with torch.no_grad():
                    for batch_no, test_batch in enumerate(test_loader, start=1):
                        output = model.evaluate(test_batch, config['evaluation']['nsample'])
                        
                        # Tensors
                        samples, observed_data, eval_points, _, _ = output
                        
                        # 反标准化 (Tensors)
                        samples_original_tensor = inverse_transform(samples, mean_tensor, std_tensor)
                        observed_original_tensor = inverse_transform(observed_data, mean_tensor, std_tensor)

                        # Numpy
                        samples_original = samples_original_tensor.detach().cpu().numpy()
                        observed_original = observed_original_tensor.detach().cpu().numpy()
                        eval_points = eval_points.detach().cpu().numpy()
                        
                        eval_mask = eval_points.astype(bool)
                        pred_eval = samples_original[eval_mask]
                        true_eval = observed_original[eval_mask]
                        
                        mse = mean_squared_error(true_eval, pred_eval)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(true_eval, pred_eval)
                        
                        all_rmse.append(rmse); all_mse.append(mse); all_mae.append(mae)
                        pbar.set_postfix({
                            'Test_RMSE': f'{np.mean(all_rmse):.4f}',
                            'Test_MAE': f'{np.mean(all_mae):.4f}'
                        })
                        pbar.update(1)
                
                final_rmse = np.mean(all_rmse)
                final_mse = np.mean(all_mse)
                final_mae = np.mean(all_mae)
                
                writer.add_scalar('Test/Epoch_RMSE', final_rmse, epoch + 1)
                writer.add_scalar('Test/Epoch_MSE', final_mse, epoch + 1)
                writer.add_scalar('Test/Epoch_MAE', final_mae, epoch + 1)
                
                # 记录最后一次的指标
                final_metrics_log = {
                    "RMSE": final_rmse,
                    "MAE": final_mae,
                    "MSE": final_mse
                }
            writer.flush()

    writer.close()
    print(f"\n训练/验证完成。最佳模型已保存至: {model_save_path}")
    
    if mode == 'full':
        print("\n--- 最终测试结果 (Full Mode, Last Epoch) ---")
        print(f"Final RMSE: {final_metrics_log.get('RMSE', 'N/A'):.4f}")
        print(f"Final MAE: {final_metrics_log.get('MAE', 'N/A'):.4f}")
        print(f"Final MSE: {final_metrics_log.get('MSE', 'N/A'):.4f}")
    elif mode == 'train':
        print("\n--- 训练/验证完成 (Train Mode) ---")
        print("未运行测试集。")


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
        '--mode',
        choices=['train', 'test', 'full'],
        default='train',
        help='运行模式: (train: 仅训练和验证, test: 仅测试, full: 训练、验证和测试)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='预训练模型的路径 (在 "test" 模式下必需)'
    )
    args = parser.parse_args()

    # 检查 'test' 模式的依赖
    if args.mode == 'test' and args.model_path is None:
        parser.error('--model_path 是 "test" 模式的必需参数')
    
    # 检查 'test' 模式下模型文件是否存在
    if args.mode == 'test' and not os.path.exists(args.model_path):
        parser.error(f'模型文件未找到: {args.model_path}')

    config = load_config(args.config)
    
    # 运行实验
    run_experiment(config, args.mode, args.model_path)


if __name__ == "__main__":
    main()