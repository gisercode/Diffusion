import os
import yaml
import subprocess
import pandas as pd
from collections import defaultdict

# 导入主实验运行函数
from main import run_experiment, load_config

def run_ablation_study():
    """
    自动执行所有消融实验并以表格形式报告结果。
    """
    base_config_path = 'config.yaml'
    base_config = load_config(base_config_path)

    # 定义所有消融实验的配置覆盖 (Ours 在最后)
    ablation_configs = {
        'w/o Self-supervision Mask': {'ablation': {'use_self_supervision': False}},
        'w/o Spatiotemporal dependencies': {'ablation': {'use_spatiotemporal': False}},
        'w/o Spatiotemporal heterogeneity': {'ablation': {'use_heterogeneity': False}},
        'w/o Dependencies and heterogeneity': {'ablation': {'use_spatiotemporal': False, 'use_heterogeneity': False}},
        'w/o Guidance module': {'ablation': {'use_guidance': False}},
        'w/o Tensor net': {'ablation': {'use_tensor_net': False}},
        'DiffuSDH (Ours)': {},
    }

    # 存储所有实验结果
    results = defaultdict(lambda: defaultdict(dict))

    # 依次执行每个实验
    for study_name, overrides in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"Running Ablation Study: {study_name}")
        print(f"{'='*80}")

        # 深拷贝基础配置以避免互相影响
        current_config = deep_copy_config(base_config)

        # 应用当前实验的配置覆盖
        for key, value in overrides.items():
            if key in current_config:
                current_config[key].update(value)
            else:
                current_config[key] = value
        
        # 定义数据集及其在表格中的显示名称
        datasets_to_run = {
            'metr_la': 'METR-LA',
            'pm25': 'AQI-36',
            'pre': 'PRE'
        }

        # 为每个数据集运行实验
        for dataset_name, display_name in datasets_to_run.items():
            print(f"\n--- Dataset: {display_name} ({dataset_name}) ---")
            current_config['dataset']['name'] = dataset_name
            
            # 运行实验并捕获指标
            rmse, mae, _ = run_experiment(current_config, quick_eval=True)
            
            # 使用显示名称存储结果
            results[study_name][display_name]['RMSE'] = rmse
            results[study_name][display_name]['MAE'] = mae

    # 格式化并打印最终结果表格
    print_results_table(results)

def deep_copy_config(config):
    """递归深拷贝配置字典"""
    if isinstance(config, dict):
        return {k: deep_copy_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [deep_copy_config(i) for i in config]
    else:
        return config

def print_results_table(results):
    """
    将结果以精美的 Markdown 表格形式打印出来。
    """
    # 定义表头
    headers = [
        "Ablation Study",
        "METR-LA (RMSE)", "METR-LA (MAE)",
        "AQI-36 (RMSE)", "AQI-36 (MAE)",
        "PRE (RMSE)", "PRE (MAE)"
    ]
    
    # 构建表格数据
    table_data = []
    for study_name in results.keys():
        row = [study_name]
        row.append(f"{results[study_name]['METR-LA'].get('RMSE', 'N/A'):.4f}")
        row.append(f"{results[study_name]['METR-LA'].get('MAE', 'N/A'):.4f}")
        row.append(f"{results[study_name]['AQI-36'].get('RMSE', 'N/A'):.4f}")
        row.append(f"{results[study_name]['AQI-36'].get('MAE', 'N/A'):.4f}")
        row.append(f"{results[study_name]['PRE'].get('RMSE', 'N/A'):.4f}")
        row.append(f"{results[study_name]['PRE'].get('MAE', 'N/A'):.4f}")
        table_data.append(row)

    # 使用 pandas 创建并打印表格
    df = pd.DataFrame(table_data, columns=headers)
    print("\n\n" + "="*120)
    print(" " * 45 + "Final Ablation Study Results")
    print("="*120)
    print(df.to_markdown(index=False))
    print("="*120)

if __name__ == "__main__":
    run_ablation_study()