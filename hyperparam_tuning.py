import yaml
import subprocess
import pandas as pd
from datetime import datetime
import copy
from tqdm import tqdm

# 1. 定义超参数搜索空间
param_grid = {
    'diffusion.num_steps': [50, 100, 200],
    'diffusion.layers': [2, 4, 6],
    'diffusion.rank': [4, 8, 16],
    'train.epochs': [50, 100] 
}

# 2. 准备结果记录文件
results_file = 'tuning_results_ovat.csv'
header = list(param_grid.keys()) + ['RMSE', 'MAE', 'MSE']
results_df = pd.DataFrame(columns=header)
results_df.to_csv(results_file, index=False)

# 3. 生成 "一次一个变量" (OVAT) 的实验组合
base_params = {key: values[0] for key, values in param_grid.items()}
experiments = []

# 添加基准实验
experiments.append(('baseline', base_params))

# 为每个参数生成实验
for param_name, values in param_grid.items():
    for value in values[1:]: # 从第二个值开始，因为第一个值是基准
        experiment_params = copy.deepcopy(base_params)
        experiment_params[param_name] = value
        experiments.append((f'tune_{param_name}', experiment_params))

print(f"总共需要进行 {len(experiments)} 次实验 (基于 OVAT 策略)...")

# 4. 循环执行实验
for exp_type, params in tqdm(experiments, desc="超参数调优总体进度"):
    print(f"\n--- 当前实验 ({exp_type}) ---")
    print(f"参数: {params}")

    # 加载基础配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新当前实验的参数
    for key, value in params.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
        
    # 强制设置为快速评估模式
    config['evaluation']['probabilistic_eval'] = False
    config['evaluation']['nsample'] = 1
    
    # 将修改后的配置写入临时文件
    temp_config_path = 'temp_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # 调用主训练脚本
    command = f"python main.py --config {temp_config_path}"
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        output = result.stdout
        rmse, mae, mse = -1, -1, -1
        for line in output.split('\n'):
            if "Final RMSE:" in line:
                rmse = float(line.split(':')[-1].strip())
            if "Final MAE:" in line:
                mae = float(line.split(':')[-1].strip())
            if "Final MSE:" in line:
                mse = float(line.split(':')[-1].strip())

        print(f"结果: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        new_row = params.copy()
        new_row.update({'RMSE': rmse, 'MAE': mae, 'MSE': mse})
        row_df = pd.DataFrame([new_row])
        row_df.to_csv(results_file, mode='a', header=False, index=False)

    except subprocess.CalledProcessError as e:
        print(f"实验失败: {params}")
        print(e.stderr)
        error_row = params.copy()
        error_row.update({'RMSE': 'failed', 'MAE': 'failed', 'MSE': 'failed'})
        row_df = pd.DataFrame([error_row])
        row_df.to_csv(results_file, mode='a', header=False, index=False)

print("\n--- 所有实验完成 ---")
print(f"结果已保存至 {results_file}")
