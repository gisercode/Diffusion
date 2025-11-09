import os
import numpy as np
import yaml
import argparse
import pickle
import pandas as pd # 引入 pandas 用于处理时间序列

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(data_path, dataset_name):
    """根据数据集名称加载原始数据"""
    if dataset_name == 'metr_la':
        data_file = os.path.join(data_path, dataset_name, 'METALA.npy')
    elif dataset_name == 'pems_bay':
        data_file = os.path.join(data_path, dataset_name, 'PEMSBAY.npy')
    elif dataset_name == 'pm25':
        data_file = os.path.join(data_path, dataset_name, 'PM25.npy')
    elif dataset_name == 'pre':
        data_file = os.path.join(data_path, dataset_name, 'PRE.npy')
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")
    
    raw_data = np.load(data_file)
    print(f"成功加载原始数据 '{dataset_name}', 原始维度: {raw_data.shape}")
    return raw_data

def preprocess(config):
    """
    执行完整的离线数据预处理流程。
    """
    # 1. 读取配置
    dataset_name = config['dataset']['name']
    raw_data_dir = config['paths']['raw_data_dir']
    output_dir = config['paths']['output_dir']
    split_ratios = config['preprocessing']['split_ratios']
    seed = config['global']['random_seed']
    
    # 设置随机种子以保证可复现性
    np.random.seed(seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================================
    # 步骤 1: 初始加载、重塑与掩码创建
    # ==================================================================================
    # 加载数据
    raw_data_3d = load_raw_data(raw_data_dir, dataset_name)
    
    # 统一重塑为 (N, T_total)
    # 对于 (N, 1, T) 这种特殊维度的数据，先压缩
    if raw_data_3d.ndim == 3 and raw_data_3d.shape[1] == 1:
        print(f"检测到数据维度为 (N, 1, T)，将进行压缩...")
        raw_data_3d = raw_data_3d.squeeze(axis=1)
        
    N = raw_data_3d.shape[0]
    raw_data = raw_data_3d.reshape(N, -1)
    print(f"数据已重塑为 2D 张量，维度: {raw_data.shape}")

    # 创建原始缺失掩码 (值为0的位置为1)
    original_missing_mask = (raw_data == 0).astype(np.float32)
    print(f"已创建原始缺失掩码，缺失率: {original_missing_mask.mean():.4f}")

    # ==================================================================================
    # 步骤 2: 数据集划分与归一化
    # ==================================================================================
    
    # <<< --- 代码修改开始 --- >>>
    
    # 针对 pm25 数据集使用按月份划分的策略
    if dataset_name == 'pm25':
        print("检测到 pm25 数据集，将采用按月份的划分策略...")
        
        # pm25 数据时间范围为 2014-05-01 到 2015-04-30，共365天，每小时一个数据点
        # 创建时间索引
        timestamps = pd.date_range(start='2014-05-01', periods=raw_data.shape[1], freq='H')
        
        # 将数据和掩码转换为 DataFrame 以便按月份筛选
        df = pd.DataFrame(raw_data.T, index=timestamps)
        df_mask = pd.DataFrame(original_missing_mask.T, index=timestamps)
        
        # 根据参考论文的划分逻辑定义训练、验证和测试月份
        # Test: 3, 6, 9, 12月
        # Valid: 2, 5, 8, 11月
        # Train: 1, 4, 7, 10月 (以及 Valid 月份中不用于验证的部分，为简化我们直接用不同月份)
        train_months = [7, 10, 1, 4] 
        val_months = [5, 8, 11, 2]
        test_months = [6, 9, 12, 3]

        # 筛选数据
        train_df = df[df.index.month.isin(train_months)]
        val_df = df[df.index.month.isin(val_months)]
        test_df = df[df.index.month.isin(test_months)]

        train_mask_df = df_mask[df_mask.index.month.isin(train_months)]
        val_mask_df = df_mask[df_mask.index.month.isin(val_months)]
        test_mask_df = df_mask[df_mask.index.month.isin(test_months)]
        
        # 将筛选后的数据转换回 numpy 数组, 并变回 (N, T) 的形状
        train_data = train_df.values.T
        val_data = val_df.values.T
        test_data = test_df.values.T
        
        train_mask = train_mask_df.values.T
        val_mask = val_mask_df.values.T
        test_mask = test_mask_df.values.T

    else:
        # 对于 metr_la 和 pems_bay，继续使用原有的按比例划分
        print(f"检测到 {dataset_name} 数据集，将采用按比例的连续时序划分策略...")
        T_total = raw_data.shape[1]
        train_len = int(T_total * split_ratios[0])
        val_len = int(T_total * split_ratios[1])
        test_len = T_total - train_len - val_len

        train_data = raw_data[:, :train_len]
        val_data = raw_data[:, train_len : train_len + val_len]
        test_data = raw_data[:, -test_len:]

        train_mask = original_missing_mask[:, :train_len]
        val_mask = original_missing_mask[:, train_len : train_len + val_len]
        test_mask = original_missing_mask[:, -test_len:]

    # <<< --- 代码修改结束 --- >>>

    print(f"数据集已划分为: Train({train_data.shape}), Val({val_data.shape}), Test({test_data.shape})")
    
    # 数据归一化 (逻辑保持不变，但现在会基于新的 train_data 计算均值和标准差)
    mean, std = None, None
    if config['preprocessing']['use_provided_meanstd']:
        if dataset_name == 'metr_la':
            meanstd_filename = 'metr_meanstd.pk'
        # 修正 pems_bay 和 pm25 的均值标准差文件名读取逻辑
        elif dataset_name == 'pems_bay':
            meanstd_filename = 'pemsbay_meanstd.pk'
        elif dataset_name == 'pm25':
            meanstd_filename = 'pm25_meanstd.pk'
        elif dataset_name == 'pre':
            meanstd_filename = 'pre_meanstd.pk'
        else:
            meanstd_filename = f'{dataset_name.replace("_", "")}_meanstd.pk'
            
        meanstd_path = os.path.join(raw_data_dir, dataset_name, meanstd_filename)
        try:
            with open(meanstd_path, 'rb') as f:
                # 兼容不同 pickle 文件的编码
                meanstd_data = pickle.load(f, encoding='latin1') if dataset_name != 'pm25' else pickle.load(f)
            
            # 检查加载的维度是否正确
            if meanstd_data[0].shape[0] == N:
                mean = meanstd_data[0]
                std = meanstd_data[1]
                print("已加载预先计算的均值和标准差。")
            else:
                print(f"[警告] 加载的均值/标准差维度 ({meanstd_data[0].shape}) 与节点数 ({N}) 不匹配。")
        except FileNotFoundError:
            print(f"[警告] 未找到预计算的均值/标准差文件: {meanstd_path}")

    # 如果未能从文件加载，则从数据中计算
    if mean is None or std is None:
        print("将从训练数据中计算统计量 (忽略缺失值)...")
        # 使用 numpy.ma 模块来处理带掩码的数组，可以忽略特定值
        train_data_for_stats = np.ma.masked_equal(train_data, 0)
        mean = np.mean(train_data_for_stats, axis=1).data
        std = np.std(train_data_for_stats, axis=1).data
        print("已成功计算均值和标准差。")

    # 增加一个小的 epsilon 防止除以零
    std[std == 0] = 1.0
    
    # Z-Score 标准化
    # np.newaxis 用于将 (N,) 的 mean/std 广播到 (N, T)
    train_data_normalized = (train_data - mean[:, np.newaxis]) / std[:, np.newaxis]
    val_data_normalized = (val_data - mean[:, np.newaxis]) / std[:, np.newaxis]
    test_data_normalized = (test_data - mean[:, np.newaxis]) / std[:, np.newaxis]
    print("已对所有数据集应用 Z-Score 归一化。")

    # ==================================================================================
    # 步骤 3: 保存处理后的数据
    # ==================================================================================
    output_path = os.path.join(output_dir, f'{dataset_name}_processed.npz')
    
    np.savez_compressed(
        output_path,
        train_data=train_data_normalized,
        val_data=val_data_normalized,
        test_data=test_data_normalized,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        mean=mean,
        std=std
    )
    print(f"预处理完成！所有数据已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='配置文件路径'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config)