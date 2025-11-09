import numpy as np

def analyze_data(file_path='debug_data.npz'):
    """
    加载并分析 debug_data.npz 文件，提供详细的诊断信息。
    """
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"错误: 调试数据文件 '{file_path}' 不存在。请先运行 debug_evaluate_data.py。")
        return

    print(f"--- 正在分析文件: {file_path} ---")

    # --- 1. 提取数据并打印基本信息 ---
    c_target = data['c_target']
    samples = data['samples']
    eval_points = data['eval_points']

    print("\n[1. 基本信息]")
    print(f"  - c_target (真实值) shape: {c_target.shape}, dtype: {c_target.dtype}")
    print(f"  - samples (预测样本) shape: {samples.shape}, dtype: {samples.dtype}")
    print(f"  - eval_points (评估点) shape: {eval_points.shape}, dtype: {eval_points.dtype}")

    # --- 2. 检查无效值 (NaN / Inf) ---
    print("\n[2. 无效值检查]")
    print(f"  - c_target 中是否存在 NaN: {np.isnan(c_target).any()}")
    print(f"  - c_target 中是否存在 Inf: {np.isinf(c_target).any()}")
    print(f"  - samples 中是否存在 NaN: {np.isnan(samples).any()}")
    print(f"  - samples 中是否存在 Inf: {np.isinf(samples).any()}")

    # --- 3. 分析数值范围和分布 ---
    print("\n[3. 数值范围分析]")
    print("  --- c_target (真实值) ---")
    print(f"    - 最小值: {np.min(c_target):.4f}")
    print(f"    - 最大值: {np.max(c_target):.4f}")
    print(f"    - 均值:   {np.mean(c_target):.4f}")
    print(f"    - 标准差: {np.std(c_target):.4f}")
    
    # 对所有样本的预测值进行整体分析
    print("  --- samples (所有预测样本) ---")
    print(f"    - 最小值: {np.min(samples):.4f}")
    print(f"    - 最大值: {np.max(samples):.4f}")
    print(f"    - 均值:   {np.mean(samples):.4f}")
    print(f"    - 标准差: {np.std(samples):.4f}")

    # --- 4. 检查评估点 ---
    print("\n[4. 评估点分析]")
    total_eval_points = np.prod(eval_points.shape)
    active_eval_points = np.sum(eval_points)
    print(f"  - eval_points 总数: {total_eval_points}")
    print(f"  - 值为 1 的评估点数: {active_eval_points}")
    if active_eval_points > 0:
        print(f"  - 评估点比例: {active_eval_points / total_eval_points * 100:.2f}%")
    else:
        print("  - 警告: 没有任何点被用于评估！")

    # --- 5. 计算样本间差异 ---
    print("\n[5. 预测样本多样性分析]")
    if samples.shape[1] > 1: # 确保有多个样本
        # 计算每个数据点在所有样本间的标准差
        std_across_samples = np.std(samples, axis=1)
        print(f"  - 跨样本标准差的均值: {np.mean(std_across_samples):.4f}")
        print("    (这个值如果接近0，说明所有预测样本都非常相似，模型可能已坍缩)")
    else:
        print("  - 只有一个预测样本，无法分析多样性。")
        
    print("\n--- 分析完成 ---")

if __name__ == "__main__":
    analyze_data()