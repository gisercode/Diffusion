# 数据处理与加载流水线

本项目旨在为时间序列数据提供一个健壮、可配置且高效的数据处理与加载流水线，特别适用于需要复杂数据增强（如动态掩码）的自监督学习模型。

## 目录结构

```
.
├── data/                     # 存放原始数据集
│   ├── metr_la/
│   ├── pems_bay/
│   └── pm25/
├── processed_data/           # 存放预处理后的数据 (.npz 文件)
├── config.yaml               # 全局配置文件
├── preprocess.py             # 离线数据预处理脚本
├── dataloader.py             # 自定义 PyTorch DataLoader
├── main.py                   # 主程序入口与使用示例
└── README.md                 # 本文档
```

## 核心组件说明

### 1. `config.yaml`

这是整个数据流水线的“控制中心”。它允许用户在不修改任何代码的情况下，调整所有关键参数。

- **`dataset`**: 选择要处理的数据集 (`metr_la`, `pems_bay`, `pm25`)。
- **`paths`**: 管理所有输入（原始数据）和输出（处理后数据）的路径。
- **`preprocessing`**: 配置离线预处理阶段的参数，如数据集划分比例。
- **`dataloader`**: 配置在线数据加载阶段的参数，包括滑动窗口的长度、步长，以及动态掩码生成的复杂参数。
- **`global`**: 全局设置，如用于保证实验可复现性的随机种子。

### 2. `preprocess.py`

**离线数据预处理脚本**。此脚本只应运行一次（或在配置更改后重新运行）。

**功能**:
- 根据 `config.yaml` 加载指定的原始数据集。
- 将数据重塑为统一的 `(N, T_total)` 二维格式。
- 创建一个记录原始缺失位置的掩码。
- 严格按时间顺序将数据划分为训练、验证和测试集。
- 对数据进行Z-Score归一化（使用训练集的统计数据）。
- 将处理后的所有数据（`data`, `mask`, `mean`, `std`）打包保存到一个 `.npz` 文件中，存放在 `processed_data/` 目录下。

### 3. `dataloader.py`

**在线数据生成与加载器**。这是连接数据与模型的核心桥梁。

**功能**:
- 定义了一个自定义的 `torch.utils.data.Dataset` 类 (`TimeSeriesDataset`)。
- 在初始化时加载由 `preprocess.py` 生成的 `.npz` 文件。
- **动态生成样本**: 在每次迭代（`__getitem__`）时，它会执行以下操作：
    1.  通过滑动窗口从完整时间序列中切分出一个样本。
    2.  **对于训练集**: 随机生成一个复杂的 `target_mask`，模拟预测、插值、重构等多种任务，从而实现数据增强。
    3.  **对于验证/测试集**: 生成一个固定的 `target_mask`，仅用于评估模型的预测能力。
    4.  根据掩码计算出模型的最终输入 `observed_data` 和 `observed_mask`，以及用于计算损失的 `loss_mask`。
- 提供 `get_dataloader` 函数，方便地为不同模式（`train`, `val`, `test`）创建 `DataLoader` 实例。

### 4. `main.py`

**主程序入口与使用示例**。

**功能**:
- 整合并展示了整个数据流水线的标准使用流程。
- 加载 `config.yaml` 配置。
- 检查预处理数据是否存在，如果不存在则提供清晰的指示。
- 初始化训练、验证和测试集的 `DataLoader`。
- 模拟一个训练循环，从 `DataLoader` 中迭代获取数据批次，并打印其维度信息，验证流程的正确性。

## 使用指南

### 步骤 1: 配置环境

确保已安装必要的 Python 库: `PyYAML`, `numpy`, `torch`。

### 步骤 2: 修改配置

打开 `config.yaml` 文件，根据您的需求修改参数。例如，您可以切换 `dataset.name` 来处理不同的数据集，或者调整 `dataloader` 中的窗口长度和掩码参数。

### 步骤 3: 运行离线预处理

在终端中运行以下命令，生成处理后的数据文件：

```bash
python preprocess.py --config config.yaml
```

成功运行后，您将在 `processed_data/` 目录下看到一个名为 `{dataset_name}_processed.npz` 的文件。

### 步骤 4: 集成与训练

现在，您可以将此数据流水线集成到您的模型训练代码中。`main.py` 提供了一个完美的起点。您可以参考其逻辑，将 `train_loader`, `val_loader` 等用于您的训练和评估循环。

```python
# 在您的训练脚本中
from dataloader import get_dataloader
import yaml

# 1. 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 获取 DataLoader
train_loader = get_dataloader(config, 'train')

# 3. 开始训练
for epoch in range(num_epochs):
    for batch in train_loader:
        # batch 中包含了模型所需的所有数据
        # observed_data = batch['observed_data']
        # ...
        # model.train(batch)
        pass

python main.py --config config.yaml --evaluate_only（只加载模型用于评估）

python main.py --config config.yaml

### 导出数据的含义

**train_data.npy:**

含义: 经过 Z-Score 标准化处理后的训练数据集。这是模型在训练阶段用于学习模式和参数的输入数据。意味着所有原始缺失（例如，传感器故障导致的0值）的位置已经被设置为了 0。这是最基础的、干净的数据。
形状: (N, T_train)，其中 N 是节点（传感器）数量，T_train 是训练集的时间步长。

**train_mask.npy**
含义: 这是第1层掩码（原始掩码）。它标记了 train_data.npy 中的数据来源。1 = 这个数据点在原始文件中存在（是真实观测值）。0 = 这个数据点在原始文件中缺失。


**val_data.npy:**

含义: 经过 Z-Score 标准化处理后的验证数据集。这部分数据用于在训练过程中评估模型的性能，并调整超参数，以防止过拟合。
形状: (N, T_val)，其中 T_val 是验证集的时间步长。

**test_data.npy:**

含义: 经过 Z-Score 标准化处理后的测试数据集。这部分数据用于在模型训练完成后，对模型的最终性能进行无偏评估。
形状: (N, T_test)，其中 T_test 是测试集的时间步长。

**train_mask.npy:**

含义: 训练数据集的原始缺失掩码。值为 1 表示该位置有原始观测值（非零），0 表示原始数据中缺失（零值）。
作用: 用于指示训练数据中哪些是真实观测值，哪些是原始缺失。

**val_eval_mask.npy:**

含义: 验证数据集的预设缺失掩码（第二层缺失）。1 = 这个点被人为挖掉，模型需要预测这个点。0 = 这个点未被人为挖掉。
作用: 在验证阶段，这部分掩码与原始数据结合，用于模拟真实世界中可能出现的各种缺失模式。

**test_eval_mask.npy:**

含义: 测试数据集的预设缺失掩码（第二层缺失）。与 val_eval_mask.npy 类似，但应用于测试集。
作用: 在测试阶段，用于评估模型在特定缺失模式下的泛化能力。

**val_gt_mask.npy:**

含义: 验证数据集的真值掩码。它代表模型在评估时允许看到的已知数据* 1 = 这个点既在原始数据中存在（observed_mask=1），又被评估掩码挖掉（eval_mask=1）。
作用: 在验证阶段，模型通常会根据 val_gt_mask 来计算损失和评估指标，因为它代表了模型可以用来评估的“真实”观测值。
s
**test_gt_mask.npy:**

含义: 测试数据集的真值掩码。与 val_gt_mask.npy 类似，但应用于测试集。
作用: 在测试阶段，用于最终评估模型在去除预设缺失后的真实性能。
总结:
这些文件共同提供了经过预处理、标准化和不同缺失模式处理后的数据集，可用于模型的训练、验证和测试。_data.npy 文件包含实际的数值数据，而 _mask.npy 文件则指示了数据点的可用性或缺失状态。gt_mask 特别重要，因为它定义了在存在预设缺失的情况下，哪些数据点被认为是可用于评估的真值。



## 基线模型
Mean
BRITS
Transformer
ImputeFormer
Informer
SAITS1
GP-VAE
TimesNet
US-GAN
M-RNN

