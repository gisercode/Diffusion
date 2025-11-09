# PriSTI 模型深度解析：架构、逻辑与数据流

本文档旨在提供对 `PriSTI` 条件扩散模型的全面、深入的分析，涵盖其高级架构、核心模块功能、关键技术实现以及贯穿整个项目的数据处理流程。

## 1. 高级架构概览

`PriSTI` 是一个专为时空数据设计的先进条件扩散模型，其核心优势在于能够在一个统一的框架内处理**插补、重构和预测**等多种复杂任务。这一能力是通过精巧的**引导机制**和灵活的**数据掩码策略**共同实现的。

整个系统可以被划分为两个主要层次：

1.  **过程管理层 (`model.py`)**: 负责编排整个扩散与去噪（逆向）过程。它不关心时空特征如何提取，只负责根据扩散理论，在正确的时间步 `t` 将带噪数据送入“噪声预测引擎”，并计算损失或生成最终结果。
2.  **噪声预测层 (`diff_models.py`)**: 这是模型的核心“大脑”，一个深度时空神经网络，其唯一职责是：接收一个特定时间步 `t` 的带噪样本，并预测其中所含的噪声。

此外，一个强大的**引导模块 (`gwnet.py`)** 作为辅助，通过处理不完整的观测数据来生成一个包含丰富时空上下文的引导信号，极大地提升了主模型的预测精度。

---

## 2. 核心组件详解

下面，我们将深入剖析项目中每个关键文件的具体职责和内部逻辑。

### 2.1 `config.yaml` - 项目的中央控制台

这是整个项目的“大脑”和“仪表盘”，它定义了从数据处理到模型训练再到最终评估的所有参数，实现了代码与实验配置的完全分离。

-   **`dataset` & `paths`**: 控制使用哪个数据集 (`metr_la`, `pems_bay`, `pm25`) 以及数据的输入输出路径。
-   **`dataloader`**:
    -   **`sliding_window`**: 定义了时间序列样本的长度 (`L_input`, `L_pred`) 和采样步长 (`stride`)。
    -   **`preset_masking`**: **第二层掩码**的配置。用于在验证/测试集上创建固定的、可复现的缺失场景（如传感器永久性故障），以评估模型的泛化能力。
    -   **`conditional_masking`**: **第三层掩码**的配置。用于在训练过程中为每个样本动态生成缺失，迫使模型学习如何从部分信息中恢复全部信息。
-   **`model`**: 定义了模型的基本属性，如嵌入维度、是否使用引导模块 (`use_guide`) 等。
-   **`diffusion`**: 定义了扩散过程的核心超参数，包括扩散步数 (`num_steps`)、噪声调度策略 (`schedule`) 以及核心去噪网络 (`Guide_diff`) 的结构参数（如层数、通道数、注意力头数等）。
-   **`train` & `evaluation`**: 控制训练循环（周期、学习率）和评估过程（是否进行多样本概率性评估）的行为。

### 2.2 `dataloader.py` - 精巧的三层掩码数据生成器

这是理解模型多任务能力的关键。它不仅仅是一个数据加载器，更是一个动态样本生成器，通过一个巧妙的**三层掩码系统**，为不同的任务和阶段提供定制化的数据。

-   **第一层：`observed_mask` (原始缺失)**
    -   **来源**: 数据集本身固有的缺失值（例如，值为 0 或 NaN）。
    -   **作用**: 反映了数据的最原始、最真实的状态。所有后续操作都必须基于这个掩码，不能在没有原始数据的地方凭空创造信息。

-   **第二层：`eval_mask` (预设缺失)**
    -   **来源**: 由 `config.yaml` 中的 `preset_masking` 配置生成。
    -   **作用**: 仅在 `val` 和 `test` 模式下生效。它在 `observed_mask` 的基础上，进一步模拟真实的、长期的传感器故障或数据丢失场景。`gt_mask = observed_mask * (1 - eval_mask)` 定义了在评估时可用于计算指标的“真值”点。

-   **第三层：`cond_mask` (条件缺失)**
    -   **来源**: 由 `config.yaml` 中的 `conditional_masking` 配置生成。
    -   **作用**: 仅在 `train` 模式下生效。它为每个训练样本动态生成缺失，从而创造出自监督学习任务。模型需要根据 `cond_mask` 中标记为 `1` 的点（条件），去预测和恢复标记为 `0` 的点。

最终，`dataloader` 为模型提供了 `cond_mask` (模型可见的条件) 和 `target_mask` (模型需要预测的目标)，这两者共同构成了扩散模型的学习目标。

### 2.3 `model.py` (`PriSTI` 类) - 扩散过程的高级管理器

此类是扩散模型理论的直接代码实现，它封装了整个 DDPM（Denoising Diffusion Probabilistic Models）流程。

-   **`__init__`**:
    -   根据 `config.yaml` 初始化扩散过程所需的参数，如 `beta` 调度表、`alpha` 和 `alpha_hat` 等。
    -   实例化核心的去噪网络 `Guide_diff`。

-   **`calc_loss` (训练)**:
    1.  随机采样一个时间步 `t`。
    2.  根据扩散公式 `x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise`，生成带噪样本 `noisy_data`。
    3.  调用 `self.diffmodel` (`Guide_diff` 的实例) 来预测噪声。
    4.  计算预测噪声与真实噪声之间的 `L2` 损失，但**只在 `target_mask` 标记的位置计算**，这是实现多任务学习的关键。

-   **`impute` (推理/生成)**:
    1.  从一个纯高斯噪声 `x_T` 开始。
    2.  从 `T-1` 到 `0` 进行迭代循环。在每个时间步 `t`：
        a.  将当前的样本 `x_t` 和条件信息送入 `self.diffmodel` 预测噪声。
        b.  使用 DDPM 的逆向采样公式，根据预测的噪声从 `x_t` 计算出去噪更彻底的 `x_{t-1}`。
    3.  循环结束后，返回最终生成的干净数据 `x_0`。

-   **`get_side_info`**: 负责生成辅助信息，包括**时间嵌入** (让模型感知序列中的位置) 和**节点嵌入** (让模型区分不同的传感器节点)。

### 2.4 `diff_models.py` (`Guide_diff` & `NoiseProject`) - 核心噪声预测引擎

这是项目的“心脏”，一个复杂的深度时空网络，被训练用于预测噪声。

-   **`Guide_diff` (顶层模块)**:
    -   **职责**: 协调整个噪声预测流程。
    -   **流程**:
        1.  **输入投影**: 使用 `Conv1d` 将输入的带噪数据 `x` 和引导数据 `itp_x` 投影到模型的内部维度。
        2.  **引导信号生成**: 如果 `use_guide` 为 `true`，则将 `itp_x` 送入 `GWNet` 模块，生成一个强有力的引导信号 `itp_info`。
        3.  **扩散步嵌入**: 将离散的时间步 `t` 通过 `DiffusionEmbedding` 模块转换为一个连续的向量表示 `diffusion_emb`。
        4.  **迭代去噪**: 将 `x`, `itp_info`, `diffusion_emb` 和 `side_info` 一同送入一系列堆叠的 `NoiseProject` 残差块中。
        5.  **输出投影**: 聚合所有 `NoiseProject` 块的输出，并通过两个 `Conv1d` 层最终输出预测的噪声。

-   **`NoiseProject` (核心构建块)**:
    -   **职责**: 这是执行实际时空特征学习的基本单元。
    -   **内部结构**: 每个 `NoiseProject` 块都包含两个并行的学习模块，用于同时从时间和空间维度提取特征：
        -   **`TemporalLearning`**: 使用一个基于 Transformer 的自注意力机制，在**时间维度**上捕捉序列的动态变化。
        -   **`SpatialLearning`**: 结合了 `AdaptiveGCN` (捕捉局部空间依赖) 和 `Attn_spa` (捕捉全局空间依赖)，在**空间维度**上学习节点间的相互影响。
    -   **引导机制**: `GWNet` 生成的引导信号 `itp_info` 会被注入到这两个模块中，从而在时间和空间两个层面上同时指导噪声的预测。

### 2.5 `gwnet.py` (`GWNet`) - 强大的时空引导模块

这是一个深度时空图卷积网络，其架构类似于 WaveNet。它被用作引导模块，专门从不完整或带噪的观测数据 (`itp_x`) 中提取高质量的时空特征，以生成引导信号 `itp_info`。

-   **核心特性**:
    -   **因果卷积**: 通过巧妙的填充（Padding）实现，确保在处理时间维度时，模型不会“看到”未来的信息。
    -   **图卷积 (`GraphConvNet`)**: 在每个卷积层后都进行图卷积操作，用于捕捉空间节点间的依赖关系。它同时支持**静态邻接矩阵**和**自适应邻接矩阵**，使其能够灵活地学习节点间的动态关联。
    -   **张量环卷积 (`TRConv2D`)**: **这是最新的关键升级**。模型中所有的标准 `Conv2d` 层都被替换为 `TRConv2D`。这项技术将高维卷积核分解为一系列低维张量的乘积，从而在**大幅减少模型参数**的同时，有效保持了模型的表达能力。

### 2.6 `layers.py` - 基础时空处理单元

该文件包含了构成 `Guide_diff` 和 `GWNet` 的所有基础构建模块。

-   **`AdaptiveGCN`**: 自适应图卷积网络。它不仅使用预定义的静态图结构，还能通过学习节点嵌入 (`nodevec1`, `nodevec2`) 来动态生成一个数据驱动的邻接矩阵，从而捕捉节点间潜在的、未在静态图中体现的关系。
-   **`TemporalLearning` & `SpatialLearning`**: 分别是 `NoiseProject` 内部的时间和空间学习模块的封装。
-   **`Attn_tem` & `Attn_spa`**: 基于 Transformer 的自定义注意力模块，分别用于捕捉时间和空间维度上的长距离依赖。
-   **`TRLinear`**: 张量环分解线性层。与 `TRConv2D` 类似，它被用于替换标准的前馈网络 (`nn.Linear`)，以降低模型复杂度和过拟合风险。
-   **`DiffusionEmbedding`**: 将离散的扩散时间步 `t` 转换为连续的、高质量的向量嵌入，作为模型的条件输入。

---

## 3. 数据流与架构可视化

为了更直观地理解上述组件如何协同工作，下面提供了两张流程图。

### 3.1 数据流与三层掩码生成图

这张图展示了数据从加载到最终送入模型的完整准备过程，重点突出了三层掩码机制。

```mermaid
graph TD
    subgraph "1. 数据加载 (DataLoader.__init__)"
        A1[原始数据文件 .npz] --> A2{加载 train/val/test 数据};
        A2 --> A3["<b>L1: observed_mask</b><br>(基于原始数据中的 0/NaN 生成)"];
        A2 --> A4["<b>L2: eval_mask</b><br>(仅 val/test, 基于 config 生成固定的预设缺失)"];
        A3 & A4 --> A5["计算 <b>gt_mask</b><br>(gt = observed AND NOT eval)"];
    end

    subgraph "2. 样本生成 (DataLoader.__getitem__)"
        B1[切分数据窗口 (长度 L)] --> B2{获取对应窗口的<br>observed_data, observed_mask, gt_mask};
        B2 -- train 模式 --> B3["<b>L3: cond_mask</b><br>(基于 config 动态生成条件缺失)"];
        B2 -- val/test 模式 --> B4["<b>cond_mask</b> = gt_mask<br>(模型看到所有真值)"];
        B3 & B4 --> B5{计算 <b>target_mask</b><br>(target = (1 - cond) AND observed)};
    end

    subgraph "3. 输入模型"
        C1[observed_data]
        C2[cond_mask]
        C3[target_mask]
        C4[side_info]
        C5[itp_info (来自引导数据)]
        C1 & C2 & C3 & C4 & C5 --> D{PriSTI 模型};
    end

    A5 --> B1;
    B5 --> C3;
```

### 3.2 端到端模型架构与逻辑流程图

这张图详细描绘了数据在 `PriSTI` 模型内部的完整处理流程，从输入到最终的噪声预测。

```mermaid
graph TD
    subgraph "输入数据 (来自 DataLoader)"
        X_obs[observed_data];
        M_cond[cond_mask];
        M_target[target_mask];
        S_info[side_info];
        I_coeffs[itp_info / CDE Coeffs];
    end

    subgraph "PriSTI 过程管理器 (model.py)"
        direction LR
        subgraph "训练 (forward)"
            T1[1. 随机采样时间步 t];
            T2[2. 生成高斯噪声 ε];
            X_obs & T1 & T2 --> T3("3. 生成带噪数据 xₜ");
            T3 & M_cond & S_info & I_coeffs & T1 --> DiffNet1[Guide_diff 去噪网络];
            DiffNet1 --> T4[预测噪声 ε_θ];
            T2 & T4 & M_target --> T5{"4. 计算损失<br>Loss = ||ε - ε_θ||² (仅在 target_mask 上)"};
        end

        subgraph "推理 (impute)"
            I1[1. 从纯噪声 xT 开始];
            subgraph "迭代 t = T-1 to 0"
                I2[当前样本 xₜ] & M_cond & S_info & I_coeffs --> DiffNet2[Guide_diff 去噪网络];
                DiffNet2 --> I3[预测噪声 ε_θ];
                I2 & I3 --> I4("2. 计算去噪后的 xₜ₋₁");
            end
            I1 --> I2;
            I4 --> I2;
            I4 --> I5[3. 最终生成数据 x'₀];
        end
    end

    subgraph "Guide_diff 噪声预测引擎 (diff_models.py)"
        direction TB
        D_IN[输入: xₜ, t, side_info, itp_info] --> D1[输入与引导数据投影];
        
        subgraph "引导路径 (Guidance Path)"
            D_itp[itp_info] --> GWNet[GWNet 时空引导模块];
            GWNet --> D_guide_signal[引导信号];
        end

        D1 & D_guide_signal --> D2["堆叠的 NoiseProject 模块"];
        
        subgraph "单个 NoiseProject 模块"
            N_IN[上层输入] & D_guide_signal --> N1[时间学习 TemporalLearning];
            N_IN & D_guide_signal --> N2[空间学习 SpatialLearning];
            subgraph "空间学习内部"
                direction LR
                SL_IN[输入] --> SL1[AdaptiveGCN];
                SL_IN --> SL2[Attn_spa];
                SL1 & SL2 --> SL_OUT[融合];
            end
            N1 & N2 --> N_OUT[输出];
        end
        
        D2 --> D3[聚合输出];
        D3 --> D_OUT[最终预测噪声 ε_θ];
    end

    %% 连接关系
    DiffNet1 -- "共享权重" --- DiffNet2;
    D_OUT -- "输出" --> T4;
    D_OUT -- "输出" --> I3;

    %% 样式
    classDef modelStyle fill:#bfa,stroke:#333,stroke-width:2px
    class DiffNet1,DiffNet2,D1,D2,D3,N1,N2,SL1,SL2,GWNet modelStyle