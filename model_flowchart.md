# PriSTI 模型架构流程图 (Mermaid 语法)

## 使用说明

您可以将下面的 Mermaid 代码块复制并粘贴到任何支持 Mermaid 的渲染器中，以生成流程图图像。推荐的工具包括：

- **[Mermaid Live Editor](https://mermaid.live/)**: 一个官方的在线编辑器，可以直接将代码转换为 SVG 或 PNG 图像。
- **VS Code**: 安装 `Markdown Preview Mermaid Support` 或 `Mermaid Markdown` 插件后，可以在 Markdown 预览中直接看到渲染后的图表。
- **Typora** 或 **Obsidian**: 这些 Markdown 编辑器原生支持 Mermaid。

---

## 流程图 Mermaid 代码

```mermaid
graph TD
    subgraph "1. 输入数据"
        direction LR
        A1[原始数据 x₀]
        A2[观测掩码 observed_mask]
        A3[条件掩码 cond_mask]
        A4[时间点 timepoints]
        A5[CDE插值系数 coeffs]
    end

    subgraph "2. 预处理与信息生成"
        direction TB
        subgraph "Side Info 生成"
            A4 --> S1[时间嵌入 Time Embedding]
            S2[节点ID] --> S3[特征嵌入 Feature Embedding]
            S1 & S3 --> S4[拼接 Side Info]
        end
        
        subgraph "引导路径 (Guidance Path)"
            A5 --> GP1[CDE插值信息 itp_info]
        end

        A1 & A3 --> C1[条件数据 cond_data]
    end

    subgraph "3. 训练阶段 (前向加噪与损失计算)"
        direction TB
        T1[1. 随机采样时间步 t]
        T2[2. 生成高斯噪声 ε]
        A1 --> T3("3. 计算带噪数据 xₜ<br>xₜ = √ᾱₜ * x₀ + √(1-ᾱₜ) * ε")
        
        T3 & S4 & C1 & T1 & GP1 --> M1[核心去噪网络 Guide_diff]
        
        M1 --> T4[预测噪声 ε_θ]
        T2 & T4 --> T5{4. 计算损失<br>Loss = &#124;&#124;ε - ε_θ&#124;&#124;²}
        T5 --> T6[反向传播更新模型]
    end

    subgraph "4. 推理阶段 (后向去噪生成)"
        direction TB
        I1[1. 从纯高斯噪声 xT 开始]
        
        subgraph "迭代循环 t = T-1 to 0"
            I2[当前带噪样本 xₜ]
            I2 & S4 & C1 & GP1 --> M2[核心去噪网络 Guide_diff]
            M2 --> I3[预测噪声 ε_θ]
            I2 & I3 --> I4("2. 计算去噪后的样本 xₜ₋₁<br>基于 xₜ 和 ε_θ")
        end
        
        I1 --> I2
        I4 --> I2
        I4 --> I5[3. 最终生成/修复的数据 x'₀]
    end

    subgraph "5. 核心去噪网络 Guide_diff 详解"
        direction TB
        D_IN["输入: xₜ, t, Side Info, cond_data, itp_info"] --> D1[输入投影]
        
        D1 --> D2["堆叠的 NoiseProject 模块 (带跳跃连接)"]
        
        subgraph "单个 NoiseProject 模块内部"
            direction TB
            N_IN[上层输入 y] --> N1[注入扩散步数嵌入<br>y = y + diff_emb]
            
            subgraph "时序学习 (Temporal Learning)"
                direction LR
                N1 -- 作为V --> TL1[Transformer Encoder]
                GP1 -- 作为Q, K --> TL1
                TL1 --> N2[时序特征]
            end
            
            subgraph "空间学习 (SpatialLearning)"
                direction TB
                N2 --> SL1["AdaptiveGCN (局部)<br>y_local = GCN(y) + y"]
                N2 --> SL2["Spatial Attention (全局)<br>y_attn = Attn(y, itp_info) + y"]
                SL1 & SL2 --> SL3[融合 y = y_local + y_attn]
                SL3 --> SL4["FFN (TRLinear)"]
            end
            
            N1 --> N2
            N2 --> SL1
            
            SL4 --> N3[注入 Side Info]
            N3 --> N4[门控激活与输出]
            N4 --> N_OUT[输出到下一层]
            N4 -- 跳跃连接 --> D3
        end
        
        D2 --> D3[聚合所有跳跃连接]
        D3 --> D4[输出投影]
        D4 --> D_OUT[最终预测噪声 ε_θ]
    end

    %% 连接关系
    M1 -- "共享权重" --- M2
    D_OUT -- "对应" --> T4
    D_OUT -- "对应" --> I3

    %% 样式
    style T5 fill:#f9f,stroke:#333,stroke-width:2px
    style I5 fill:#bbf,stroke:#333,stroke-width:2px
    classDef modelStyle fill:#bfa,stroke:#333,stroke-width:2px
    class M1,M2,D1,D2,D3,D4,N1,N2,N3,N4,SL1,SL2,SL3,SL4,TL1 modelStyle