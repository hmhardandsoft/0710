# 个体内长/短 ISI VEP 与个体间 VEP–SSVEP 联系的分析方案

## 1. 研究目标与总体框架

本项目包含三种视觉刺激状态：

1. **长 ISI VEP**：ISI 约为 1.1–2.1 s，用于估计视觉系统充分恢复后的孤立瞬态反应。
2. **短 ISI VEP**：ISI 约为 60–135 ms，用于研究快速重复刺激下的反应重叠、适应、不应期和恢复过程。
3. **多目标 10 Hz SSVEP**：所有目标以相同的 10 Hz 基频刺激、使用不同相位编码，用于研究周期跟随、相位传递、注意调制和解码性能。

短 ISI 的中位数接近 100 ms，与 10 Hz SSVEP 的 100 ms 周期接近，因此可以把三类数据放进一个共同的**视觉时间动力学框架**：

```text
长 ISI VEP：恢复充分的基准瞬态响应
       ↓
短 ISI VEP：约 10 Hz 速度下的适应、恢复和相位锁定
       ↓
10 Hz SSVEP：周期驱动与注意状态下的稳态响应
```

当前数据最适合回答：

> 个体的视觉恢复、快速适应和事件锁相特征，能否解释其多目标 10 Hz SSVEP 的信噪比、相位稳定性和分类表现？

当前数据不适合直接声称：

> 中央单目标 VEP 的一个响应核可以完整生成或解释多目标 SSVEP。

## 2. 数据条件和解释边界

### 2.1 八通道数据的空间限制

当前通道为：

```text
FCz、TP9、Pz、POz、O1、Oz、O2、TP10
```

其中主要的后部视觉通道为 `Pz、POz、O1、Oz、O2`，TP9/TP10 主要作为参考。该配置可以支持：

- 固定枕叶 ROI 分析；
- Oz 单通道复现；
- O1–O2 左右差异；
- 后部通道结果的一致性检查；
- 经过交叉验证的后部通道空间加权。

但不适合支持：

- 全头皮拓扑比较；
- TANOVA 或微状态分析；
- 可靠的 EEG 源定位；
- 根据头皮分布区分多个视觉皮层源。

因此，后续报告中应使用“后部通道模式”或“枕叶 ROI”，避免使用“全头皮拓扑”或“源分布”。

### 2.2 多目标同频 SSVEP 的可辨识性限制

理想的多输入模型为：

\[
y_c(t)=\sum_{p=1}^{P}x_p(t)*h_{c,p}(t)+\epsilon_c(t)
\]

其中 \(x_p(t)\) 是第 \(p\) 个目标的亮度序列，\(h_{c,p}(t)\) 是该空间位置到通道 \(c\) 的响应核。

当前 VEP 只测量中央单方块，不能得到不同 SSVEP 目标位置、注意状态和非注意状态各自的响应核。并且所有 SSVEP 目标具有相同基频，仅相位不同；在单一固定相位配置下，各目标的基频输入高度共线，不能仅凭 EEG 唯一分解每个目标的贡献。

因此，当前项目应采用两层解释：

- **主要分析**：VEP 与 SSVEP 的被试级特征关联。
- **探索分析**：使用 VEP 响应预测 SSVEP 的总体趋势，但不把预测误差完全解释为神经非线性。

## 3. 通用预处理与质量控制

VEP 和 SSVEP 应尽量使用相同的预处理规范：

1. 确认设备输出的通道顺序和幅度单位。
2. 使用一致的 TP9/TP10 参考或其他预先规定的参考方式。
3. 连续数据上进行零相位带通滤波和 50 Hz 陷波。
4. 检查坏通道、异常峰峰值、突变、肌电和运动伪迹。
5. 保存每个文件、block 和条件的有效数据比例与伪迹指标。
6. 如果不同范式或日期使用不同显示器，记录刷新率、亮度、刺激尺寸和视距。
7. 相位分析前最好使用光电二极管估计真实视觉 onset、固定显示延迟和时间抖动。

短 ISI epoch 大量重叠，不能把相邻 epoch 当作独立统计样本。置信区间和重采样应优先以 **block** 为单位，而不是以单个短 ISI epoch 为单位。

## 4. 长 ISI VEP 分析

长 ISI 条件用于估计恢复充分的基准响应 \(h_{\mathrm{long}}(t)\)。

### 4.1 推荐处理

- epoch 建议覆盖刺激前至少 200 ms、刺激后 500–800 ms；
- 使用不受前一刺激影响的刺激前基线；
- 先按 block 平均，再形成被试平均；
- 主分析采用固定枕叶 ROI：`POz、O1、Oz、O2`；
- Oz 单通道结果作为复现分析；
- 避免直接在 0–400 ms 内寻找全局最大值并统一标记为 P 峰。

### 4.2 推荐特征

- 预定义时间窗内的 P1、N1、P2 平均振幅；
- 成分局部峰值或分数面积延迟；
- 峰间振幅，例如 P1–N1；
- 指定时间窗曲线下面积；
- 枕叶 ROI 波形；
- block 间相关和分半信度；
- ITPC 和 ERSP，作为平均 VEP 的补充。

固定时间窗应根据刺激类型和全体数据确定，并避免针对每名被试单独挑选最显著的窗口。

## 5. 短 ISI VEP 分析

短 ISI 为 60–135 ms，而现有分析 epoch 为 −50–400 ms。普通平均存在三个问题：

1. 刺激前基线包含前一次刺激的反应；
2. 一个 epoch 内通常包含多个后续刺激；
3. 相邻 epoch 共享大量相同 EEG 样本。

因此，短 ISI 的普通触发平均只能作为描述性图形，主分析应使用连续数据的回归去卷积。

### 5.1 回归去卷积模型

基础模型为：

\[
EEG(t)=\sum_i h(t-t_i,ISI_i)+\epsilon(t)
\]

建议设计矩阵至少包含：

```text
event
spline(previous_ISI)
spline(previous_previous_ISI)
trial_number
block
```

`previous_ISI` 应优先作为连续变量使用 spline 建模。也可以生成若干便于展示的条件预测，例如 ISI 为 70、80、90、100、110、120 和 130 ms 时的响应波形。

该模型的目标是同时分离：

- 相邻刺激的线性重叠；
- 前一 ISI 对当前反应的非线性影响；
- 连续刺激中的累积适应或疲劳；
- block 间状态差异。

### 5.2 恢复与适应指标

以 long VEP 作为恢复充分的基准，定义：

\[
G_{100}=\frac{A_{\mathrm{short}}(ISI=100\,ms)}{A_{\mathrm{long}}}
\]

其中 \(G_{100}\) 表示接近 10 Hz 刺激速度时的适应增益。

其他推荐指标：

- short–long 成分振幅差；
- short–long 成分延迟差；
- long–short 波形相关；
- 不同 ISI 下的响应恢复曲线；
- 恢复时间常数 \(\tau\)；
- short 条件的 ITPC；
- short 条件的事件序列—EEG 相干；
- block 内随时间变化的适应斜率。

恢复曲线可以使用 spline，也可以在数据支持时拟合：

\[
A(ISI)=A_\infty(1-e^{-ISI/\tau})
\]

## 6. SSVEP 分析

### 6.1 被试级核心指标

对每名被试提取：

- 10 Hz 复数响应振幅和相位；
- 10 Hz SNR；
- 10 Hz PLV；
- 相位传递斜率和截距；
- 相位误差或相位噪声；
- 不同分析窗长度下的相位稳定性；
- 分类准确率、错误类型和容量指标；
- 如果条件允许，注意/非注意增益；
- 10、20、30 Hz 的基频与谐波响应。

二值方波刺激包含谐波，脑响应也可能产生额外非线性谐波。因此不应只分析 10 Hz；至少应同时报告 1f、2f、3f。

跨被试比较时优先使用 SNR、PLV、相位噪声和注意增益，而不是只使用绝对幅度。绝对 EEG 幅度同时受头骨传导、电极接触、参考方式等非神经因素影响。

## 7. VEP–SSVEP 跨范式联系

### 7.1 被试级特征表

最终建立一张每名被试一行的特征表：

```text
subject
long_P1_amplitude
long_N1_amplitude
long_P2_amplitude
long_component_latency
short_long_gain_100ms
recovery_tau
short_ITPC
short_event_train_coherence
SSVEP_10Hz_SNR
SSVEP_10Hz_PLV
SSVEP_phase_slope
SSVEP_phase_intercept
SSVEP_phase_noise
SSVEP_accuracy
SSVEP_attention_gain
```

所有跨被试相关或回归中，每个点必须代表一名被试，不能把 trial 当成独立被试。

### 7.2 优先检验的假设

1. long VEP 振幅是否预测 SSVEP SNR？
2. VEP 成分延迟是否预测 SSVEP 相位滞后？
3. \(G_{100}\) 是否预测稳态 SSVEP 增益或注意增益？
4. 恢复时间常数 \(\tau\) 是否预测 SSVEP PLV 或相位噪声？
5. short ITPC 是否预测 SSVEP PLV 和相位分类表现？
6. short 的事件序列—EEG 相干是否预测 10 Hz SSVEP SNR？
7. short 响应是否比 long 响应更能预测 SSVEP 个体差异？

可采用以下被试级模型：

```text
SSVEP_SNR        ~ long_VEP_amplitude + short_long_gain_100ms
SSVEP_PLV        ~ short_ITPC + recovery_tau
SSVEP_phase_noise ~ recovery_tau + short_event_train_coherence
SSVEP_accuracy   ~ SSVEP_SNR + SSVEP_phase_noise + short_long_gain_100ms
```

相位结果应使用圆统计或圆—线性回归。10 Hz 下 1 ms 相当于 3.6° 相位，因此触发和显示延迟校准非常重要。

### 7.3 统计策略

- 个体内恢复曲线：回归或广义加性模型；
- 多被试条件效应：线性混合效应模型；
- 被试级 VEP–SSVEP 联系：稳健回归、Spearman 相关或置换检验；
- 不确定性：按 block bootstrap；
- 多指标检验：控制 FDR，或预先指定少量主要指标；
- 预测分析：留一被试交叉验证，避免在同一批被试上选特征并报告训练拟合度。

如果当前只有两名被试，只能进行个案描述、方法验证和可视化，不能据此进行个体间相关推断。正式样本量应根据预期效应和被试级重复测量设计进行功效分析或模拟。

## 8. VEP 预测 SSVEP 的探索性分析

在线性系统假设下，可以使用 VEP 响应核和实际周期刺激序列生成预测 SSVEP：

\[
\widehat{SSVEP}(t)=x_{10Hz}(t)*h(t)
\]

建议分别构造：

1. 使用 \(h_{\mathrm{long}}(t)\) 的预测，表示无快速适应的基准模型；
2. 使用 \(h_{\mathrm{short}}(t\mid ISI=100ms)\) 的预测，表示考虑速率特异适应的模型。

比较指标包括：

- 预测与真实 SSVEP 的波形相关；
- 10 Hz 振幅比例；
- 10 Hz 相位误差；
- 20/30 Hz 谐波误差；
- held-out block 的解释方差。

由于当前 VEP 是中央单目标，而 SSVEP 是多位置、同频、注意调制的刺激，此分析只能作为探索性整体预测。预测残差同时可能来自位置差异、注意、输入共线、视觉适应和真正的神经非线性，不能全部归因于“非线性 SSVEP 机制”。

## 9. 建议增加的桥接实验

如果后续目标是建立可辨识的生成模型，应增加位置匹配的单目标校准：

1. 在每个 SSVEP 目标位置分别呈现单目标刺激；
2. 刺激大小、亮度、对比度、占空比和视距与 SSVEP 一致；
3. 同时采集长 ISI 和以 100 ms 为中心的随机抖动 ISI；
4. 分别测量注意与非注意状态；
5. 使用光电二极管保存每个目标的实际亮度时间序列；
6. 在训练数据中估计位置/注意特异响应核，在未参与拟合的 SSVEP block 上验证。

这样可以估计：

\[
h_{\mathrm{left}},\quad
h_{\mathrm{right}},\quad
h_{\mathrm{attended}},\quad
h_{\mathrm{unattended}}
\]

并建立真正的多输入预测模型。

## 10. 推荐结果目录和表格

```text
analysis_results/
├── qc/
│   ├── recording_qc.csv
│   └── block_qc.csv
├── vep/
│   ├── long_subject_waveforms.npz
│   ├── short_deconvolved_waveforms.npz
│   ├── recovery_curve_by_subject.csv
│   └── vep_subject_features.csv
├── ssvep/
│   ├── ssvep_trial_metrics.csv
│   ├── ssvep_subject_features.csv
│   └── harmonic_summary.csv
└── cross_paradigm/
    ├── vep_ssvep_subject_features.csv
    ├── association_models.csv
    └── figures/
```

每名被试至少生成以下图形：

1. long 平均 VEP 与 short 去卷积 VEP；
2. ISI—振幅恢复曲线；
3. short/long 适应增益；
4. SSVEP 10 Hz 复数响应和 PLV；
5. SSVEP 基频与谐波谱；
6. VEP 指标与 SSVEP 指标的被试级关系图。

## 11. 推荐实施顺序

### 第一阶段：确保数据可靠

- 修正 VEP 采集与分析的文件名和目录规则；
- 核对通道顺序、幅度单位和参考方式；
- 加入伪迹标记、block 级 QC 和实际触发间隔检查；
- 统一 VEP 与 SSVEP 的通道和预处理规范。

### 第二阶段：建立被试内 VEP 特征

- 完成长 ISI 成分窗分析；
- 完成短 ISI 回归去卷积；
- 得到 \(G_{100}\)、恢复时间常数、延迟差和 ITPC；
- 检查这些指标的 block 间可靠性。

### 第三阶段：整理 SSVEP 特征

- 汇总 10 Hz SNR、PLV、相位误差和分类表现；
- 增加 20/30 Hz 谐波；
- 优先保留稳定且可复现的被试级指标。

### 第四阶段：跨范式检验

- 合并被试级 VEP 和 SSVEP 特征；
- 先检验预先规定的少量主要假设；
- 使用 block bootstrap 和被试级置换/稳健回归；
- 将 VEP 预测 SSVEP 的生成分析明确标记为探索性。

## 12. 方法学参考

- Ehinger BV, Dimigen O. *Unfold: an integrated toolbox for overlap correction, non-linear modeling, and regression-based EEG analysis.* PeerJ, 2019. <https://pmc.ncbi.nlm.nih.gov/articles/PMC6815663/>
- Smith NJ, Kutas M. *Regression-based estimation of ERP waveforms: II. Nonlinear effects, overlap correction, and practical considerations.* Psychophysiology, 2015. <https://pubmed.ncbi.nlm.nih.gov/25195691/>
- Capilla A, Pazo-Alvarez P, Darriba A, Campo P, Gross J. *Steady-state visual evoked potentials can be explained by temporal superposition of transient event-related responses.* PLoS ONE, 2011. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3022588/>
- Kitajima S. *The cumulative inhibitory effect of repetitively flashed stimuli on the recovery process of the human visual evoked potential to a test stimulus.* Electroencephalography and Clinical Neurophysiology, 1978. <https://pubmed.ncbi.nlm.nih.gov/76542/>

