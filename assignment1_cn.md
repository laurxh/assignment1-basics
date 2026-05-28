# CS336 作业1（基础）：构建Transformer语言模型

版本 26.0.3 | CS336 教学团队 | 2026年春季

---

## 完成进度

最近验证：2026-05-27，模型组件测试已通过至 `test_transformer_lm` 和 `test_transformer_lm_truncated_input`。

| # | 题目ID | 状态 |
|---|--------|------|
| 8 | linear | ✅ 已通过 |
| 9 | embedding | ✅ 已通过 |
| 10 | rmsnorm | ✅ 已通过 |
| 11 | positionwise_feedforward / swiglu | ✅ 已通过 |
| 12 | rope | ✅ 已通过 |
| 13 | softmax | ✅ 已通过 |
| 14 | scaled_dot_product_attention | ✅ 已通过 |
| 15 | multihead_self_attention | ✅ 已通过 |
| 16 | transformer_block | ✅ 已通过 |
| 17 | transformer_lm | ✅ 已通过 |
| 19 | cross_entropy | ⬜ 下一步 |
| 25 | data_loading | ⬜ 待做 |
| 3 | train_bpe | ⬜ 进行中 |
| 6 | tokenizer | ⬜ 待做 |

---

## 1 作业概述

本作业要求你从零构建训练一个标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

### 需要实现的内容

1. Byte-Pair Encoding (BPE) 分词器（第2节）
2. Transformer 语言模型（第3节）
3. 交叉熵损失函数和 AdamW 优化器（第4节）
4. 训练循环，支持模型和优化器状态的序列化与加载（第5节）

### 需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器
2. 用训练好的分词器编码数据集，将文本转换为整数 ID 序列
3. 在 TinyStories 数据集上训练 Transformer LM
4. 用训练好的 Transformer LM 生成样本并评估困惑度(perplexity)
5. 在 OpenWebText 上训练模型，并提交困惑度到排行榜

### 允许使用的内容

必须从零构建每个组件。不得使用 `torch.nn`、`torch.nn.functional`、`torch.optim` 中的任何定义，以下除外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）
- `torch.optim.Optimizer` 基类

可以使用任何其他 PyTorch 定义。

### 代码结构

- `cs336_basics/*`：在此编写代码（目录内无初始代码，从零开始）
- `adapters.py`：为每项功能填写适配器实现（如 `run_scaled_dot_product_attention`），仅作为胶水代码调用你的实现
- `test_*.py`：必须通过的所有测试，不要编辑测试文件

### 提交方式

- 运行 `make_submission.sh` 构建提交 zip 文件
- 提交到 Gradescope：`writeup.pdf`（书面回答）和 `code.zip`（代码）
- 排行榜提交：向 `github.com/stanford-cs336/assignment1-basics-leaderboard` 提交 PR

### 数据集

本作业使用两个预处理数据集：
- **TinyStories** [R. Eldan et al., 2023]
- **OpenWebText** [A. Gokaslan et al., 2019]

两者都是单一的大型纯文本文件。

---

## 2 Byte-Pair Encoding (BPE) 分词器

### 2.1 Unicode 标准

Unicode 是将字符映射到整数码点的文本编码标准。Python 中用 `ord()` 将单个 Unicode 字符转为整数，用 `chr()` 将整数码点转为字符。

**题目 (unicode1)：理解 Unicode（1分）**

(a) `chr(0)` 返回什么 Unicode 字符？
(b) 该字符的字符串表示 (`__repr__()`) 与打印表示有何不同？
(c) 当该字符出现在文本中时会发生什么？

### 2.2 Unicode 编码

直接在 Unicode 码点上训练分词器不实际（词表约15万项且稀疏）。我们使用 UTF-8 编码将 Unicode 字符转换为字节序列。UTF-8 是互联网上的主流编码（超过98%的网页使用）。

通过将码点转换为字节序列，我们将21位整数序列转化为0-255范围的字节值序列。256长度的字节词表更易管理，且不会出现词表外(OOV)的问题。

**题目 (unicode2)：Unicode 编码（3分）**

(a) 为什么更倾向于在 UTF-8 编码的字节上训练分词器，而非 UTF-16 或 UTF-32？
(b) 给出一个会导致 `decode_utf8_bytes_to_str_wrong` 函数产生错误结果的输入字节串示例，并解释为什么该函数不正确。
(c) 给出一个无法解码为任何 Unicode 字符的双字节序列。

### 2.3 子词分词 (Subword Tokenization)

子词分词是词级分词器和字节级分词器之间的折中。BPE 通过迭代替换（"合并"）最频繁的字节对来实现压缩，用更大的词表换取更好的输入压缩。

### 2.4 BPE 分词器训练

训练过程包含三个主要步骤：

#### 词表初始化

初始词表是所有256个可能的字节值。

#### 预分词 (Pre-tokenization)

使用正则表达式对语料进行粗粒度分词：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

使用 `re.finditer` 避免存储所有预分词结果。

#### 计算 BPE 合并

BPE 算法迭代地：
1. 统计每个字节对出现的频率
2. 找到频率最高的对
3. 合并该对为新 token，加入词表
4. 不考虑跨预分词边界的对
5. 频率相同时，选择字典序更大的对

#### 特殊 token

一些字符串（如 `<|endoftext|>`）作为特殊 token，编码时永远不被拆分。

**BPE 训练示例：** 给定语料 "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"，预分词后得到频率表 `{low: 5, lower: 2, widest: 3, newest: 6}`。合并序列为 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

### 2.5 BPE 分词器训练实验

**并行化预分词：** 可用 `multiprocessing` 加速。按 chunk 分割语料，确保边界在特殊 token 的开头。

**去除特殊 token：** 在运行预分词正则前，先将特殊 token 从语料中去除，以防止跨文档边界的合并。

**优化合并步骤：** 通过索引所有对的计数并增量更新（而非每次合并后重新遍历所有对）来加速。

**题目 (train_bpe)：BPE 分词器训练（15分）**

实现一个函数，给定输入文本文件路径，训练字节级 BPE 分词器。

输入参数：
- `input_path: str` — 训练数据文本文件路径
- `vocab_size: int` — 最大最终词表大小
- `special_tokens: list[str]` — 特殊 token 列表

输出：
- `vocab: dict[int, bytes]` — 词表映射（token ID → token 字节）
- `merges: list[tuple[bytes, bytes]]` — BPE 合并列表，按创建顺序排列

测试：实现 `adapters.run_train_bpe`，然后运行 `uv run pytest tests/test_train_bpe.py`

**题目 (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2分）**

(a) 在 TinyStories 数据集上训练词表大小为10,000的 BPE 分词器，使用 `<|endoftext|>` 作为特殊 token。报告训练时间、内存使用和词表中最长的 token。
   - 资源要求：≤30分钟（无GPU），≤30GB RAM
   - 提示：使用 multiprocessing 进行预分词并利用特殊 token 作为边界，应该能在2分钟内完成。
(b) 分析代码性能瓶颈。

**题目 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2分）**

(a) 在 OpenWebText 上训练词表大小为32,000的 BPE 分词器。报告最长 token。
   - 资源要求：≤12小时（无GPU），≤100GB RAM
(b) 比较 TinyStories 和 OpenWebText 上训练得到的分词器。

### 2.6 BPE 分词器：编码和解码

#### 2.6.1 编码文本

1. **预分词**：将输入文本预分词并表示为 UTF-8 字节序列
2. **应用合并**：按训练时的创建顺序应用合并操作

编码示例：输入 `'the cat ate'`，合并列表 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。预分词得到 `['the', ' cat', ' ate']`，最终编码为 `[9, 7, 1, 5, 10, 3]`。

#### 2.6.2 解码文本

查找每个 ID 对应的词表条目（字节序列），拼接后解码为 Unicode 字符串。无效字节用 Unicode 替换字符 U+FFFD 代替（使用 `errors='replace'`）。

**题目 (tokenizer)：实现分词器（15分）**

实现 `Tokenizer` 类，包含：
- `__init__(self, vocab, merges, special_tokens=None)` — 从词表和合并列表构建
- `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` — 从文件加载
- `encode(self, text: str) -> list[int]` — 编码文本为 token ID
- `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` — 惰性编码（内存高效）
- `decode(self, ids: list[int]) -> str` — 解码 token ID 为文本

测试：实现 `adapters.get_tokenizer`，然后运行 `uv run pytest tests/test_tokenizer.py`

### 2.7 实验

**题目 (tokenizer_experiments)：分词器实验（4分）**

(a) 从 TinyStories 和 OpenWebText 各取10个文档，用对应分词器编码，报告压缩比（bytes/token）。
(b) 用 TinyStories 分词器处理 OpenWebText 样本会怎样？
(c) 估计分词器吞吐量（bytes/second），处理 Pile 数据集（825GB）需要多久？
(d) 用分词器编码训练集和验证集，保存为 `uint16` 的 NumPy 数组。为什么 `uint16` 是合适的选择？

---

## 3 Transformer 语言模型架构

语言模型输入为形状 `(batch_size, sequence_length)` 的整数 token ID 张量，输出形状为 `(batch_size, sequence_length, vocab_size)` 的词表概率分布，预测每个位置的下一个 token。

### 3.1 Transformer LM 整体结构

1. **Token 嵌入**：将 token ID 转换为密集向量
2. **num_layers 个 Transformer 块**
3. **最终 RMSNorm**
4. **线性投影（LM head）**：输出 next-token logits

### 3.2 批处理、Einsum 和高效计算

强烈推荐使用 einsum 表示法（`einops` 或 `einx` 库）。好处：
- 自文档化，清晰表达张量维度
- 可处理任意批维度
- 几乎总是性能更优

注意：本作业数学符号使用列向量，但 PyTorch 使用行主序。使用 einsum 可避免此问题。

### 3.3 基本构建模块

#### 参数初始化

- 线性层权重：截断正态分布 N(0, 2/(d_in+d_out))，截断在 [-3σ, 3σ]
- 嵌入：截断正态分布 N(0, 1)，截断在 [-3, 3]
- RMSNorm：全1初始化
- 使用 `torch.nn.init.trunc_normal_` 初始化

#### 题目 (linear)：实现线性模块（1分）

实现 `Linear` 类（继承 `nn.Module`），执行 y = Wx（无偏置）。
- 不得使用 `nn.Linear` 或 `nn.functional.linear`
- 测试：实现 `adapters.run_linear`，运行 `uv run pytest -k test_linear`

#### 题目 (embedding)：实现嵌入模块（1分）

实现 `Embedding` 类（继承 `nn.Module`），通过索引嵌入矩阵查找 token 向量。
- 不得使用 `nn.Embedding` 或 `nn.functional.embedding`
- 测试：实现 `adapters.run_embedding`，运行 `uv run pytest -k test_embedding`

### 3.4 Pre-Norm Transformer 块

每个块包含两个子层：多头自注意力 + 位置前馈网络。使用"pre-norm"结构（在子层输入处进行归一化）。

#### 3.4.1 题目 (rmsnorm)：RMS 层归一化（1分）

实现 RMSNorm：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} \cdot g_i$$

其中 RMS(a) = sqrt(1/d_model * Σa_i² + ε)，ε 通常为 1e-5。

注意：需将输入上转为 `torch.float32` 以防止溢出，计算完成后转回原始 dtype。

测试：实现 `adapters.run_rmsnorm`，运行 `uv run pytest -k test_rmsnorm`

#### 3.4.2 题目 (positionwise_feedforward)：位置前馈网络（2分）

实现 SwiGLU 前馈网络：

$$\text{FFN}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$$

其中 SiLU(x) = x · σ(x)，d_ff ≈ 8/3 × d_model（取64的倍数）。

可以使用 `torch.sigmoid`。

测试：实现 `adapters.run_swiglu`，运行 `uv run pytest -k test_swiglu`

#### 3.4.3 题目 (rope)：旋转位置嵌入 RoPE（2分）

实现 Rotary Position Embeddings。对 query 和 key 向量的每对元素应用旋转矩阵：

$$R_k^i = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}$$

其中 θ_{i,k} = i / Θ^{(2k)/d}。

接口：
- `__init__(self, theta, d_k, max_seq_len, device=None)`
- `forward(self, x, token_positions)` — 输入 `(..., seq_len, d_k)`，输出同形

测试：实现 `adapters.run_rope`，运行 `uv run pytest -k test_rope`

#### 3.4.4 题目 (softmax)：实现 softmax（1分）

实现带数值稳定性的 softmax（减去最大值技巧）。

测试：实现 `adapters.run_softmax`，运行 `uv run pytest -k test_softmax_matches_pytorch`

#### 3.4.4 题目 (scaled_dot_product_attention)：缩放点积注意力（5分）

实现：Attention(Q, K, V) = softmax(QK^T / √d_k) V

支持：
- Q, K 形状 `(batch_size, ..., seq_len, d_k)`
- V 形状 `(batch_size, ..., seq_len, d_v)`
- 可选布尔 mask `(seq_len, seq_len)`
- 输出 `(batch_size, ..., seq_len, d_v)`

测试：实现 `adapters.run_scaled_dot_product_attention`，运行相应 pytest

#### 3.4.5 题目 (multihead_self_attention)：因果多头自注意力（5分）

实现：
$$\text{MultiHeadSelfAttention}(x) = W_O \cdot \text{MultiHead}(W_Q x, W_K x, W_V x)$$

其中 d_k = d_v = d_model / h。

要求：
- **因果掩码**：token i 只能注意位置 j ≤ i
- **应用 RoPE**：对 query 和 key 应用（不对 value 应用），head 维度作为 batch 维度

测试：实现 `adapters.run_multihead_self_attention`，运行 `uv run pytest -k test_multihead_self_attention`

### 3.5 完整 Transformer LM

#### 题目 (transformer_block)：实现 Transformer 块（3分）

实现 pre-norm Transformer 块：
- y = x + MultiHeadSelfAttention(RMSNorm(x))
- output = y + FFN(RMSNorm(y))

参数：`d_model`, `num_heads`, `d_ff`

测试：实现 `adapters.run_transformer_block`，运行 `uv run pytest -k test_transformer_block`

#### 题目 (transformer_lm)：实现 Transformer LM（3分）

组装完整模型：嵌入 → num_layers 个 Transformer 块 → 最终 RMSNorm → LM head

额外参数：`vocab_size`, `context_length`, `num_layers`

测试：实现 `adapters.run_transformer_lm`，运行 `uv run pytest -k test_transformer_lm`

#### 题目 (transformer_accounting)：Transformer 资源核算（5分）

GPT-2 XL 配置：vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288

(a) 模型有多少可训练参数？加载需要多少内存？
(b) 一次前向传播的矩阵乘法需要多少 FLOPs？
(c) 哪些部分需要最多 FLOPs？
(d) 对 GPT-2 small/medium/large 重复分析。随模型增大，各部分 FLOPs 比例如何变化？
(e) 将上下文长度增加到16384，FLOPs 如何变化？

---

## 4 训练 Transformer LM

### 4.1 题目 (cross_entropy)：交叉熵损失（1分）

实现交叉熵损失：ℓ_i = -log softmax(o_i)[x_{i+1}]

要求：减去最大值保证数值稳定；尽可能抵消 log 和 exp；处理额外 batch 维度并返回平均值。

测试：实现 `adapters.run_cross_entropy`，运行 `uv run pytest -k test_cross_entropy`

**困惑度：** perplexity = exp(平均交叉熵损失)

### 4.2 SGD 优化器

**题目 (learning_rate_tuning)：调整学习率（1分）**

用学习率 1e1, 1e2, 1e3 运行 SGD 示例10次迭代，观察损失行为。

### 4.3 题目 (adamw)：实现 AdamW（2分）

AdamW 算法：
1. 初始化 m=0, v=0
2. 对每步 t=1,...,T：
   - 计算梯度 g
   - α_t = α · √(1-β₂ᵗ) / (1-β₁ᵗ)
   - θ ← θ - α·λ·θ （权重衰减）
   - m ← β₁·m + (1-β₁)·g
   - v ← β₂·v + (1-β₂)·g²
   - θ ← θ - α_t · m / (√v + ε)

实现 `adapters.get_adamw_cls`，运行 `uv run pytest -k test_adamw`

**题目 (adamw_accounting)：AdamW 资源核算（2分）**

(a) 使用 AdamW 训练的峰值内存（分解为参数/激活/梯度/优化器状态）
(b) GPT-2 XL 在80GB内存下的最大 batch size
(c) 一步 AdamW 需要多少 FLOPs
(d) 使用50% MFU，在单个 H100 上训练 GPT-2 XL 400K步需要多久？（H100 峰值495 TFLOP/s）

### 4.4 题目 (learning_rate_schedule)：余弦学习率调度（1分）

实现带 warmup 的余弦退火调度：
- t < T_w 时：α_t = (t/T_w) · α_max（线性预热）
- T_w ≤ t ≤ T_c 时：α_t = α_min + 0.5·(1+cos((t-T_w)/(T_c-T_w)·π))·(α_max-α_min)
- t > T_c 时：α_t = α_min

测试：实现 `adapters.get_lr_cosine_schedule`，运行 `uv run pytest -k test_get_lr_cosine_schedule`

### 4.5 题目 (gradient_clipping)：梯度裁剪（1分）

当梯度 ℓ₂ 范数超过最大值 M 时，按 M/(‖g‖₂+ε) 缩放梯度（ε=1e-6）。

测试：实现 `adapters.run_gradient_clipping`，运行 `uv run pytest -k test_gradient_clipping`

---

## 5 训练循环

### 5.1 题目 (data_loading)：数据加载（2分）

实现从 token ID 的 numpy 数组中采样批次的函数：
- 输入：numpy 数组、batch_size、context_length、device
- 输出：(inputs, targets) 两个形状为 `(batch_size, context_length)` 的张量

使用 `np.memmap` 进行内存映射加载大数据集。

测试：实现 `adapters.run_get_batch`，运行 `uv run pytest -k test_get_batch`

### 5.2 题目 (checkpointing)：模型检查点（1分）

实现：
- `save_checkpoint(model, optimizer, iteration, out)` — 保存模型、优化器状态和迭代数
- `load_checkpoint(src, model, optimizer)` — 加载检查点并恢复状态，返回迭代数

测试：实现 `adapters.run_save_checkpoint` 和 `adapters.run_load_checkpoint`，运行 `uv run pytest -k test_checkpointing`

### 5.3 题目 (training_together)：组合训练循环（4分）

编写训练脚本，支持：
- 配置模型和优化器超参数
- 用 `np.memmap` 内存高效加载大数据集
- 将检查点序列化到用户指定路径
- 定期记录训练和验证性能

---

## 6 生成文本

### 题目 (decoding)：解码（3分）

实现从语言模型生成文本的函数，支持：
- 给定 prompt 生成补全（直到 `<|endoftext|>` 或最大 token 数）
- **温度缩放**：softmax(v/τ)，τ→0 时趋向 one-hot
- **Top-p（核）采样**：截断低概率 token，只保留累积概率 ≥ p 的最小集合 V(p)

---

## 7 实验

### 7.1 题目 (experiment_log)：实验日志（3分）

创建实验追踪基础设施，记录实验和损失曲线（步数和时间）。

### 7.2 TinyStories 实验

#### 基础超参数

| 参数 | 值 |
|------|-----|
| Vocab size | 10000 |
| Context length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| RoPE theta (Θ) | 10000 |
| 层数 | 4 |
| 注意力头数 | 16 |
| 总处理 token 数 | 327,680,000 |

需自行调优：学习率、warmup、AdamW 参数（β₁, β₂, ε）、权重衰减。

#### 题目 (learning_rate)：调优学习率（2 B200 hrs）（3分）

(a) 对学习率进行超参数搜索，报告最终损失或发散情况。
   - 交付：多个学习率的学习曲线 + 验证损失 ≤ 1.45 的模型
(b) 研究学习率发散点与最佳学习率的关系。

#### 题目 (batch_size_experiment)：批大小实验（1 B200 hr）（1分）

从 batch_size=1 到 GPU 内存上限变化 batch size。

#### 题目 (generate)：生成文本（1分）

用训练好的模型生成至少256个 token 的文本，讨论流畅度。

### 7.3 消融实验和架构修改

#### 题目 (layer_norm_ablation)：移除 RMSNorm（0.5 B200 hrs）（1分）

移除所有 RMSNorm 并训练，观察效果。能否通过降低学习率获得稳定性？

#### 题目 (pre_norm_ablation)：Post-norm 实验（0.5 B200 hrs）（1分）

将 pre-norm 改为 post-norm：
- Pre-norm: z = x + MHSA(RMSNorm(x))
- Post-norm: z = RMSNorm(x + MHSA(x))

#### 题目 (no_pos_emb)：无位置嵌入 NoPE（0.5 B200 hrs）（1分）

移除 RoPE，比较与有 RoPE 时的性能。

#### 题目 (swiglu_ablation)：SwiGLU vs. SiLU（0.5 B200 hrs）（1分）

比较 SwiGLU 和纯 SiLU 前馈网络（无门控）的性能：
- SiLU 版本：FFN(x) = W₂ SiLU(W₁x)，d_ff = 4 × d_model（近似匹配参数量）

### 7.4 题目 (main_experiment)：OpenWebText 实验（2 B200 hrs）（2分）

使用相同模型架构和训练迭代在 OpenWebText 上训练。报告学习曲线和生成文本，讨论与 TinyStories 的差异。

### 7.5 题目 (leaderboard)：排行榜（10 B200 hrs）（6分）

规则：
- 运行时间 ≤ 45分钟（在 B200 上）
- 只能使用提供的 OpenWebText 训练数据
- 其他不限

目标：最小化验证损失，至少低于5.0的基线。

改进建议：
- 参考 Llama 3、Qwen 2.5 等架构
- 参考 NanoGPT speedrun 仓库的修改（如权重绑定等）

提交到：`github.com/stanford-cs336/assignment1-basics-leaderboard`

---

---

## 附录：全部子任务清单

### 汇总统计

| 类别 | 题目数 | 总分 |
|------|--------|------|
| 代码题（有自动测试） | 14 | 58分 |
| 书面/实验题 | 15 | 47分 |
| **总计** | **29** | **105分** |

---

### 第2节：BPE 分词器

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 1 | unicode1 | 1 | 书面 | 回答3个关于 `chr(0)` 的问题 | writeup.pdf 中每题一句话 |
| 2 | unicode2 | 3 | 书面 | (a) 为什么选 UTF-8 (b) 错误解码函数反例 (c) 无法解码的双字节序列 | writeup.pdf 中每题一两句话 |
| 3 | train_bpe | 15 | 代码 | 实现 BPE 训练函数（输入: input_path, vocab_size, special_tokens；输出: vocab, merges） | 适配器: `adapters.run_train_bpe`<br>测试: `uv run pytest tests/test_train_bpe.py` |
| 4 | train_bpe_tinystories | 2 | 实验+书面 | (a) TinyStories 上训练 vocab_size=10000，报告时间/内存/最长token (b) 性能瓶颈分析 | 资源限制: ≤30min, ≤30GB RAM |
| 5 | train_bpe_expts_owt | 2 | 实验+书面 | (a) OpenWebText 上训练 vocab_size=32000 (b) 对比两个分词器 | 资源限制: ≤12hrs, ≤100GB RAM |
| 6 | tokenizer | 15 | 代码 | 实现 Tokenizer 类（`__init__`, `from_files`, `encode`, `encode_iterable`, `decode`） | 适配器: `adapters.get_tokenizer`<br>测试: `uv run pytest tests/test_tokenizer.py` |
| 7 | tokenizer_experiments | 4 | 实验+书面 | (a) 压缩比 (b) 跨数据集对比 (c) 吞吐量估计 (d) 编码为 uint16 numpy | writeup.pdf 中每题一两句话 |

### 第3节：Transformer 模型架构

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 8 | linear | 1 | 代码 | 实现 Linear 类（y=Wx，无偏置），初始化: 截断正态 N(0, 2/(d_in+d_out)) | 适配器: `adapters.run_linear`<br>测试: `uv run pytest -k test_linear` |
| 9 | embedding | 1 | 代码 | 实现 Embedding 类，初始化: 截断正态 N(0,1) 截断在[-3,3] | 适配器: `adapters.run_embedding`<br>测试: `uv run pytest -k test_embedding` |
| 10 | rmsnorm | 1 | 代码 | 实现 RMSNorm（上转 float32 计算后转回原 dtype） | 适配器: `adapters.run_rmsnorm`<br>测试: `uv run pytest -k test_rmsnorm` |
| 11 | positionwise_feedforward | 2 | 代码 | 实现 SwiGLU: FFN(x) = W₂(SiLU(W₁x) ⊙ W₃x)，d_ff ≈ 8/3 × d_model 取64倍数 | 适配器: `adapters.run_swiglu`<br>测试: `uv run pytest -k test_swiglu` |
| 12 | rope | 2 | 代码 | 实现 RotaryPositionalEmbedding（对 Q/K 向量施加旋转） | 适配器: `adapters.run_rope`<br>测试: `uv run pytest -k test_rope` |
| 13 | softmax | 1 | 代码 | 实现带数值稳定性的 softmax（减最大值技巧） | 适配器: `adapters.run_softmax`<br>测试: `uv run pytest -k test_softmax_matches_pytorch` |
| 14 | scaled_dot_product_attention | 5 | 代码 | 实现缩放点积注意力，支持任意 batch 维度和可选 mask | 适配器: `adapters.run_scaled_dot_product_attention`<br>测试: `uv run pytest -k test_scaled_dot_product_attention` + `test_4d_scaled_dot_product_attention` |
| 15 | multihead_self_attention | 5 | 代码 | 实现因果多头自注意力（带 RoPE + causal mask），d_k=d_v=d_model/h | 适配器: `adapters.run_multihead_self_attention`<br>测试: `uv run pytest -k test_multihead_self_attention` |
| 16 | transformer_block | 3 | 代码 | 组装 pre-norm 块: y=x+MHSA(Norm(x)); out=y+FFN(Norm(y)) | 适配器: `adapters.run_transformer_block`<br>测试: `uv run pytest -k test_transformer_block` |
| 17 | transformer_lm | 3 | 代码 | 组装完整模型: Embedding → N×Block → Norm → Linear | 适配器: `adapters.run_transformer_lm`<br>测试: `uv run pytest -k test_transformer_lm` |
| 18 | transformer_accounting | 5 | 书面 | GPT-2 XL 参数量/FLOPs 计算，不同模型大小和上下文长度对比 | writeup.pdf 中代数表达式+解释 |

### 第4节：训练组件

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 19 | cross_entropy | 1 | 代码 | 实现交叉熵损失（减最大值 + 抵消 log/exp + batch 均值） | 适配器: `adapters.run_cross_entropy`<br>测试: `uv run pytest -k test_cross_entropy` |
| 20 | learning_rate_tuning | 1 | 书面 | 用 lr=1e1, 1e2, 1e3 跑 SGD 示例10步，观察损失 | writeup.pdf 中一两句话 |
| 21 | adamw | 2 | 代码 | 实现 AdamW 优化器（继承 torch.optim.Optimizer） | 适配器: `adapters.get_adamw_cls`<br>测试: `uv run pytest -k test_adamw` |
| 22 | adamw_accounting | 2 | 书面 | (a) 峰值内存分解 (b) 最大 batch size (c) AdamW FLOPs (d) H100 训练时间 | writeup.pdf 中代数表达式 |
| 23 | learning_rate_schedule | 1 | 代码 | 实现余弦退火+warmup 学习率调度 | 适配器: `adapters.get_lr_cosine_schedule`<br>测试: `uv run pytest -k test_get_lr_cosine_schedule` |
| 24 | gradient_clipping | 1 | 代码 | 实现梯度裁剪（‖g‖₂ > M 时缩放，ε=1e-6） | 适配器: `adapters.run_gradient_clipping`<br>测试: `uv run pytest -k test_gradient_clipping` |

### 第5节：训练循环

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 25 | data_loading | 2 | 代码 | 实现 batch 采样函数，返回 (inputs, targets) 形状 (B, context_len) | 适配器: `adapters.run_get_batch`<br>测试: `uv run pytest -k test_get_batch` |
| 26 | checkpointing | 1 | 代码 | 实现 save_checkpoint 和 load_checkpoint | 适配器: `adapters.run_save_checkpoint` + `adapters.run_load_checkpoint`<br>测试: `uv run pytest -k test_checkpointing` |
| 27 | training_together | 4 | 代码 | 编写完整训练脚本（超参数配置、memmap、检查点、日志） | 无自动测试，靠后续实验验证 |

### 第6节：生成文本

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 28 | decoding | 3 | 代码 | 实现文本生成（prompt续写 + 温度缩放 + top-p 采样） | 无自动测试，靠生成结果验证 |

### 第7节：实验

| # | 题目ID | 分值 | 类型 | 任务描述 | 验收标准 |
|---|--------|------|------|----------|----------|
| 29 | experiment_log | 3 | 代码+文档 | 创建实验追踪基础设施 | 日志代码 + 实验记录文档 |
| 30 | learning_rate | 3 | GPU实验 | (a) 学习率搜索 (b) 发散边界分析 | 学习曲线 + **验证损失 ≤ 1.45 的模型**<br>GPU: 2 B200 hrs |
| 31 | batch_size_experiment | 1 | GPU实验 | batch_size 从1到GPU上限的实验 | 多条学习曲线 + 讨论<br>GPU: 1 B200 hr |
| 32 | generate | 1 | 实验 | 生成 ≥256 token 文本 | 文本输出 + 流畅度评论 |
| 33 | layer_norm_ablation | 1 | GPU实验 | 移除所有 RMSNorm 训练 | 学习曲线 + 评论<br>GPU: 0.5 B200 hr |
| 34 | pre_norm_ablation | 1 | GPU实验 | 改为 post-norm 训练并对比 | 学习曲线对比<br>GPU: 0.5 B200 hr |
| 35 | no_pos_emb | 1 | GPU实验 | 移除 RoPE 并对比 | 学习曲线对比<br>GPU: 0.5 B200 hr |
| 36 | swiglu_ablation | 1 | GPU实验 | SwiGLU vs 纯SiLU（d_ff=4×d_model） | 学习曲线 + 讨论<br>GPU: 0.5 B200 hr |
| 37 | main_experiment | 2 | GPU实验 | 在 OpenWebText 上训练并对比 TinyStories | 学习曲线 + 生成文本 + 讨论<br>GPU: 2 B200 hrs |
| 38 | leaderboard | 6 | GPU实验 | 自由优化，最小化 OWT 验证损失 | 验证损失 < 5.0 + 学习曲线 + 描述<br>约束: ≤45min B200, 只用提供的 OWT 数据<br>GPU: 10 B200 hrs |

---

## 参考文献

主要参考论文包括：Attention is All You Need (Vaswani et al., 2017)、GPT-2/3、LLaMA、PaLM、BPE (Sennrich et al., 2016)、RoPE (Su et al., 2021)、AdamW (Loshchilov & Hutter, 2019)、GLU Variants (Shazeer, 2020) 等。
