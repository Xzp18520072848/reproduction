# DB4 肌电手势识别：简单 MLP 基线

这是把原来的复杂 AEMG/Transformer 模型换成简单 MLP 后的独立版本。

这个仓库的目的很简单：先确认 DB4 数据、标签和测试划分能不能让一个普通模型学到东西。它不是论文 AEMG 模型的复现结果，也没有使用 VQ、MEM、Transformer 或论文里的 8 个预训练数据集。

## 代码怎么做

每个 DB4 样本先被切成多个肌电窗口：

- 采样率：200 Hz
- 窗口长度：250 ms，也就是 50 个采样点
- 步长：50 ms
- 通道数：12
- 手势类别：1 到 10

MLP 的流程是：

1. 把每个窗口的 `12×50` 个数展平成 600 维；
2. 用一个小 MLP 提取窗口特征；
3. 把一个样本里的有效窗口取平均；
4. 输出 10 个手势类别。

## 数据放在哪里

由于 Ninapro DB4 原始数据不放进 GitHub，请自己准备原始数据。原始文件应类似：

```text
ninapro_db4/
├── s1/S1_E1_A1.mat
├── s2/S2_E1_A1.mat
└── ...
```

先生成 NCT 数据：

```bash
python scripts/prepare_db4.py --raw-root /你的路径/ninapro_db4
```

处理后会生成：

```text
data/db4/
├── raw/
└── nct/
    ├── subject_001.npz
    └── ...
```

## 怎么运行

在项目目录下运行一折，例如留出 subject 1：

```bash
python scripts/run_mlp_loso.py --held-out 1 --epochs 100
```

全部 10 折可以依次运行：

```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/run_mlp_loso.py --held-out $s --epochs 100
done
```

每一折结果会保存到 `results/fold_XXX.json`。

## 当前跑出来的结果

当前使用 DB4 的 10 个 subject 做 LOSO 测试，MLP 每一折训练集准确率都是 100%，但测试集平均准确率只有约 11.8%。详细结果见 [RESULTS.md](RESULTS.md)。

这说明模型记住了训练 subject，但不能很好地识别没见过的 subject。这个结果不能直接说明 DB4 数据坏了，还需要继续检查 subject 差异、阈值过滤和预处理方式。
