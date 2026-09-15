# DB2+DB3 训练、DB4 测试：简单 MLP

这是一个用来检查数据和实验流程的简单 MLP 基线，不是论文里的 AEMG、Transformer 或 VQ/MEM 模型。

## 主实验协议

按照学长的建议，主实验是：

- 训练集：NinaPro DB2 全部 40 个 subject + DB3 全部 11 个 subject
- 测试集：NinaPro DB4 全部 10 个 subject
- DB2、DB3 不包含在 DB4 测试集中
- 三个数据集都使用 Exercise 1 的 gesture 1-10
- gesture 1-10 映射为分类标签 0-9
- 12 个 EMG 通道，2 kHz 重采样到 200 Hz
- 250 ms 窗口，50 ms 步长

主流程是跨数据集测试：用 DB2+DB3 训练一个有监督 MLP，然后直接在 DB4 上测试。

## 代码结构

```text
reproduction_mlp/
├── config_mlp.yaml
├── mlp_baseline/
│   ├── data.py
│   ├── db4.py                  # 旧的 DB4 单独读取接口
│   ├── loaders.py
│   ├── model.py                # 简单 MLP
│   ├── nct.py                  # 窗口化和能量阈值
│   └── ninapro.py              # DB2/DB3/DB4 统一读取
└── scripts/
    ├── prepare_db4.py          # 旧的 DB4-only 流程
    ├── prepare_db23_to_db4.py  # 准备 DB2、DB3 训练和 DB4 测试数据
    ├── run_mlp_db23_to_db4.py  # 主流程
    └── run_mlp_loso.py         # 旧的 DB4 LOSO 对照流程
```

## 数据处理

官方 zip 不能直接被训练脚本读取，需要先解压成 `.mat` 文件目录。DB2、DB3 的完整数据目录应类似：

```text
ninapro_db2_full/DB2_s1/S1_E1_A1.mat
ninapro_db3_full/DB3_s1/S1_E1_A1.mat
ninapro_db4/s1/S1_E1_A1.mat
```

然后运行：

```bash
python scripts/prepare_db23_to_db4.py \
  --db2-root /你的路径/ninapro_db2_full \
  --db3-root /你的路径/ninapro_db3_full \
  --db4-root /你的路径/ninapro_db4
```

处理后的文件会放在：

```text
data/db23_to_db4/nct/train/db2/
data/db23_to_db4/nct/train/db3/
data/db23_to_db4/nct/test/db4/
```

脚本会检查每个原始 subject 是否包含 10 个目标类别、12 个通道和有效 rest 信号。处理后的数据还会生成 `manifest.json`。

## 训练

```bash
python scripts/run_mlp_db23_to_db4.py --epochs 100
```

结果保存到：

```text
results/db23_to_db4.json
```

结果中的 `db4_overall.accuracy` 是 DB2+DB3 训练后在 DB4 上的总体准确率。

## 当前结果

当前完整实验使用 2887 个 DB2+DB3 训练样本，在 593 个 DB4 测试样本上得到约 9.11%。训练集准确率为 100%。

这个结果说明简单 MLP 能记住训练数据，但跨数据集泛化很差。它不能直接与论文中使用 AEMG 预训练和目标数据微调后的 80% 左右结果比较。

## 数据和 GitHub

原始数据和处理后的 `.npz` 文件不上传 GitHub。GitHub 只保存代码、配置、README 和实验结果说明。
