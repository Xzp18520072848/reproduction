# DB2+DB3 自监督预训练、DB4 LOSO 测试：简单 MLP

这是一个用于验证数据和实验思路的简单模型版本，不是论文中的 AEMG、Transformer 或 VQ/MEM 模型。

## 实验流程

1. **DB2+DB3 只做 SSL 预训练**：不使用 gesture label。训练脚本只读取 NCT 后的 `windows`、`time_positions` 和 `attention_masks`。
2. 对每个样本随机 mask 50% 的有效 token，用 MLP encoder 加 decoder，根据被 mask token 的重建误差训练。
3. DB2+DB3 的 subject 文件按 9:1 划分预训练 train/val，根据 val reconstruction loss 保存 `checkpoints/ssl_best_encoder.pt`。
4. **DB4 做 10-fold LOSO**：每折留一个 subject 做 test；剩下 9 个中留一个做 validation，另外 8 个做训练。
5. 加载并冻结 SSL encoder，只训练 DB4 的 linear classification head；根据 validation accuracy 保存每折最佳 head，再评估 held-out test subject。
6. 最后报告 10 个 test subject 的 mean accuracy 和整体 accuracy。

这里的“自监督”指预训练损失不使用 gesture label。原始数据预处理阶段仍需要使用 `restimulus`/`rerepetition` 定位动作片段；预训练真正开始后，label 不会被读取。

## 代码结构

```text
reproduction_mlp/
├── config_mlp.yaml
├── mlp_baseline/
│   ├── data.py
│   ├── db4.py                  # 旧的 DB4 单独读取接口
│   ├── loaders.py              # 有标签 DB4 loader 和无标签 SSL loader
│   ├── model.py                # MLP encoder、SSL autoencoder、旧接口
│   ├── nct.py                  # 窗口化和能量阈值
│   └── ninapro.py              # DB2/DB3/DB4 统一读取
└── scripts/
    ├── prepare_db4.py          # 旧的 DB4-only 流程
    ├── prepare_db23_to_db4.py  # 准备 DB2、DB3、DB4 的 NCT 数据
    ├── run_mlp_db23_to_db4.py  # 当前主流程：SSL + DB4 LOSO
    └── run_mlp_loso.py         # 旧的 DB4 LOSO 对照流程
```

## 数据处理

官方 zip 不能直接被训练脚本读取，需要先解压成 `.mat` 文件目录。然后运行：

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

当前配置使用 Exercise 1 的 gesture 1-10、12 个通道、200 Hz、250 ms 窗口和 50 ms 步长。原始数据和处理后的 `.npz` 不上传 GitHub。

## 运行

在仓库根目录运行：

```bash
python scripts/run_mlp_db23_to_db4.py
```

默认配置在 `config_mlp.yaml`：预训练和分类头各 100 个 epoch，每 2 个 epoch 打印一次 train/val 效果。也可以临时缩短测试：

```bash
python scripts/run_mlp_db23_to_db4.py \
  --pretrain-epochs 2 \
  --head-epochs 2 \
  --print-every 1
```

输出文件：

```text
checkpoints/ssl_best_encoder.pt
checkpoints/pretrain_split.json
checkpoints/db4_loso/fold_XXX_best_head.pt
results/ssl_db4_loso/summary.json
```

`summary.json` 中的 `mean_accuracy` 是 10 个 held-out DB4 subject accuracy 的平均值，`overall_accuracy` 是把 10 折测试样本合并后的总体准确率。

## 旧的监督 MLP 结果

仓库之前曾经有一个“DB2+DB3 有监督训练、DB4 直接测试”的基线，AutoDL 上约为 9.11%。它不是当前 SSL + LOSO 流程的结果，也不能和论文的完整 AEMG 预训练/适配结果直接比较。当前新流程跑完后，以 `results/ssl_db4_loso/summary.json` 为准。
