# 实验结果

## 当前实验协议

当前主流程是：

```text
DB2+DB3-SSL-pretrain-DB4-LOSO-linear-probe
```

- DB2+DB3：只用于自监督预训练，不使用 gesture label
- 预训练：NCT token 随机 mask 50%，MLP encoder + decoder 重建被 mask token
- 预训练划分：DB2+DB3 subject 文件按 9:1 划分 train/val
- DB4：10 个 subject 做 LOSO
- 每折 DB4：1 个 test、1 个 validation、8 个 train
- 分类阶段：冻结 pretrained encoder，只训练 linear classification head
- 模型选择：encoder 按 validation reconstruction loss，classification head 按 validation accuracy

## 数据规模

当前 NCT 文件统计：

- DB2：40 个 subject，2290 个窗口样本
- DB3：11 个 subject，597 个窗口样本
- DB2+DB3：51 个 subject 文件，2887 个窗口样本
- DB4：10 个 subject，593 个窗口样本
- 类别数：10
- 通道数：12
- token 输入窗口：12×50

预训练阶段虽然 NCT 文件内部保留了 label 字段用于后续检查和 DB4 分类，但 SSL loader 只读取 `windows`、`time_positions`、`attention_masks`，不会读取 label。

## 结果文件

完整新流程运行后，结果保存为：

```text
results/ssl_db4_loso/summary.json
```

其中：

- `pretraining.best_val_loss`：预训练最佳验证重建损失
- `fold_results[*].best_validation_accuracy`：每折最佳分类验证准确率
- `fold_results[*].test.accuracy`：每个 held-out DB4 subject 的测试准确率
- `mean_accuracy`：10 折测试准确率的平均值
- `overall_accuracy`：10 折测试样本合并后的总体准确率

当前还没有把新 SSL + LOSO 流程的最终 accuracy 写死在这里，避免把旧监督 MLP 的 9.11% 误认为新实验结果。以实际生成的 `summary.json` 为准。

## 运行命令

```bash
python scripts/run_mlp_db23_to_db4.py
```

默认每 2 个 epoch 打印一次训练和验证效果，方便判断是否继续训练。短跑验证命令：

```bash
python scripts/run_mlp_db23_to_db4.py \
  --pretrain-epochs 2 \
  --head-epochs 2 \
  --print-every 1
```
