# iTP-AGN-XGB

[English](README.en.md)

## 项目简介

基于多特征表示与加性高斯噪声增强的治疗肽序列识别方法。

本项目是论文配套的简洁 Python 实现，用于治疗肽序列识别。流程包括 FASTA 数据读取、8 类人工特征提取、特征拼接、标准化、可选加性高斯噪声增强，以及 XGBoost 或软投票 XGBoost 分类。

## 功能

- 支持带标签的 FASTA 输入，例如 `seq_001|1`。
- 提取 8 类人工特征：AAC、DPC、ASDC、AAIndex、PAAC、APAAC、CTD 和 BLOSUM62。
- 支持多特征横向拼接与标准化。
- 支持训练集加性高斯噪声增强。
- 支持单个 XGBoost 模型和软投票集成模型。
- 输出二分类指标：ACC、SEN/Recall、SPE、PRE、F1、MCC、ROC-AUC 和 PR-AUC。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py --train Exampledata/train.fa --test Exampledata/test.fa --use_agn --model xgb
python main.py --train Exampledata/train.fa --test Exampledata/test.fa --use_agn --model soft_voting --n_models 3
```

预测结果默认保存到 `predictions.csv`。可使用 `--output` 指定其他输出路径。

## FASTA 格式

每条序列标题行必须包含序列 ID 和二分类标签：

```text
>seq_001|1
KLLKLLKKLLKLLK
>seq_002|0
AGVTSLAGTAA
```

`1` 表示治疗肽正样本，`0` 表示非治疗肽负样本。

## 命令行参数

- `--train`：训练集 FASTA 路径。
- `--test`：测试集 FASTA 路径。
- `--use_agn`：是否启用加性高斯噪声增强。
- `--sigma`：高斯噪声标准差，默认 `0.01`。
- `--n_aug`：每个训练样本生成的增强样本数，默认 `9`。
- `--model`：模型类型，可选 `xgb` 或 `soft_voting`。
- `--n_models`：软投票基学习器数量，默认 `5`。
- `--output`：预测结果 CSV 保存路径，默认 `predictions.csv`。

## 项目结构

```text
iTP-AGN-XGB/
├── README.md
├── README.zh-CN.md
├── README.en.md
├── requirements.txt
├── .gitignore
├── Exampledata/
│   ├── train.fa
│   └── test.fa
├── preprocessing/
├── src/
└── main.py
```

