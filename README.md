# 多模态纠错实验 - 第一批：最小闭环

## 项目目标

跑通 **Data → Model → Loss → Generation → Metric** 全链路，验证代码正确性。  
不追求指标，只要结果合理即可。

## 项目结构

```
multimodal_correct/
├── configs/
│   ├── __init__.py
│   └── config.py              # 实验配置（超参数、路径）
├── data/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset & DataLoader
│   ├── flickr30k-images/      # 图片目录（真实/dummy）
│   ├── flickr30k-entities/    # Flickr30k 原始标注
│   └── cache/                 # 生成的纠错 JSON 数据
├── models/
│   ├── __init__.py
│   └── model.py               # CLIP ViT + GPT2 多模态纠错模型
├── scripts/
│   ├── prepare_data.py        # 数据准备（规则替换构造纠错对）
│   └── sanity_check.py        # Overfit 单 batch 验证代码正确性
├── logs/                      # 训练日志
├── outputs/                   # 模型 checkpoint & 评估结果
├── train.py                   # 主训练脚本
├── eval.py                    # 评估脚本（BLEU/ROUGE/纠错准确率）
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 模型架构

```
Image (224×224) ──▶ CLIP ViT (frozen) ──▶ Projection ──┐
                                                        ▼
Prompt (text)   ──▶ GPT2 Embedding ─────▶ Concat [img;text]
                                                        ▼
                                                   GPT2 Decoder ──▶ 纠正文本
```

- **Vision Encoder**: `openai/clip-vit-base-patch32`（冻结）
- **Text Decoder**: `gpt2`（可训练）
- **Projection**: 2层MLP（可训练）

## 快速开始

### 1. 安装依赖

```bash
cd multimodal_correct
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 如果有 Flickr30k-Entities 真实数据，放到 data/ 对应目录
# 如果没有，脚本会自动生成 dummy 数据（用于跑通链路）
python scripts/prepare_data.py
```

### 3. Sanity Check（推荐先跑）

```bash
# Overfit 单 batch，验证全链路
python scripts/sanity_check.py
```

预期结果：
- Loss 逐步降到接近 0
- 生成文本逐渐逼近目标文本

### 4. 训练

```bash
python train.py
```

训练过程会：
- 每 10 步打印 loss
- 每个 epoch 在验证集上评估并打印 case study
- 自动保存 best / final checkpoint

### 5. 评估

```bash
# 使用默认 best checkpoint
python eval.py

# 指定 checkpoint
python eval.py --checkpoint outputs/exp1_minimal_loop_final.pt
```

输出：BLEU-1/2/3/4、ROUGE-1/2/L、Exact Match、Word-level F1

## 数据流程

```
Flickr30k 原始 caption（正确文本）
        │
        ▼  规则替换（名词/形容词/数词）
错误文本（wrong_caption）
        │
        ▼  组成三元组
(image, wrong_caption, correct_caption)
        │
        ▼  Prompt 模板
"Please correct the following description based on the image: {wrong} Corrected:"
        │
        ▼  模型生成
generated_caption  ←→  correct_caption（计算指标）
```

## 关键配置（configs/config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_train_samples` | 200 | 训练样本数（最小闭环用少量） |
| `max_val_samples` | 50 | 验证样本数 |
| `batch_size` | 4 | 批大小 |
| `num_epochs` | 10 | 训练轮数 |
| `learning_rate` | 5e-5 | 学习率 |
| `overfit_batches` | 0 | >0 启用 overfit 调试模式 |

## 运行顺序 Checklist

- [ ] `pip install -r requirements.txt`
- [ ] `python scripts/prepare_data.py` — 生成纠错数据
- [ ] `python scripts/sanity_check.py` — 验证代码正确性
- [ ] `python train.py` — 正式训练
- [ ] `python eval.py` — 评估并输出报告
