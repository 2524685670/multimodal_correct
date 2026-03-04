"""
第一批实验 - 评估脚本

功能：
  1. 加载训练好的 checkpoint
  2. 在验证集上生成纠正文本
  3. 计算 BLEU-4 和 ROUGE-L 指标
  4. 输出详细 case study 报告

用法：
  python eval.py                           # 使用默认 best checkpoint
  python eval.py --checkpoint outputs/exp1_minimal_loop_final.pt
"""
import os
import sys
import json
import argparse

import torch
import nltk
from rouge_score import rouge_scorer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config import ExpConfig
from data.dataset import build_dataloaders
from models.model import build_model


# ============================================================
# 评估指标
# ============================================================

def compute_bleu(predictions: list, references: list) -> dict:
    """
    计算 BLEU-1/2/3/4 分数。

    Args:
        predictions: 生成的文本列表
        references:  参考文本列表（与 predictions 一一对应）

    Returns:
        dict: {"bleu1": float, "bleu2": float, "bleu3": float, "bleu4": float}
    """
    # 确保 nltk 的 punkt tokenizer 可用
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # 分词
    tokenized_preds = [pred.lower().split() for pred in predictions]
    tokenized_refs = [[ref.lower().split()] for ref in references]  # corpus_bleu 要求双层嵌套

    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "bleu1": round(bleu1 * 100, 2),
        "bleu2": round(bleu2 * 100, 2),
        "bleu3": round(bleu3 * 100, 2),
        "bleu4": round(bleu4 * 100, 2),
    }


def compute_rouge(predictions: list, references: list) -> dict:
    """
    计算 ROUGE-1/2/L 分数。

    Returns:
        dict: {"rouge1": float, "rouge2": float, "rougeL": float}
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {
        "rouge1": round(sum(scores["rouge1"]) / len(scores["rouge1"]) * 100, 2),
        "rouge2": round(sum(scores["rouge2"]) / len(scores["rouge2"]) * 100, 2),
        "rougeL": round(sum(scores["rougeL"]) / len(scores["rougeL"]) * 100, 2),
    }


def compute_correction_accuracy(predictions: list, references: list, wrong_captions: list) -> dict:
    """
    计算纠错准确率（简单的精确匹配 + 部分匹配）。

    Returns:
        dict: {"exact_match": float, "word_level_f1": float}
    """
    exact_match_count = 0
    total_f1 = 0.0

    for pred, ref, wrong in zip(predictions, references, wrong_captions):
        pred_lower = pred.lower().strip()
        ref_lower = ref.lower().strip()

        # 精确匹配
        if pred_lower == ref_lower:
            exact_match_count += 1

        # 词级别 F1
        pred_tokens = set(pred_lower.split())
        ref_tokens = set(ref_lower.split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            total_f1 += 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            total_f1 += 0.0
        else:
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            total_f1 += f1

    n = len(predictions)
    return {
        "exact_match": round(exact_match_count / n * 100, 2) if n > 0 else 0.0,
        "word_level_f1": round(total_f1 / n * 100, 2) if n > 0 else 0.0,
    }


# ============================================================
# 主评估流程
# ============================================================

@torch.no_grad()
def evaluate(model, val_loader, device, cfg):
    """
    在验证集上运行完整评估。

    Returns:
        metrics: dict, 所有指标
        all_results: list, 每个样本的详细结果
    """
    model.eval()
    all_predictions = []
    all_references = []
    all_wrong = []
    all_results = []

    print("\n[INFO] 开始生成纠正文本...")
    for step, batch in enumerate(val_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 生成
        generated = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.model.max_target_length,
        )

        for i in range(len(generated)):
            result = {
                "image_id": batch["image_id"][i],
                "wrong_caption": batch["wrong_caption"][i],
                "correct_caption": batch["correct_caption"][i],
                "generated": generated[i],
            }
            all_results.append(result)
            all_predictions.append(generated[i])
            all_references.append(batch["correct_caption"][i])
            all_wrong.append(batch["wrong_caption"][i])

        if (step + 1) % 5 == 0:
            print(f"  已处理 {step + 1}/{len(val_loader)} batches")

    print(f"  生成完成，共 {len(all_predictions)} 条")

    # 计算指标
    print("\n[INFO] 计算评估指标...")
    bleu_scores = compute_bleu(all_predictions, all_references)
    rouge_scores = compute_rouge(all_predictions, all_references)
    correction_scores = compute_correction_accuracy(all_predictions, all_references, all_wrong)

    metrics = {**bleu_scores, **rouge_scores, **correction_scores}
    return metrics, all_results


def main():
    parser = argparse.ArgumentParser(description="第一批实验 - 评估脚本")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型 checkpoint 路径 (默认使用 best checkpoint)"
    )
    args = parser.parse_args()

    cfg = ExpConfig()

    # 设备
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] 使用设备: {device}")

    # 构建 DataLoader（只需验证集）
    print("[INFO] 构建验证集 DataLoader...")
    _, val_loader = build_dataloaders(cfg)

    # 构建模型
    print("[INFO] 构建模型...")
    model = build_model(cfg)

    # 加载 checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_best.pt")

    if os.path.exists(ckpt_path):
        print(f"[INFO] 加载 checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Epoch: {checkpoint.get('epoch', '?')}, Loss: {checkpoint.get('loss', '?')}")
    else:
        print(f"[WARNING] 未找到 checkpoint: {ckpt_path}，使用未训练的模型进行评估")

    model = model.to(device)

    # 运行评估
    metrics, all_results = evaluate(model, val_loader, device, cfg)

    # 打印结果
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)
    print(f"\n  BLEU Scores:")
    print(f"    BLEU-1: {metrics['bleu1']:.2f}")
    print(f"    BLEU-2: {metrics['bleu2']:.2f}")
    print(f"    BLEU-3: {metrics['bleu3']:.2f}")
    print(f"    BLEU-4: {metrics['bleu4']:.2f}")
    print(f"\n  ROUGE Scores:")
    print(f"    ROUGE-1: {metrics['rouge1']:.2f}")
    print(f"    ROUGE-2: {metrics['rouge2']:.2f}")
    print(f"    ROUGE-L: {metrics['rougeL']:.2f}")
    print(f"\n  Correction Accuracy:")
    print(f"    Exact Match:   {metrics['exact_match']:.2f}%")
    print(f"    Word-level F1: {metrics['word_level_f1']:.2f}%")

    # 打印 Case Study
    print(f"\n{'='*60}")
    print(f"  Case Study (前 5 条)")
    print(f"{'='*60}")
    for i, r in enumerate(all_results[:5]):
        print(f"\n  --- 样本 {i+1} ---")
        print(f"  Image:    {r['image_id']}")
        print(f"  错误文本: {r['wrong_caption']}")
        print(f"  正确文本: {r['correct_caption']}")
        print(f"  生成文本: {r['generated']}")
        # 简单判断是否修正成功
        gen_lower = r["generated"].lower().strip()
        ref_lower = r["correct_caption"].lower().strip()
        status = "✓ 完全匹配" if gen_lower == ref_lower else "✗ 不完全匹配"
        print(f"  状态:     {status}")

    # 保存评估结果
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    eval_output_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_eval_results.json")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 详细评估结果已保存至: {eval_output_path}")
    print("[DONE] 评估完成!")


if __name__ == "__main__":
    main()
