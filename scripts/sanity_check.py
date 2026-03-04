"""
Sanity Check 脚本 - Overfit 单 batch 验证代码正确性

核心逻辑：
  1. 取训练集的 1 个 batch（4条数据）
  2. 反复训练 50~100 步
  3. 观察 loss 是否能逼近 0
  4. 在这 4 条数据上做 generate，看输出是否逐渐趋近 target

如果 loss 能降到 ~0 且生成能复现 target，则说明：
  - 数据加载无误
  - 模型前向/反向梯度链路通畅
  - Loss 计算逻辑正确
  - 生成逻辑正确

用法：
  python scripts/sanity_check.py
"""
import os
import sys
import time

import torch
from torch.optim import AdamW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config import ExpConfig
from data.dataset import build_dataloaders
from models.model import build_model


def main():
    cfg = ExpConfig()

    print("=" * 60)
    print("  Sanity Check: Overfit 单 Batch")
    print("  目标: Loss -> 0, 生成 == Target")
    print("=" * 60)

    # ---- 0. 数据准备 ----
    if not os.path.exists(cfg.data.train_json):
        print("\n[INFO] 数据文件不存在，先运行数据准备...")
        from scripts.prepare_data import main as prepare_main
        prepare_main()

    # ---- 1. 设备 ----
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] 使用 CPU")

    # ---- 2. 取一个 batch ----
    print("\n[INFO] 构建 DataLoader...")
    train_loader, _ = build_dataloaders(cfg)

    # 取第一个 batch 并固定
    fixed_batch = next(iter(train_loader))
    print(f"[INFO] 固定 batch: {len(fixed_batch['image_id'])} 条样本")
    for i, img_id in enumerate(fixed_batch["image_id"]):
        print(f"  [{i}] {img_id}")
        print(f"      Wrong:   {fixed_batch['wrong_caption'][i][:60]}...")
        print(f"      Correct: {fixed_batch['correct_caption'][i][:60]}...")

    # 移到设备
    pixel_values = fixed_batch["pixel_values"].to(device)
    input_ids = fixed_batch["input_ids"].to(device)
    attention_mask = fixed_batch["attention_mask"].to(device)
    labels = fixed_batch["labels"].to(device)

    # ---- 3. 构建模型 ----
    print("\n[INFO] 构建模型...")
    model = build_model(cfg)
    model = model.to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,  # sanity check 用稍大的 lr
    )

    # ---- 4. Overfit 循环 ----
    num_steps = 100
    print(f"\n[INFO] 开始 Overfit {num_steps} 步...\n")

    model.train()
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step+1:3d}/{num_steps} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")

    final_loss = loss.item()
    print(f"\n  最终 Loss: {final_loss:.6f}")

    if final_loss < 0.1:
        print("  ✓ Loss 已降至接近 0，梯度链路正确！")
    elif final_loss < 1.0:
        print("  △ Loss 已显著下降但未接近 0，可能需要更多步数")
    else:
        print("  ✗ Loss 仍较高，需要检查代码逻辑")

    # ---- 5. 在固定 batch 上 Generate ----
    print(f"\n[INFO] 在固定 batch 上生成纠正文本...\n")
    model.eval()

    generated = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.model.max_target_length,
    )

    print("  " + "-" * 56)
    all_match = True
    for i in range(len(generated)):
        ref = fixed_batch["correct_caption"][i]
        gen = generated[i]
        match = gen.lower().strip() == ref.lower().strip()
        if not match:
            all_match = False
        status = "✓" if match else "✗"

        print(f"  [{i}] {status}")
        print(f"      Wrong:     {fixed_batch['wrong_caption'][i]}")
        print(f"      Target:    {ref}")
        print(f"      Generated: {gen}")
        print()

    print("  " + "-" * 56)
    if all_match:
        print("  ✓ 所有样本生成完全匹配 Target！Sanity Check 通过！")
    else:
        print("  △ 部分样本未完全匹配（最小闭环阶段可接受，")
        print("    只要 loss 在下降且生成内容合理即可）")

    print(f"\n{'='*60}")
    print("  Sanity Check 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
