"""
第一批实验 - 训练脚本

完整训练流程：
  1. 准备数据（如果尚未准备）
  2. 构建 DataLoader
  3. 构建模型
  4. 训练 + 验证循环
  5. 日志记录 + 模型保存
  6. Case Study 输出

用法：
  python train.py
"""
import os
import sys
import time
import json
import random
import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 将项目根目录加入 path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config import ExpConfig
from data.dataset import build_dataloaders
from models.model import build_model


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        # overfit 模式：只用前 N 个 batch
        if cfg.train.overfit_batches > 0 and step >= cfg.train.overfit_batches:
            break

        # 数据移到设备
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # 反向
        loss.backward()

        # 梯度累积
        if (step + 1) % cfg.train.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        # 日志
        if (step + 1) % cfg.train.log_every_n_steps == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch+1}] Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device, cfg):
    """验证一个 epoch，返回平均 loss 和 case study 结果"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    case_studies = []

    for step, batch in enumerate(val_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 计算 loss
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs["loss"].item()
        num_batches += 1

        # 对前几个 batch 做 case study（生成纠正文本）
        if step < 2:
            generated = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.model.max_target_length,
            )

            for i in range(len(generated)):
                case_studies.append({
                    "image_id": batch["image_id"][i],
                    "wrong_caption": batch["wrong_caption"][i],
                    "correct_caption": batch["correct_caption"][i],
                    "generated": generated[i],
                })

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, case_studies


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存模型 checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, save_path)
    print(f"  [SAVE] Checkpoint saved to {save_path}")


def main():
    cfg = ExpConfig()
    set_seed(cfg.seed)

    print("=" * 70)
    print(f"  第一批实验: {cfg.exp_name}")
    print(f"  时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 0. 检查数据是否已准备 ----
    if not os.path.exists(cfg.data.train_json) or not os.path.exists(cfg.data.val_json):
        print("\n[INFO] 数据文件不存在，先运行数据准备脚本...")
        from scripts.prepare_data import main as prepare_main
        prepare_main()
        print()

    # ---- 1. 设备 ----
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] 使用 CPU 训练")

    # ---- 2. 构建 DataLoader ----
    print("\n[INFO] 构建 DataLoader...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ---- 3. 构建模型 ----
    print("\n[INFO] 构建模型...")
    model = build_model(cfg)
    model = model.to(device)

    # ---- 4. 优化器 & 调度器 ----
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    total_steps = len(train_loader) * cfg.train.num_epochs
    if cfg.train.overfit_batches > 0:
        total_steps = cfg.train.overfit_batches * cfg.train.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # ---- 5. 日志准备 ----
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    log_file = os.path.join(cfg.train.log_dir, f"{cfg.exp_name}_log.jsonl")
    log_fh = open(log_file, "w", encoding="utf-8")

    best_val_loss = float("inf")
    training_history = []

    # ---- 6. 训练循环 ----
    print(f"\n{'='*70}")
    print(f"  开始训练! Epochs: {cfg.train.num_epochs}, Batch Size: {cfg.train.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(cfg.train.num_epochs):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg)

        # 验证
        val_loss, case_studies = validate(model, val_loader, device, cfg)

        epoch_time = time.time() - epoch_start

        # 日志记录
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(epoch_time, 1),
        }
        training_history.append(log_entry)

        log_fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        log_fh.flush()

        print(f"\n  Epoch {epoch+1}/{cfg.train.num_epochs} 完成")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    耗时: {epoch_time:.1f}s")

        # 打印 case study
        if case_studies:
            print(f"\n    --- Case Study (Epoch {epoch+1}) ---")
            for cs in case_studies[:3]:
                print(f"    Image: {cs['image_id']}")
                print(f"      错误文本: {cs['wrong_caption']}")
                print(f"      正确文本: {cs['correct_caption']}")
                print(f"      生成文本: {cs['generated']}")
                print()

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_best.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

        # 定期保存
        if (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            save_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

    # ---- 7. 训练结束 ----
    log_fh.close()

    # 保存最终模型
    final_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_final.pt")
    save_checkpoint(model, optimizer, cfg.train.num_epochs - 1, val_loss, final_path)

    # 保存完整训练历史
    history_path = os.path.join(cfg.train.log_dir, f"{cfg.exp_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)

    # 保存最后一轮的 case study
    if case_studies:
        case_path = os.path.join(cfg.train.output_dir, f"{cfg.exp_name}_case_study.json")
        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case_studies, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  训练完成!")
    print(f"  最优验证 Loss: {best_val_loss:.4f}")
    print(f"  模型保存至: {cfg.train.output_dir}")
    print(f"  日志保存至: {cfg.train.log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
