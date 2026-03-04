"""
纠错数据集 Dataset - 加载 (图片, 错误文本, 正确文本) 三元组

核心职责：
  1. 从 JSON 读取纠错样本
  2. 加载图片并用 CLIP processor 预处理
  3. 对文本进行 tokenize
  4. 返回模型所需的 input_ids / attention_mask / pixel_values / labels
"""
import os
import json
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, GPT2Tokenizer


class CorrectionDataset(Dataset):
    """
    多模态纠错数据集

    每个样本包含：
      - pixel_values: CLIP 预处理后的图像张量
      - input_ids:    编码后的 prompt（含错误文本）
      - attention_mask
      - labels:       编码后的正确文本（训练目标）
    """

    # 纠错任务的 prompt 模板
    PROMPT_TEMPLATE = "Please correct the following description based on the image: {wrong_caption} Corrected:"

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "gpt2",
        max_source_length: int = 128,
        max_target_length: int = 64,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        # 加载数据
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.images_dir = images_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # 初始化 processors / tokenizers
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)

        # GPT2 没有 pad_token，用 eos_token 代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- 1. 加载图片 ----
        img_path = os.path.join(self.images_dir, sample["image_id"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # fallback: 如果图片不存在，创建一张纯色占位图
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        # CLIP 图像预处理
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)  # [3, 224, 224]

        # ---- 2. 构造 prompt (输入) ----
        prompt = self.PROMPT_TEMPLATE.format(wrong_caption=sample["wrong_caption"])

        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = source_encoding.input_ids.squeeze(0)           # [max_source_length]
        attention_mask = source_encoding.attention_mask.squeeze(0)  # [max_source_length]

        # ---- 3. 编码 target (正确文本) ----
        target_encoding = self.tokenizer(
            sample["correct_caption"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding.input_ids.squeeze(0)  # [max_target_length]
        # 将 pad 位置设为 -100，不参与 loss 计算
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "wrong_caption": sample["wrong_caption"],
            "correct_caption": sample["correct_caption"],
            "image_id": sample["image_id"],
        }


def build_dataloaders(cfg) -> tuple:
    """
    根据配置构建训练和验证的 DataLoader。

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CorrectionDataset(
        json_path=cfg.data.train_json,
        images_dir=cfg.data.flickr30k_images_dir,
        clip_model_name=cfg.model.vision_model_name,
        text_model_name=cfg.model.text_model_name,
        max_source_length=cfg.model.max_source_length,
        max_target_length=cfg.model.max_target_length,
    )

    val_dataset = CorrectionDataset(
        json_path=cfg.data.val_json,
        images_dir=cfg.data.flickr30k_images_dir,
        clip_model_name=cfg.model.vision_model_name,
        text_model_name=cfg.model.text_model_name,
        max_source_length=cfg.model.max_source_length,
        max_target_length=cfg.model.max_target_length,
    )

    # DataLoader 的 collate_fn：把字符串字段单独处理
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # 保留字符串信息（用于日志和 case study）
            "wrong_caption": [b["wrong_caption"] for b in batch],
            "correct_caption": [b["correct_caption"] for b in batch],
            "image_id": [b["image_id"] for b in batch],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows 下先用 0，避免多进程问题
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )

    print(f"[INFO] 训练集: {len(train_dataset)} 条, {len(train_loader)} batches")
    print(f"[INFO] 验证集: {len(val_dataset)} 条, {len(val_loader)} batches")

    return train_loader, val_loader
