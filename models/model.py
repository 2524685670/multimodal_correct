"""
多模态纠错模型 - CLIP Vision Encoder + GPT2 Text Decoder

架构：
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Image      │────▶│ CLIP ViT     │────▶│ Projection  │──┐
  │  (224x224)  │     │ (frozen)     │     │ Linear      │  │
  └─────────────┘     └──────────────┘     └─────────────┘  │
                                                             ▼
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
  │  Prompt     │────▶│ GPT2 Embed   │────▶│ Concat      │◀─┘
  │  (text)     │     │              │     │ [img;text]   │
  └─────────────┘     └──────────────┘     └──────┬──────┘
                                                   ▼
                                            ┌─────────────┐
                                            │  GPT2       │
                                            │  Decoder    │──▶ 生成纠正文本
                                            └─────────────┘

设计原则（最小闭环）：
  - CLIP Vision Encoder 冻结参数，只训练 projection + GPT2
  - Projection: 一层线性映射，将 CLIP 视觉特征对齐到 GPT2 hidden dim
  - 训练目标: 标准自回归语言模型 CrossEntropy Loss
"""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer


class MultimodalCorrectionModel(nn.Module):
    """
    多模态纠错模型

    将 CLIP 图像特征通过线性投影注入 GPT2，
    GPT2 以 [image_tokens, prompt_tokens] 拼接后自回归生成纠正文本。
    """

    def __init__(self, vision_model_name: str, text_model_name: str):
        super().__init__()

        # ---- Vision Encoder (CLIP ViT, frozen) ----
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        # 冻结视觉编码器
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        vision_hidden_size = self.vision_encoder.config.hidden_size  # 768 for base

        # ---- Text Decoder (GPT2) ----
        self.text_decoder = GPT2LMHeadModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_decoder.config.n_embd  # 768 for gpt2

        # ---- Projection: vision_hidden -> text_hidden ----
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

        # 视觉 token 数量（CLIP ViT-B/32 输出 50 个 patch token + 1 CLS）
        # 为了控制序列长度，我们只取 CLS token + 少量 patch tokens
        self.num_visual_tokens = 8  # 用前 8 个 token，够用且省显存

        # Tokenizer（用于 generate 时 decode）
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像，返回投影后的视觉 token embeddings。

        Args:
            pixel_values: [B, 3, 224, 224]

        Returns:
            visual_embeds: [B, num_visual_tokens, text_hidden_size]
        """
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            # last_hidden_state: [B, num_patches+1, vision_hidden_size]
            hidden_states = vision_outputs.last_hidden_state

        # 取前 num_visual_tokens 个 token
        visual_tokens = hidden_states[:, :self.num_visual_tokens, :]  # [B, 8, 768]
        # 投影到 GPT2 的隐空间
        visual_embeds = self.vision_projection(visual_tokens)  # [B, 8, 768]

        return visual_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        前向传播：拼接视觉 token 和文本 token，计算自回归 loss。

        Args:
            pixel_values:  [B, 3, 224, 224]
            input_ids:     [B, src_len]   prompt tokens
            attention_mask:[B, src_len]
            labels:        [B, tgt_len]   target tokens

        Returns:
            dict with 'loss' and 'logits'
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. 编码图像 -> visual embeddings
        visual_embeds = self.encode_image(pixel_values)  # [B, num_vis, hidden]

        # 2. 获取文本 embeddings
        text_embeds = self.text_decoder.transformer.wte(input_ids)  # [B, src_len, hidden]

        # 3. 拼接: [visual_tokens | prompt_tokens]
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        # [B, num_vis + src_len, hidden]

        # 4. 构造对应的 attention mask
        vis_mask = torch.ones(batch_size, self.num_visual_tokens, dtype=torch.long, device=device)
        combined_mask = torch.cat([vis_mask, attention_mask], dim=1)
        # [B, num_vis + src_len]

        # 5. 构造 labels：视觉部分和 prompt 部分设为 -100（不计算 loss）
        #    只有 target 部分参与 loss
        prompt_ignore = torch.full(
            (batch_size, self.num_visual_tokens + input_ids.size(1)),
            -100, dtype=torch.long, device=device
        )
        combined_labels = torch.cat([prompt_ignore, labels], dim=1)
        # [B, num_vis + src_len + tgt_len]

        # 6. target 的 embeddings 也要拼上
        # 把 labels 中的 -100 替换为 pad_token_id 以便 embedding（-100 不能做 index）
        labels_for_embed = labels.clone()
        labels_for_embed[labels_for_embed == -100] = self.tokenizer.pad_token_id
        target_embeds = self.text_decoder.transformer.wte(labels_for_embed)  # [B, tgt_len, hidden]

        # 完整输入: [visual | prompt | target]
        full_embeds = torch.cat([combined_embeds, target_embeds], dim=1)

        # target 的 attention mask
        target_mask = (labels != -100).long()
        full_mask = torch.cat([combined_mask, target_mask], dim=1)

        # 7. 构造 position_ids
        seq_len = full_embeds.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 8. 前向 GPT2
        outputs = self.text_decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            labels=combined_labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        num_beams: int = 2,
    ) -> list:
        """
        推理生成纠正后的文本。

        Args:
            pixel_values:  [B, 3, 224, 224]
            input_ids:     [B, src_len]
            attention_mask:[B, src_len]
            max_new_tokens: 最大生成 token 数
            num_beams:     beam search 宽度

        Returns:
            生成的文本列表 [str, str, ...]
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. 图像 + prompt embeddings
        visual_embeds = self.encode_image(pixel_values)
        text_embeds = self.text_decoder.transformer.wte(input_ids)
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        vis_mask = torch.ones(batch_size, self.num_visual_tokens, dtype=torch.long, device=device)
        combined_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 2. 用 GPT2 的 generate 方法
        #    由于我们用 inputs_embeds，需要特殊处理
        #    这里用一个简单的逐 token 生成方式

        generated_tokens = []
        for i in range(batch_size):
            cur_embeds = combined_embeds[i:i+1]  # [1, seq_len, hidden]
            cur_mask = combined_mask[i:i+1]      # [1, seq_len]

            generated_ids = []
            for step in range(max_new_tokens):
                seq_len = cur_embeds.size(1)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

                outputs = self.text_decoder(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_mask,
                    position_ids=position_ids,
                )

                # 取最后一个 token 的 logits
                next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                next_token_id = next_logits.argmax(dim=-1)  # [1]

                # 检查是否生成了 EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id.item())

                # 把新 token embed 拼到序列末尾
                next_embed = self.text_decoder.transformer.wte(next_token_id.unsqueeze(0))  # [1,1,hidden]
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                cur_mask = torch.cat([cur_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_tokens.append(text.strip())

        return generated_tokens


def build_model(cfg) -> MultimodalCorrectionModel:
    """根据配置构建模型"""
    model = MultimodalCorrectionModel(
        vision_model_name=cfg.model.vision_model_name,
        text_model_name=cfg.model.text_model_name,
    )

    # 打印可训练参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[INFO] 模型参数统计:")
    print(f"  总参数量:   {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数:   {frozen_params:,}")
    print(f"  可训练比例: {trainable_params/total_params*100:.1f}%")

    return model
