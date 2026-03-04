"""
第一批实验配置 - 最小闭环
目标：跑通链路，不追求指标，只要结果合理即可
数据集：Flickr30k-Entities
"""
import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """数据相关配置"""
    # Flickr30k-Entities 数据路径
    flickr30k_images_dir: str = "./data/flickr30k-images"
    flickr30k_annotations_dir: str = "./data/flickr30k-entities"

    # 生成的纠错数据集缓存路径
    cache_dir: str = "./data/cache"
    train_json: str = "./data/cache/train_correction.json"
    val_json: str = "./data/cache/val_correction.json"

    # 最小闭环：只取少量数据跑通
    max_train_samples: int = 200
    max_val_samples: int = 50

    # 图像预处理
    image_size: int = 224


@dataclass
class ModelConfig:
    """模型相关配置"""
    # Vision Encoder
    vision_model_name: str = "openai/clip-vit-base-patch32"

    # Text Decoder
    text_model_name: str = "gpt2"

    # 生成参数
    max_source_length: int = 128  # 输入 prompt 最大长度
    max_target_length: int = 64   # 生成目标最大长度
    num_beams: int = 2            # beam search 宽度（最小闭环用2即可）


@dataclass
class TrainConfig:
    """训练相关配置"""
    # 基础训练参数
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 50
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # 设备
    device: str = "cuda"  # 如无 GPU 会自动回退到 cpu

    # 日志与保存
    log_dir: str = "./logs"
    output_dir: str = "./outputs"
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 10

    # 调试：overfit 模式（只用少量数据反复训练，验证代码正确性）
    overfit_batches: int = 0  # >0 时启用 overfit 模式，值为 batch 数


@dataclass
class ExpConfig:
    """总实验配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp_name: str = "exp1_minimal_loop"
    seed: int = 42
