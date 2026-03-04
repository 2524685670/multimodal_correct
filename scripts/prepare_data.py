"""
数据准备脚本 - 从 Flickr30k-Entities 构造纠错训练数据

核心思路：
  1. 读取 Flickr30k 原始 caption（正确文本）
  2. 通过规则替换，将 caption 中的名词/形容词/数词随机替换，生成"错误文本"
  3. 组成 (image_path, wrong_caption, correct_caption) 三元组
  4. 输出为 JSON 文件供 Dataset 读取

用法：
  python scripts/prepare_data.py
"""
import os
import sys
import json
import random
import re
from pathlib import Path

# 将项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import ExpConfig

# ============================================================
# 1. 规则替换词表（简单粗暴，够用就行）
# ============================================================

# 名词替换对：原词 -> 可替换的错误词列表
NOUN_SWAPS = {
    "dog": ["cat", "horse", "bird"],
    "cat": ["dog", "rabbit", "fish"],
    "man": ["woman", "boy", "child"],
    "woman": ["man", "girl", "child"],
    "boy": ["girl", "man", "child"],
    "girl": ["boy", "woman", "child"],
    "child": ["man", "woman", "dog"],
    "car": ["bus", "truck", "bike"],
    "bus": ["car", "train", "truck"],
    "bike": ["car", "scooter", "skateboard"],
    "tree": ["building", "pole", "fence"],
    "ball": ["frisbee", "hat", "bottle"],
    "hat": ["helmet", "cap", "scarf"],
    "shirt": ["jacket", "dress", "sweater"],
    "grass": ["sand", "snow", "concrete"],
    "water": ["sand", "mud", "ice"],
    "street": ["field", "beach", "park"],
    "beach": ["street", "park", "forest"],
    "park": ["beach", "street", "yard"],
    "house": ["building", "tent", "tower"],
    "table": ["chair", "bench", "desk"],
    "chair": ["table", "bench", "stool"],
    "food": ["drink", "toy", "book"],
    "guitar": ["piano", "drum", "violin"],
    "phone": ["camera", "book", "tablet"],
    "horse": ["dog", "cow", "deer"],
    "bird": ["fish", "dog", "butterfly"],
}

# 形容词替换对
ADJ_SWAPS = {
    "red": ["blue", "green", "yellow"],
    "blue": ["red", "green", "purple"],
    "green": ["red", "blue", "orange"],
    "yellow": ["red", "blue", "purple"],
    "white": ["black", "gray", "brown"],
    "black": ["white", "gray", "brown"],
    "big": ["small", "tiny", "little"],
    "small": ["big", "large", "huge"],
    "old": ["young", "new"],
    "young": ["old", "elderly"],
    "tall": ["short", "small"],
    "long": ["short", "brief"],
    "two": ["three", "four", "five"],
    "three": ["two", "four", "six"],
}

ALL_SWAPS = {**NOUN_SWAPS, **ADJ_SWAPS}


def perturb_caption(caption: str, min_swaps: int = 1, max_swaps: int = 2) -> tuple:
    """
    对一条 caption 进行规则替换，生成错误版本。

    Returns:
        (wrong_caption, swap_info_list) 或 (None, None) 如果无法替换
    """
    words = caption.lower().split()
    # 找出所有可替换的位置
    swappable = []
    for i, w in enumerate(words):
        # 去掉标点后匹配
        clean_w = re.sub(r'[^a-z]', '', w)
        if clean_w in ALL_SWAPS:
            swappable.append((i, clean_w, w))

    if not swappable:
        return None, None

    # 随机选择要替换的数量和位置
    num_swaps = min(random.randint(min_swaps, max_swaps), len(swappable))
    chosen = random.sample(swappable, num_swaps)

    swap_info = []
    result_words = list(words)
    for idx, clean_w, original_w in chosen:
        replacement = random.choice(ALL_SWAPS[clean_w])
        # 保留原词的标点（如逗号、句号）
        suffix = original_w[len(clean_w):]  # 提取尾部标点
        result_words[idx] = replacement + suffix
        swap_info.append({"position": idx, "original": clean_w, "replaced": replacement})

    wrong_caption = " ".join(result_words)
    return wrong_caption, swap_info


def load_flickr30k_captions(annotations_dir: str) -> dict:
    """
    加载 Flickr30k 的 caption 文件。
    期望格式为每行: image_id|caption_number\tcaption_text
    或者 Flickr30k 的 results_20130124.token 格式。

    如果找不到真实数据，则生成模拟数据（用于跑通链路）。
    """
    captions_by_image = {}

    # 尝试多种可能的文件名
    possible_files = [
        os.path.join(annotations_dir, "results_20130124.token"),
        os.path.join(annotations_dir, "results.token"),
        os.path.join(annotations_dir, "captions.txt"),
    ]

    caption_file = None
    for f in possible_files:
        if os.path.exists(f):
            caption_file = f
            break

    if caption_file:
        print(f"[INFO] 从 {caption_file} 加载真实 caption 数据...")
        with open(caption_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # 格式: image_filename#caption_idx \t caption_text
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                img_id_part = parts[0]
                caption_text = parts[1] if len(parts) > 1 else parts[0]

                # 提取 image_id: "1000092795.jpg#0" -> "1000092795.jpg"
                img_id = img_id_part.split("#")[0].strip()
                if img_id not in captions_by_image:
                    captions_by_image[img_id] = []
                captions_by_image[img_id].append(caption_text.strip())
    else:
        print("[WARNING] 未找到真实 Flickr30k caption 文件，生成模拟数据以跑通链路...")
        captions_by_image = generate_dummy_captions()

    return captions_by_image


def generate_dummy_captions() -> dict:
    """
    生成模拟 caption 数据（当真实数据不可用时，用于跑通链路）。
    每张 '假图' 配5条描述。
    """
    templates = [
        "a man in a red shirt is playing guitar on the street",
        "two children are running on the green grass near a tree",
        "a woman in a white dress is sitting on a chair in the park",
        "a boy is riding a bike on the street near a big house",
        "a girl with a hat is walking a dog on the beach",
        "a young man is throwing a ball to his dog in the park",
        "a woman is holding a phone and sitting at a table",
        "three boys are playing with a ball in the grass",
        "an old man in a black shirt is sitting on a chair",
        "a small child is playing with a cat near a tree",
        "a tall woman in a blue shirt is standing near a car",
        "two girls are eating food at a table in the park",
        "a man and a woman are walking on the beach near the water",
        "a boy in a yellow shirt is riding a horse on the street",
        "a young girl is playing guitar near a big tree",
        "a man in a green jacket is walking a dog on the street",
        "a woman with long hair is sitting on a bench near the water",
        "three children are playing in the grass near a house",
        "a big dog is running on the beach near the water",
        "a small bird is sitting on a tree in the park",
        "a man is standing near a white car on the street",
        "an old woman in a red hat is sitting at a table",
        "two boys are riding bikes on the street near a tree",
        "a girl is playing with a ball in the grass",
        "a young woman in a black dress is holding a guitar",
        "a tall man is throwing a ball to a boy in the park",
        "a child is sitting on the grass near a big dog",
        "a woman is walking on the street near a bus",
        "two men are sitting at a table and eating food",
        "a boy and a girl are playing near the water on the beach",
        "a man in a white shirt is sitting on a chair in the park",
        "a small cat is sitting near a tree in the grass",
        "a woman is riding a horse near a big house",
        "three girls are walking a dog on the street",
        "a boy in a blue shirt is playing with a ball",
        "a young man is sitting at a table near a tree",
        "a woman with a hat is standing near a car",
        "a big bird is flying near the water on the beach",
        "two children are sitting on the grass in the park",
        "a girl in a green dress is playing guitar near the street",
        "a man is walking on the beach with a small dog",
        "an old man is sitting on a chair near a table",
        "a young boy is riding a bike in the park",
        "a tall woman is walking near a tree on the street",
        "a child and a dog are running on the grass near the water",
        "a woman in a yellow shirt is sitting on a bench",
        "three men are standing near a bus on the street",
        "a girl is playing with a cat near a big tree",
        "a small boy is throwing a ball in the park",
        "a man and a dog are walking near the water on the beach",
    ]

    captions_by_image = {}
    for i in range(len(templates)):
        img_id = f"dummy_{i:05d}.jpg"
        # 每张图配 5 条 caption（部分重复变体）
        caps = []
        base = templates[i]
        caps.append(base)
        # 简单变体
        caps.append(base.replace("a ", "the ", 1))
        caps.append(base + " today")
        caps.append("in the photo , " + base)
        caps.append(base.replace("is", "was", 1))
        captions_by_image[img_id] = caps

    return captions_by_image


def create_dummy_images(image_dir: str, image_ids: list):
    """
    为模拟数据创建 dummy 图片（单色 224x224 图片）。
    """
    from PIL import Image
    os.makedirs(image_dir, exist_ok=True)
    for img_id in image_ids:
        img_path = os.path.join(image_dir, img_id)
        if not os.path.exists(img_path):
            # 随机颜色的纯色图片
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", (224, 224), color)
            img.save(img_path)


def build_correction_dataset(captions_by_image: dict, max_samples: int) -> list:
    """
    从 caption 数据构造纠错样本列表。

    每个样本：
    {
        "image_id": "xxx.jpg",
        "wrong_caption": "a cat is running ...",
        "correct_caption": "a dog is running ...",
        "swap_info": [{"position": 1, "original": "dog", "replaced": "cat"}]
    }
    """
    samples = []
    image_ids = list(captions_by_image.keys())
    random.shuffle(image_ids)

    for img_id in image_ids:
        if len(samples) >= max_samples:
            break

        captions = captions_by_image[img_id]
        for cap in captions:
            if len(samples) >= max_samples:
                break

            wrong_cap, swap_info = perturb_caption(cap)
            if wrong_cap is None:
                continue

            samples.append({
                "image_id": img_id,
                "wrong_caption": wrong_cap,
                "correct_caption": cap.lower(),
                "swap_info": swap_info,
            })

    return samples


def main():
    cfg = ExpConfig()
    random.seed(cfg.seed)

    print("=" * 60)
    print("  第一批实验 - 数据准备脚本")
    print("  目标：构造 Flickr30k-Entities 纠错数据集")
    print("=" * 60)

    # 1. 加载 caption
    captions_by_image = load_flickr30k_captions(cfg.data.flickr30k_annotations_dir)
    print(f"[INFO] 共加载 {len(captions_by_image)} 张图片的 caption 数据")

    # 2. 如果是 dummy 数据，创建 dummy 图片
    is_dummy = any(k.startswith("dummy_") for k in captions_by_image)
    if is_dummy:
        print("[INFO] 创建 dummy 图片...")
        create_dummy_images(cfg.data.flickr30k_images_dir, list(captions_by_image.keys()))

    # 3. 构造纠错数据集
    all_image_ids = list(captions_by_image.keys())
    random.shuffle(all_image_ids)

    # 按 8:2 划分训练/验证
    split_idx = int(len(all_image_ids) * 0.8)
    train_ids = set(all_image_ids[:split_idx])
    val_ids = set(all_image_ids[split_idx:])

    train_captions = {k: v for k, v in captions_by_image.items() if k in train_ids}
    val_captions = {k: v for k, v in captions_by_image.items() if k in val_ids}

    print(f"[INFO] 构造训练集 (最多 {cfg.data.max_train_samples} 条)...")
    train_samples = build_correction_dataset(train_captions, cfg.data.max_train_samples)

    print(f"[INFO] 构造验证集 (最多 {cfg.data.max_val_samples} 条)...")
    val_samples = build_correction_dataset(val_captions, cfg.data.max_val_samples)

    # 4. 保存为 JSON
    os.makedirs(cfg.data.cache_dir, exist_ok=True)

    with open(cfg.data.train_json, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 训练集保存至 {cfg.data.train_json}  ({len(train_samples)} 条)")

    with open(cfg.data.val_json, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 验证集保存至 {cfg.data.val_json}  ({len(val_samples)} 条)")

    # 5. 打印几条样本预览
    print("\n" + "=" * 60)
    print("  样本预览 (前3条训练样本)")
    print("=" * 60)
    for i, s in enumerate(train_samples[:3]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"  Image ID:     {s['image_id']}")
        print(f"  错误文本:     {s['wrong_caption']}")
        print(f"  正确文本:     {s['correct_caption']}")
        print(f"  替换信息:     {s['swap_info']}")

    print("\n[DONE] 数据准备完成！")


if __name__ == "__main__":
    main()
