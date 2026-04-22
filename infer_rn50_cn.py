import argparse
import time
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 OpenCLIP RN50 做推理")
    parser.add_argument("--image", type=Path, help="输入图片路径")
    parser.add_argument(
        "--text",
        nargs="+",
        help="候选文本，可同时传多条，例如 --text 一张图 一只狗 一只猫",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备，默认自动选择 cuda 或 cpu",
    )
    parser.add_argument(
        "--benchmark-cifar10",
        action="store_true",
        help="下载并使用 CIFAR-10 官方 test split 做批量性能评估",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="批量评估使用的图片数量，默认 50",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="正式统计前的热身张数，默认 3",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="数据集下载目录，默认 data",
    )
    args = parser.parse_args()

    if args.benchmark_cifar10 and args.image is not None:
        parser.error("--benchmark-cifar10 模式下不要同时传 --image")
    if not args.benchmark_cifar10 and args.image is None:
        parser.error("普通推理模式需要提供 --image；批量评估模式请加 --benchmark-cifar10")
    if not args.benchmark_cifar10 and not args.text:
        parser.error("普通推理模式需要提供 --text")
    if args.num_images <= 0:
        parser.error("--num-images 必须是正整数")
    if args.warmup < 0:
        parser.error("--warmup 不能是负数")
    if args.benchmark_cifar10 and not args.text:
        args.text = CIFAR10_CLASS_NAMES.copy()

    return args


def load_model_and_tokenizer(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",
        pretrained="openai",
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("RN50")
    return model, preprocess, tokenizer


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_single_inference(args: argparse.Namespace, device: torch.device) -> None:
    model, preprocess, tokenizer = load_model_and_tokenizer(device)

    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer(args.text).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    print(f"device: {device}")
    print(f"image: {args.image}")
    print("results:")
    for candidate, prob in sorted(zip(args.text, text_probs.tolist()), key=lambda x: x[1], reverse=True):
        print(f"{prob:.6f}\t{candidate}")


def benchmark_cifar10(args: argparse.Namespace, device: torch.device) -> None:
    model, preprocess, tokenizer = load_model_and_tokenizer(device)
    dataset = CIFAR10(root=args.data_root, train=False, download=True)

    total_needed = args.warmup + args.num_images
    if total_needed > len(dataset):
        raise ValueError(f"需要 {total_needed} 张图，但 CIFAR-10 test split 只有 {len(dataset)} 张")

    text_start = time.perf_counter()
    text = tokenizer(args.text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        maybe_sync(device)
    text_elapsed = time.perf_counter() - text_start

    preprocess_times = []
    image_encode_times = []
    score_times = []
    total_times = []

    for idx in range(total_needed):
        pil_image, _ = dataset[idx]

        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        image = preprocess(pil_image).unsqueeze(0).to(device)
        maybe_sync(device)
        preprocess_elapsed = time.perf_counter() - preprocess_start

        with torch.no_grad():
            image_encode_start = time.perf_counter()
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            maybe_sync(device)
            image_encode_elapsed = time.perf_counter() - image_encode_start

            score_start = time.perf_counter()
            _ = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            maybe_sync(device)
            score_elapsed = time.perf_counter() - score_start

        total_elapsed = time.perf_counter() - total_start

        if idx >= args.warmup:
            preprocess_times.append(preprocess_elapsed)
            image_encode_times.append(image_encode_elapsed)
            score_times.append(score_elapsed)
            total_times.append(total_elapsed)

    avg_preprocess = sum(preprocess_times) / len(preprocess_times)
    avg_image_encode = sum(image_encode_times) / len(image_encode_times)
    avg_score = sum(score_times) / len(score_times)
    avg_total = sum(total_times) / len(total_times)
    fps = 1.0 / avg_total if avg_total > 0 else float("inf")

    print(f"device: {device}")
    print("benchmark: CIFAR-10 test split")
    print(f"data_root: {args.data_root}")
    print(f"num_images: {args.num_images}")
    print(f"warmup: {args.warmup}")
    print(f"text_count: {len(args.text)}")
    print("metrics:")
    print(f"text_encode_seconds: {text_elapsed:.6f}")
    print(f"avg_preprocess_seconds: {avg_preprocess:.6f}")
    print(f"avg_image_encode_seconds: {avg_image_encode:.6f}")
    print(f"avg_score_seconds: {avg_score:.6f}")
    print(f"avg_total_seconds_per_image: {avg_total:.6f}")
    print(f"fps: {fps:.3f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.benchmark_cifar10:
        benchmark_cifar10(args, device)
    else:
        run_single_inference(args, device)


if __name__ == "__main__":
    main()
