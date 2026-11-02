"""使用指定仓库的 CNNSpot 网络和权重，进行正常预测。"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent / "AIGCDetectBenchmark-main" / "AIGCDetectBenchmark-main"
WEIGHT_PATH = PROJECT_DIR / "weights" / "classifier" / "CNNSpot.pth"

# 使用原仓库的网络定义。
sys.path.insert(0, str(PROJECT_DIR))
from networks.resnet import resnet50


def main():
    parser = argparse.ArgumentParser(description="CNNSpot 正常预测：先读取少量真假图片。")
    parser.add_argument("--data-root", type=Path, default=DEMO_DIR / "test",
                        help="直接包含 0_real、1_fake 的目录")
    parser.add_argument("--max-per-class", type=int, default=10,
                        help="每类最多读取的图片数，按文件路径排序选取")
    parser.add_argument("--output", type=Path, default=DEMO_DIR / "baseline_results.csv",
                        help="预测结果 CSV 的保存位置")
    args = parser.parse_args()
    if args.max_per_class < 1:
        parser.error("--max-per-class 必须大于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 创建网络，再装入已经训练好的参数。
    model = resnet50(num_classes=1)
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    # 2. 沿用原项目 CNNSpot 的默认测试预处理。
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    if dataset.class_to_idx != {"0_real": 0, "1_fake": 1}:
        raise ValueError("测试目录应包含 0_real 和 1_fake 两类。")

    # 按固定文件顺序各取少量图片；CSV 记录相对路径，方便后面用同一批图片攻击。
    selected_indices = []
    selected_counts = {0: 0, 1: 0}
    available_counts = {0: 0, 1: 0}
    for index, (_, label) in enumerate(dataset.samples):
        available_counts[label] += 1
        if selected_counts[label] < args.max_per_class:
            selected_indices.append(index)
            selected_counts[label] += 1

    # 3. 图片送入模型，概率大于 0.5 时预测为生成图。
    names = {0: "真实图", 1: "生成图"}
    rows = []
    print(f"计算设备：{device}")
    print(f"图片目录：{args.data_root}")
    print(f"目录内图片：真实图 {available_counts[0]} 张，生成图 {available_counts[1]} 张")
    print(f"本次选取：真实图 {selected_counts[0]} 张，生成图 {selected_counts[1]} 张")
    with torch.no_grad():
        for index in selected_indices:
            image, label = dataset[index]
            image_batch = image.unsqueeze(0).to(device)
            probability = model(image_batch).sigmoid().item()
            prediction = int(probability > 0.5)
            filename = Path(dataset.samples[index][0]).name
            correct = prediction == label
            rows.append({
                "filename": filename,
                "relative_path": str(Path(dataset.samples[index][0]).relative_to(args.data_root)),
                "true_label": label,
                "fake_probability": probability,
                "predicted_label": prediction,
                "correct": correct,
            })
            print(f"{filename} | 实际：{names[label]} | 生成图概率：{probability:.6f}"
                  f" | 预测：{names[prediction]} | {'正确' if correct else '错误'}")

    # 4. 保存本次正常预测结果，供后面的攻击实验对照。
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    print(f"本次小样本准确率：{accuracy:.1%}（共 {len(rows)} 张，不代表整个测试集结果）")
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
