"""固定名单上的 PGD 实验：仅攻击原本预测正确的图片，并分别统计真假两类。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

from predict_samples import DEMO_DIR, WEIGHT_PATH, resnet50

ATTACK_DIR = DEMO_DIR.parent / "adversarial-attacks-pytorch-master"
sys.path.insert(0, str(ATTACK_DIR))
import torchattacks

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TwoClassCNNSpot(nn.Module):
    """将 z 整理成 [-z/2,z/2]，保留预测含义并改善梯度数值。"""

    def __init__(self, cnnspot):
        super().__init__()
        self.cnnspot = cnnspot

    def forward(self, images):
        z = self.cnnspot(images)
        # 两个分数之差仍是 z，生成概率仍是 sigmoid(z)。
        # [0,z] 在高置信生成图上会因 p-1 舍入为零而丢失梯度。
        return torch.cat([-z / 2, z / 2], dim=1)


def main():
    parser = argparse.ArgumentParser(description="在固定的小样本上运行非定向 pgd")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/2026lw/AIGI_test/progan/airplane"))
    parser.add_argument("--baseline", type=Path,
                        default=DEMO_DIR / "baseline_progan_airplane_10perclass.csv")
    parser.add_argument("--eps", type=float, default=4 / 255)
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR / "pgd_results_centered")
    args = parser.parse_args()
    if not 0 <= args.eps <= 1:
        parser.error("eps 应在 0 到 1 之间")

    # 1. 直接从正常预测 CSV 读取名单，保证攻击前后比较的是同一批图片。
    with args.baseline.open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if not samples:
        raise ValueError("正常预测记录为空")
    root = args.data_root.resolve()
    for sample in samples:
        path = (root / sample["relative_path"]).resolve()
        label = int(sample["true_label"])
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"图片路径不正确：{path}")
        if path.relative_to(root).parts[0] != {0: "0_real", 1: "1_fake"}[label]:
            raise ValueError(f"图片目录与标签不匹配：{path}")

    # 2. 加载相同网络和权重，固定模型参数，梯度只用于修改输入图片。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    cnnspot = resnet50(num_classes=1)
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    cnnspot.load_state_dict(checkpoint["model"], strict=True)
    cnnspot.to(device).eval().requires_grad_(False)
    model = TwoClassCNNSpot(cnnspot).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    atk = torchattacks.PGD(
    model,
    eps=args.eps,
    alpha=0.25/ 255,
    steps=10,
    random_start=True,
    )
    atk.set_normalization_used(mean=MEAN, std=STD)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    rows, clean_images, attacked_images = [], [], []
    previewed_labels = set()
    maximum_adapter_error = 0.0
    print(f"计算设备：{device}；PGD 扰动上限：{args.eps:.8f}", flush=True)
    print(f"固定图片数量：{len(samples)}；攻击库：{torchattacks.__file__}", flush=True)

    for index, sample in enumerate(samples, start=1):
        label = int(sample["true_label"])
        with Image.open(root / sample["relative_path"]) as image:
            images = transform(image.convert("RGB")).unsqueeze(0).to(device)
        labels = torch.tensor([label], device=device, dtype=torch.long)

        # 先验证适配前后概率一致，并与已保存的正常预测结果对应。
        with torch.no_grad():
            original_z = cnnspot(images)
            original_probability = original_z.sigmoid().item()
            scores = model(images)
            clean_probability = scores.softmax(dim=1)[0, 1].item()
            clean_loss = nn.functional.cross_entropy(scores, labels).item()
        adapter_error = abs(clean_probability - original_probability)
        maximum_adapter_error = max(maximum_adapter_error, adapter_error)
        if adapter_error > 1e-6:
            raise RuntimeError("适配前后的概率不一致")
        if abs(original_probability - float(sample["fake_probability"])) > 1e-5:
            raise RuntimeError("本次正常预测与先前记录不一致，请检查模型和预处理")
        clean_prediction = int(clean_probability > 0.5)
        if clean_prediction != int(sample["predicted_label"]):
            raise RuntimeError("本次正常预测类别与先前记录不一致")

        clean_correct = clean_prediction == label
        # 4. 只对原本预测正确的图片生成对抗图片；此处必须保留梯度计算。
        if clean_correct:
            adv_images = atk(images, labels)
        else:
            adv_images = images.clone().detach()

        with torch.no_grad():
            adv_probability = model(adv_images).softmax(dim=1)[0, 1].item()
        adv_prediction = int(adv_probability > 0.5)

        # 返回值是标准化后的图片。先还原到像素空间，再检查范围和 L_inf。
        raw_clean = images * std + mean
        raw_adv = adv_images * std + mean
        if not torch.isfinite(raw_adv).all():
            raise RuntimeError("对抗图片出现非有限数值")
        if raw_adv.min().item() < -1e-6 or raw_adv.max().item() > 1 + 1e-6:
            raise RuntimeError("对抗图片超出合法像素范围")
        clean_pixels = raw_clean.clamp(0, 1).detach()
        adv_pixels = raw_adv.clamp(0, 1).detach()
        linf = (adv_pixels - clean_pixels).abs().max().item()
        if linf > args.eps + 1e-6:
            raise RuntimeError(f"扰动超限：{linf}")
        adv_correct = adv_prediction == label
        success = clean_correct and not adv_correct
        rows.append({
            "relative_path": sample["relative_path"], "true_label": label,
            "clean_logit": original_z.item(), "clean_probability": clean_probability,
            "clean_prediction": clean_prediction, "clean_correct": clean_correct,
            "clean_cross_entropy": clean_loss, "adv_probability": adv_probability,
            "adv_prediction": adv_prediction, "adv_correct": adv_correct,
            "attack_success": success, "linf_pixel": linf,
            "negligible_change": linf <= 1e-6,
            "attack_attempted": clean_correct,
        })
        clean_images.append(clean_pixels.cpu())
        attacked_images.append(adv_pixels.cpu())
        print(
            f"[{index}/{len(samples)}] {sample['relative_path']} | "
            f"概率 {clean_probability:.6f} -> {adv_probability:.6f} | "
            f"类别 {clean_prediction} -> {adv_prediction} | "
            f"L_inf={linf:.8f} | "
            f"{'原本预测错误；跳过攻击' if not clean_correct else '攻击成功' if success else '未攻击成功'}",
            flush=True,
        )

        # 每类保存一对裁剪后的图片供观看；准确评估以浮点张量为准。
        if label not in previewed_labels:
            stem = f"label{label}_{Path(sample['relative_path']).stem}"
            save_image(clean_pixels, preview_dir / f"{stem}_clean.png")
            save_image(adv_pixels, preview_dir / f"{stem}_pgd.png")
            previewed_labels.add(label)

    with (args.output_dir / "pgd_comparison.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean_correct_count = sum(row["clean_correct"] for row in rows)
    success_count = sum(row["attack_success"] for row in rows)
    class_statistics = []
    for class_label in (0, 1):
        class_rows = [row for row in rows if row["true_label"] == class_label]
        class_correct = sum(row["clean_correct"] for row in class_rows)
        class_success = sum(row["attack_success"] for row in class_rows)
        class_statistics.append({
            "label": class_label, "sample_count": len(class_rows),
            "clean_correct_count": class_correct, "attack_attempted_count": class_correct,
            "attack_success_count": class_success,
            "attack_success_rate_on_clean_correct": (
                class_success / class_correct if class_correct else None),
        })
    summary = {
        "seed": 2026,
        "alpha_pixel": atk.alpha,
        "steps": atk.steps,
        "random_start": atk.random_start,
        "attack": "pgd", "mode": "untargeted", "eps_pixel": args.eps,
        "output_adapter": "[-z/2,z/2]; softmax fake probability equals sigmoid(z)",
        "data_root": str(root), "baseline_manifest": str(args.baseline.resolve()),
        "weight_path": str(WEIGHT_PATH), "torch_version": torch.__version__,
        "torchattacks_version": torchattacks.__version__, "sample_count": len(rows),
        "clean_correct_count": clean_correct_count, "attack_success_count": success_count,
        "attack_scope": "originally_correct_only",
        "attack_attempted_count": clean_correct_count,
        "skipped_clean_incorrect_count": len(rows) - clean_correct_count,
        "class_statistics": class_statistics,
        "clean_accuracy": clean_correct_count / len(rows),
        "adversarial_accuracy": sum(row["adv_correct"] for row in rows) / len(rows),
        "attack_success_rate_on_clean_correct": (
            success_count / clean_correct_count if clean_correct_count else None),
        "max_linf_pixel": max(row["linf_pixel"] for row in rows),
        "max_adapter_probability_error": maximum_adapter_error,
        "negligible_change_count": sum(
            row["attack_attempted"] and row["negligible_change"] for row in rows),
        "negligible_change_scope": "attempted_images_only",
        "unchanged_return_count": sum(row["negligible_change"] for row in rows),
        "normalization_mean": MEAN, "normalization_std": STD,
        "notes": "只对本次固定名单成立；仅攻击原本预测正确的图片，原本预测错误的图片保留原图。"
                 "攻击成功率以原本预测正确的图片为分母；攻击后准确率以整份名单为分母。"
                 "negligible_change_count 只统计实际尝试攻击的图片。"
                 "PNG 为量化预览，浮点张量保存在 pgd_images.pt。",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({
        "clean_images": torch.cat(clean_images),
        "adv_images": torch.cat(attacked_images),
        "labels": torch.tensor([int(sample["true_label"]) for sample in samples]),
        "relative_paths": [sample["relative_path"] for sample in samples],
        "pixel_range": [0.0, 1.0], "eps": args.eps,
        "attack_attempted": torch.tensor([row["attack_attempted"] for row in rows]),
    }, args.output_dir / "pgd_images.pt")
    print(f"实际尝试攻击：{clean_correct_count} 张；"
          f"跳过原本预测错误：{summary['skipped_clean_incorrect_count']} 张", flush=True)
    for class_record in class_statistics:
        class_name = {0: "真实图→误判为生成图", 1: "生成图→误判为真实图"}[class_record["label"]]
        class_rate = class_record["attack_success_rate_on_clean_correct"]
        class_rate_text = f"{class_rate:.2%}" if class_rate is not None else "不适用（没有原本预测正确的样本）"
        print(f"{class_name}：新增成功 {class_record['attack_success_count']}/"
              f"{class_record['clean_correct_count']}；成功率 {class_rate_text}", flush=True)
    print(f"正常准确率：{summary['clean_accuracy']:.2%}；"
          f"攻击后准确率：{summary['adversarial_accuracy']:.2%}", flush=True)
    print(f"攻击成功：{success_count}/{clean_correct_count}（以原本判断正确的图片为分母）", flush=True)
    print(f"最大像素扰动：{summary['max_linf_pixel']:.8f}；结果目录：{args.output_dir}", flush=True)
    if args.eps > 0 and summary["negligible_change_count"]:
        print(f"注意：{summary['negligible_change_count']} 张图片的修改接近零，"
              "需要进一步检查梯度数值，不能据此判断模型抗攻击能力。", flush=True)


if __name__ == "__main__":
    main()
