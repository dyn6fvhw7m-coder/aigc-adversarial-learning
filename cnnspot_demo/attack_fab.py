"""FAB-T 组件实验：CNNSpot + torchattacks，从固定名单选择小样本。"""

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
    parser = argparse.ArgumentParser(description="在固定的小样本上运行 FAB 多目标搜索组件实验")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/2026lw/AIGI_test/progan/airplane"))
    parser.add_argument("--baseline", type=Path,
                        default=DEMO_DIR / "baseline_progan_airplane_10perclass.csv")
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="每类最多取多少张；不指定时使用正常预测名单中的全部图片")
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR / "fab_results")
    args = parser.parse_args()
    if not 0 <= args.eps <= 1:
        parser.error("eps 应在 0 到 1 之间")
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("max-per-class 必须是正整数")

    # 1. 直接从正常预测 CSV 读取名单，保证攻击前后比较的是同一批图片。
    with args.baseline.open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if not samples:
        raise ValueError("正常预测记录为空")
    baseline_manifest_sample_count = len(samples)
    root = args.data_root.resolve()
    for sample in samples:
        path = (root / sample["relative_path"]).resolve()
        label = int(sample["true_label"])
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"图片路径不正确：{path}")
        if path.relative_to(root).parts[0] != {0: "0_real", 1: "1_fake"}[label]:
            raise ValueError(f"图片目录与标签不匹配：{path}")

    # 小样本检查从同一份名单取图，按原顺序保留每类前几张。
    if args.max_per_class is not None:
        class_counts = {0: 0, 1: 0}
        selected_samples = []
        for sample in samples:
            label = int(sample["true_label"])
            if class_counts[label] < args.max_per_class:
                selected_samples.append(sample)
                class_counts[label] += 1
        if any(count == 0 for count in class_counts.values()):
            raise ValueError("小样本检查需要同时包含真实图和生成图")
        samples = selected_samples

    # 2. 加载相同网络和权重，固定模型参数，梯度只用于修改输入图片。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 3. 调用 FAB 多目标搜索组件：二分类仅有一个错误目标类别。
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    atk = torchattacks.FAB(
        model,
        norm="Linf",
        eps=args.eps,
        steps=10,
        n_restarts=1,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        seed=2026,
        multi_targeted=True,
        n_classes=2,
        verbose=False,
    )
    atk.set_normalization_used(mean=MEAN, std=STD)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selected_samples.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    preview_dir = args.output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    rows, clean_images, attacked_images = [], [], []
    previewed_labels = set()
    maximum_adapter_error = 0.0
    print(f"计算设备：{device}；FAB-T 扰动上限：{args.eps:.8f}", flush=True)
    print(f"步数：{atk.steps}；重启轮数：{atk.n_restarts}；"
          f"多目标搜索：{atk.multi_targeted}；类别数：{atk.n_target_classes + 1}；"
          f"随机种子：{atk.seed}", flush=True)
    print(f"alpha_max：{atk.alpha_max}；eta：{atk.eta}；beta：{atk.beta}", flush=True)
    if args.max_per_class is not None:
        print(f"小样本检查：每类最多 {args.max_per_class} 张；"
              f"源名单共 {baseline_manifest_sample_count} 张", flush=True)
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

        # 4. 这一句真正生成对抗图片；此处必须保留梯度计算。
        adv_images = atk(images, labels)

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
        clean_correct = clean_prediction == label
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
        })
        clean_images.append(clean_pixels.cpu())
        attacked_images.append(adv_pixels.cpu())
        print(
            f"[{index}/{len(samples)}] {sample['relative_path']} | "
            f"概率 {clean_probability:.6f} -> {adv_probability:.6f} | "
            f"类别 {clean_prediction} -> {adv_prediction} | "
            f"L_inf={linf:.8f} | "
            f"{'攻击成功' if success else '未攻击成功'}",
            flush=True,
        )

        # 每类保存一对裁剪后的图片供观看；准确评估以浮点张量为准。
        if label not in previewed_labels:
            stem = f"label{label}_{Path(sample['relative_path']).stem}"
            save_image(clean_pixels, preview_dir / f"{stem}_clean.png")
            save_image(adv_pixels, preview_dir / f"{stem}_fab.png")
            previewed_labels.add(label)

    with (args.output_dir / "fab_comparison.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean_correct_count = sum(row["clean_correct"] for row in rows)
    success_count = sum(row["attack_success"] for row in rows)
    summary = {
        "norm": atk.norm,
        "n_restarts": atk.n_restarts,
        "alpha_max": atk.alpha_max,
        "eta": atk.eta,
        "beta": atk.beta,
        "multi_targeted": atk.multi_targeted,
        "n_classes": atk.n_target_classes + 1,
        "seed": atk.seed,
        "steps": atk.steps,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "attack": "FAB-T", "mode": "untargeted", "eps_pixel": args.eps,
        "output_adapter": "[-z/2,z/2]; softmax fake probability equals sigmoid(z)",
        "data_root": str(root), "baseline_manifest": str(args.baseline.resolve()),
        "baseline_manifest_sample_count": baseline_manifest_sample_count,
        "selection_max_per_class": args.max_per_class,
        "selected_manifest": str((args.output_dir / "selected_samples.csv").resolve()),
        "weight_path": str(WEIGHT_PATH), "torch_version": torch.__version__,
        "torchattacks_version": torchattacks.__version__, "sample_count": len(rows),
        "clean_correct_count": clean_correct_count, "attack_success_count": success_count,
        "clean_accuracy": clean_correct_count / len(rows),
        "adversarial_accuracy": sum(row["adv_correct"] for row in rows) / len(rows),
        "attack_success_rate_on_clean_correct": (
            success_count / clean_correct_count if clean_correct_count else None),
        "max_linf_pixel": max(row["linf_pixel"] for row in rows),
        "max_adapter_probability_error": maximum_adapter_error,
        "negligible_change_count": sum(row["negligible_change"] for row in rows),
        "normalization_mean": MEAN, "normalization_std": STD,
        "notes": "只对本次选定的小样本成立；本实验为 FAB 多目标搜索单组件，"
                 "并非完整标准 AutoAttack；成功标准仍为原本正确的样本被误判且扰动不超限。"
                 "未攻击成功的样本可能返回原图。"
                 "PNG 为量化预览，浮点张量保存在 fab_images.pt。",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({
        "clean_images": torch.cat(clean_images),
        "adv_images": torch.cat(attacked_images),
        "labels": torch.tensor([int(sample["true_label"]) for sample in samples]),
        "relative_paths": [sample["relative_path"] for sample in samples],
        "pixel_range": [0.0, 1.0], "eps": args.eps,
    }, args.output_dir / "fab_images.pt")
    print(f"正常准确率：{summary['clean_accuracy']:.1%}；"
          f"攻击后准确率：{summary['adversarial_accuracy']:.1%}", flush=True)
    print(f"攻击成功：{success_count}/{clean_correct_count}（以原本判断正确的图片为分母）", flush=True)
    print(f"最大像素扰动：{summary['max_linf_pixel']:.8f}；结果目录：{args.output_dir}", flush=True)
    if args.eps > 0 and summary["negligible_change_count"]:
        print(f"说明：{summary['negligible_change_count']} 张返回图片的修改接近零。"
              "本库 FAB 在扰动上限内未找到成功结果时可能返回原图；"
              "当前结果只反映本次参数设置下的攻击效果。", flush=True)


if __name__ == "__main__":
    main()
