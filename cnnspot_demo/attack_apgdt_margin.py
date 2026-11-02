"""二分类 APGD-T-Margin 小样本实验；自定义损失改造，非标准 AutoAttack。"""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from attack_pgd import MEAN, STD, TwoClassCNNSpot
from predict_samples import DEMO_DIR, WEIGHT_PATH, resnet50
from torchattacks.attacks.apgdt import APGDT
import torchattacks


class BinaryAPGDTMargin(APGDT):
    """继承原库搜索流程，以二分类目标 margin 替换 targeted DLR。"""

    def __init__(self, model, eps=8 / 255, steps=10, seed=2026):
        super().__init__(model, norm="Linf", eps=eps, steps=steps,
                         n_restarts=1, seed=seed, eot_iter=1,
                         rho=0.75, verbose=False, n_classes=2)
        self.attack = "APGD-T-Margin"

    def dlr_loss_targeted(self, logits, labels, y_target):
        # 沿用父类接口名称。该函数现在计算 margin，而不是 DLR。
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("本改造仅适用于每张图片输出两个 logits 的模型")
        indices = torch.arange(logits.shape[0], device=logits.device)
        # 0=真实图、1=生成图。二分类唯一的错误目标是原始标签的另一类。
        # 显式选择 1-label，避免原始分数相等时排序选到原始标签。
        target_labels = 1 - labels
        target_scores = logits[indices, target_labels]
        original_label_scores = logits[indices, labels]
        # 原库使用梯度上升：让希望误判成的类别分数超过原始标签分数。
        return target_scores - original_label_scores


def main():
    parser = argparse.ArgumentParser(description="二分类 APGD-T-Margin 改造实验，默认每类一张")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/2026lw/AIGI_test/progan/airplane"))
    parser.add_argument("--baseline", type=Path,
                        default=DEMO_DIR / "baseline_progan_airplane_10perclass.csv")
    parser.add_argument("--max-per-class", type=int, default=1)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--output-dir", type=Path,
                        default=DEMO_DIR / "apgdt_margin_eps8_2samples_seed2026")
    args = parser.parse_args()
    if not 0 <= args.eps <= 1 or args.max_per_class < 1:
        parser.error("eps 应在 [0,1]；max-per-class 必须大于 0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("结果目录已有文件，请换一个新的 --output-dir")
    root = args.data_root.resolve()
    with args.baseline.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    samples, counts = [], {0: 0, 1: 0}
    for sample in manifest:
        label = int(sample["true_label"])
        if label not in counts:
            raise ValueError("基线标签必须为 0 或 1")
        if counts[label] < args.max_per_class:
            path = (root / sample["relative_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"图片路径不合法或不存在：{path}")
            if path.relative_to(root).parts[0] != {0: "0_real", 1: "1_fake"}[label]:
                raise ValueError(f"目录与标签不一致：{path}")
            samples.append(sample)
            counts[label] += 1
    if not all(counts.values()):
        raise ValueError("本次检查需要同时选到真实图与生成图")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnnspot = resnet50(num_classes=1)
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    cnnspot.load_state_dict(checkpoint["model"], strict=True)
    cnnspot.to(device).eval().requires_grad_(False)
    model = TwoClassCNNSpot(cnnspot).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    atk = BinaryAPGDTMargin(model, eps=args.eps, steps=10, seed=2026)
    atk.set_normalization_used(mean=MEAN, std=STD)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    with (args.output_dir / "selected_samples.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(f"设备：{device}；APGD-T-Margin 改造实验，非标准 AutoAttack", flush=True)
    print(f"eps={args.eps:.8f}；steps={atk.steps}；n_restarts=1；seed=2026", flush=True)
    print(f"选取 {len(samples)} 张（各类最多 {args.max_per_class} 张）；"
          "目标类别=1-原始标签；最大化目标分数减原始标签分数", flush=True)
    rows, clean_images, adv_images_list = [], [], []
    previewed_labels = set()
    for index, sample in enumerate(samples, start=1):
        label = int(sample["true_label"])
        target = 1 - label
        labels = torch.tensor([label], device=device)
        targets = 1 - labels
        with Image.open(root / sample["relative_path"]) as image:
            images = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(images)
            clean_probability = scores.softmax(1)[0, 1].item()
            original_probability = cnnspot(images).sigmoid().item()
            clean_prediction = int(scores.argmax(1).item())
            clean_margin = atk.dlr_loss_targeted(scores, labels, targets).item()
        adapter_error = abs(clean_probability - original_probability)
        if adapter_error > 1e-6:
            raise RuntimeError("适配前后概率不一致")
        if (abs(original_probability - float(sample["fake_probability"])) > 1e-5
                or int(original_probability > 0.5) != int(sample["predicted_label"])):
            raise RuntimeError("当前预测与固定基线不一致，请检查权重和预处理")
        if clean_prediction != int(sample["predicted_label"]):
            raise RuntimeError("原始分数预测与基线概率阈值预测不一致")
        clean_correct = clean_prediction == label
        raw_clean = (images * std + mean).clamp(0, 1).detach()
        gradient_max = None
        if clean_correct:
            # 额外做一次检查：margin 对原始像素的梯度应有限且非零。
            probe = raw_clean.clone().requires_grad_(True)
            probe_loss = atk.dlr_loss_targeted(model((probe - mean) / std), labels, targets)
            gradient = torch.autograd.grad(probe_loss.sum(), probe)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError("margin 输入梯度出现非有限数值")
            gradient_max = gradient.abs().max().item()
            if gradient_max == 0:
                raise RuntimeError("本次 margin 输入梯度全为零，需要检查后再继续")
            adv_images = atk(images, labels)
        else:
            adv_images = images.clone().detach()
        with torch.no_grad():
            adv_scores = model(adv_images)
            adv_probability = adv_scores.softmax(1)[0, 1].item()
            adv_prediction = int(adv_scores.argmax(1).item())
            adv_margin = atk.dlr_loss_targeted(adv_scores, labels, targets).item()
        raw_adv = adv_images * std + mean
        if (not torch.isfinite(raw_adv).all() or raw_adv.min().item() < -1e-6
                or raw_adv.max().item() > 1 + 1e-6):
            raise RuntimeError("返回图片数值或像素范围不合法")
        adv_pixels = raw_adv.clamp(0, 1).detach()
        linf = (adv_pixels - raw_clean).abs().max().item()
        if linf > args.eps + 1e-6:
            raise RuntimeError(f"扰动超限：{linf}")
        success = clean_correct and adv_prediction == target
        row = {
            "relative_path": sample["relative_path"], "true_label": label,
            "target_label": target, "clean_probability": clean_probability,
            "clean_prediction": clean_prediction, "clean_correct": clean_correct,
            "adv_probability": adv_probability, "adv_prediction": adv_prediction,
            "adv_correct": adv_prediction == label, "attack_attempted": clean_correct,
            "attack_success": success, "clean_target_margin": clean_margin,
            "adv_target_margin": adv_margin, "linf_pixel": linf,
            "max_abs_margin_gradient_pixel": gradient_max,
            "adapter_probability_error": adapter_error,
        }
        rows.append(row)
        clean_images.append(raw_clean.cpu())
        adv_images_list.append(adv_pixels.cpu())
        status = "跳过原本错误的图片" if not clean_correct else "攻击成功" if success else "未成功"
        print(f"[{index}/{len(samples)}] {sample['relative_path']} | 原始标签 {label}，目标 {target} | "
              f"类别 {clean_prediction}→{adv_prediction} | 生成概率 "
              f"{clean_probability:.6f}→{adv_probability:.6f} | "
              f"margin {clean_margin:.6f}→{adv_margin:.6f} | L_inf={linf:.8f} | {status}", flush=True)
        if label not in previewed_labels:
            stem = f"label{label}_{Path(sample['relative_path']).stem}"
            save_image(raw_clean, preview_dir / f"{stem}_clean.png")
            save_image(adv_pixels, preview_dir / f"{stem}_apgdt_margin.png")
            previewed_labels.add(label)
    with (args.output_dir / "apgdt_margin_comparison.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    attempted = sum(row["attack_attempted"] for row in rows)
    succeeded = sum(row["attack_success"] for row in rows)
    library_source = Path(inspect.getfile(APGDT))
    summary = {
        "attack": "APGD-T-Margin (custom binary modification)",
        "standard_autoattack": False, "mode": "binary_targeted_search",
        "loss": "target_logit - original_label_logit", "target_rule": "1 - original_label",
        "base_class": "torchattacks.attacks.apgdt.APGDT",
        "base_source": str(library_source),
        "base_source_sha256": hashlib.sha256(library_source.read_bytes()).hexdigest(),
        "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": "https://github.com/fra31/auto-attack/issues/70",
        "norm": atk.norm, "eps_pixel": args.eps, "steps": atk.steps,
        "n_restarts": atk.n_restarts, "n_classes": 2, "seed": atk.seed,
        "rho": atk.thr_decr, "eot_iter": atk.eot_iter,
        "data_root": str(root), "baseline_manifest": str(args.baseline.resolve()),
        "baseline_manifest_sample_count": len(manifest), "selection_max_per_class": args.max_per_class,
        "weight_path": str(WEIGHT_PATH), "torch_version": torch.__version__,
        "torchattacks_version": torchattacks.__version__, "sample_count": len(rows),
        "clean_correct_count": attempted, "attack_success_count": succeeded,
        "skipped_clean_incorrect_count": len(rows) - attempted,
        "clean_accuracy": attempted / len(rows),
        "adversarial_accuracy": sum(row["adv_correct"] for row in rows) / len(rows),
        "attack_success_rate_on_clean_correct": succeeded / attempted if attempted else None,
        "max_linf_pixel": max(row["linf_pixel"] for row in rows),
        "normalization_mean": MEAN, "normalization_std": STD,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "notes": "继承原库 APGDT 搜索流程，覆盖 targeted DLR 为二分类 margin；"
                 "原始标签的另一类为唯一目标。margin 不具备 DLR 的尺度不变性。"
                 "这是需独立验证的自定义改造，不是完整标准 AutoAttack；不保证最小扰动。"
                 "成功标准为原本正确的图片被判为目标类，且扰动在上限内。"
                 "margin=0 的分数平局以 argmax 的实际预测为准。"
                 "PNG 为量化预览，准确评估使用 apgdt_margin_images.pt 中的浮点张量。",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"clean_images": torch.cat(clean_images), "adv_images": torch.cat(adv_images_list),
                "labels": torch.tensor([row["true_label"] for row in rows]),
                "target_labels": torch.tensor([row["target_label"] for row in rows]),
                "relative_paths": [row["relative_path"] for row in rows],
                "pixel_range": [0.0, 1.0], "eps": args.eps},
               args.output_dir / "apgdt_margin_images.pt")
    print(f"正常准确率：{summary['clean_accuracy']:.2%}；"
          f"攻击后准确率：{summary['adversarial_accuracy']:.2%}", flush=True)
    print(f"攻击成功：{succeeded}/{attempted}（原本预测正确的图片为分母）", flush=True)
    print(f"最大像素扰动：{summary['max_linf_pixel']:.8f}；结果目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
