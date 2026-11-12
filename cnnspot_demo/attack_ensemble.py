"""二分类自定义组合：APGD-CE → FAB-T → Square，记录每阶段新增成功数。"""

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


class RecordedMultiAttack(torchattacks.MultiAttack):
    """使用原库的统计回调，取得每个阶段剩余的图片数量。"""

    def __init__(self, attacks):
        super().__init__(attacks, verbose=False)
        self.last_remaining_counts = []
        self._accumulate_multi_atk_records = True

    def _update_multi_atk_records(self, multi_atk_records):
        # 原库 forward 在这里提供 [初始数量, 第一阶段剩余, ...]。
        # 若提前全部成功，列表中不会出现后续未执行阶段。
        self.last_remaining_counts = list(multi_atk_records)


def make_stage_records(remaining, stage_names):
    """将原库的剩余数量转为阶段记录，提前结束的后续阶段记为未执行。"""
    remaining = list(remaining)
    if not remaining or len(remaining) > len(stage_names) + 1:
        raise RuntimeError(f"组合阶段记录长度不符合预期：{remaining}")
    if any(not isinstance(count, int) or count < 0 for count in remaining):
        raise RuntimeError(f"组合阶段数量不合法：{remaining}")
    if any(after > before for before, after in zip(remaining, remaining[1:])):
        raise RuntimeError(f"组合阶段剩余数量增加：{remaining}")
    if len(remaining) < len(stage_names) + 1 and remaining[-1] != 0:
        raise RuntimeError(f"组合阶段记录提前结束但仍有剩余图片：{remaining}")
    remaining += [0] * (len(stage_names) + 1 - len(remaining))
    return [
        {"attack": name, "attempted_count": remaining[index],
         "new_success_count": remaining[index] - remaining[index + 1],
         "remaining_count": remaining[index + 1],
         "executed": remaining[index] > 0}
        for index, name in enumerate(stage_names)
    ]


def build_ensemble(model, eps, *, include_margin=False):
    """构建三个原库组件；可在第二阶段加入已独立验证的二分类 margin 改造。"""
    apgd_ce = torchattacks.APGD(
        model, norm="Linf", eps=eps, steps=10, n_restarts=1,
        seed=2026, loss="ce", eot_iter=1, rho=0.75, verbose=False,
    )
    fab_t = torchattacks.FAB(
        model, norm="Linf", eps=eps, steps=10, n_restarts=1,
        alpha_max=0.1, eta=1.05, beta=0.9, seed=2026,
        multi_targeted=True, n_classes=2, verbose=False,
    )
    square = torchattacks.Square(
        model, norm="Linf", eps=eps, n_queries=1000, n_restarts=1,
        p_init=0.8, loss="margin", resc_schedule=True, seed=2026, verbose=False,
    )
    attacks = [apgd_ce, fab_t, square]
    stage_names = ["APGD-CE", "FAB-T", "Square"]
    stage_keys = ["apgd_ce", "fab_t", "square"]
    parameters = [
        {"attack": "APGD-CE", "norm": apgd_ce.norm, "eps_pixel": apgd_ce.eps,
         "steps": apgd_ce.steps, "n_restarts": apgd_ce.n_restarts,
         "seed": apgd_ce.seed, "loss": apgd_ce.loss, "eot_iter": apgd_ce.eot_iter,
         "rho": apgd_ce.thr_decr},
        {"attack": "FAB-T", "norm": fab_t.norm, "eps_pixel": fab_t.eps,
         "steps": fab_t.steps, "n_restarts": fab_t.n_restarts,
         "seed": fab_t.seed, "alpha_max": fab_t.alpha_max,
         "eta": fab_t.eta, "beta": fab_t.beta,
         "multi_targeted": fab_t.multi_targeted, "n_classes": fab_t.n_target_classes + 1},
        {"attack": "Square", "norm": square.norm, "eps_pixel": square.eps,
         "n_queries": square.n_queries, "n_restarts": square.n_restarts,
         "seed": square.seed, "p_init": square.p_init, "loss": square.loss,
         "resc_schedule": square.rescale_schedule},
    ]
    if include_margin:
        from attack_apgdt_margin import BinaryAPGDTMargin

        margin = BinaryAPGDTMargin(model, eps=eps, steps=10, seed=2026)
        attacks.insert(1, margin)
        stage_names.insert(1, "APGD-T-Margin")
        stage_keys.insert(1, "apgdt_margin")
        parameters.insert(1, {
            "attack": "APGD-T-Margin", "norm": margin.norm, "eps_pixel": margin.eps,
            "steps": margin.steps, "n_restarts": margin.n_restarts,
            "seed": margin.seed, "n_classes": 2, "eot_iter": margin.eot_iter,
            "rho": margin.thr_decr,
            "loss": "target_logit - original_label_logit", "target_rule": "1 - label",
            "implementation": "attack_apgdt_margin.BinaryAPGDTMargin",
        })
    atk = RecordedMultiAttack(attacks)
    atk.set_normalization_used(mean=MEAN, std=STD)
    return atk, stage_names, stage_keys, parameters


def main(*, include_margin=False):
    order = "APGD-CE → APGD-T-Margin → FAB-T → Square" if include_margin else "APGD-CE → FAB-T → Square"
    default_output = ("ensemble_margin_eps8_20samples_seed2026" if include_margin
                      else "ensemble_apgd_fab_square_eps8_seed2026")
    parser = argparse.ArgumentParser(description=f"二分类自定义组合：{order}")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/2026lw/AIGI_test/progan/airplane"))
    parser.add_argument("--baseline", type=Path,
                        default=DEMO_DIR / "baseline_progan_airplane_10perclass.csv")
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--output-dir", type=Path,
                        default=DEMO_DIR / default_output)
    args = parser.parse_args()
    if not 0 <= args.eps <= 1:
        parser.error("eps 应在 0 到 1 之间")
    # 防止旧的单组件结果被新实验覆盖。
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("结果目录已有文件，请换一个新的 --output-dir 保存本次实验")

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

    atk, stage_names, stage_keys, component_parameters = build_ensemble(
        model, args.eps, include_margin=include_margin)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    rows, clean_images, attacked_images = [], [], []
    previewed_labels = set()
    maximum_adapter_error = 0.0
    print(f"计算设备：{device}；组合顺序：{' → '.join(stage_names)}；"
          f"扰动上限：{args.eps:.8f}", flush=True)
    for component in component_parameters:
        budget = (f"步数：{component['steps']}" if "steps" in component
                  else f"每轮查询预算参数：{component['n_queries']}")
        print(f"{component['attack']}：{budget}；重启轮数：{component['n_restarts']}；"
              f"随机种子：{component['seed']}", flush=True)
    print("实验类型：二分类自定义组合；非完整标准 AutoAttack。", flush=True)
    if include_margin:
        print("第二阶段以目标分数减原始标签分数替换 targeted DLR；目标类别为 1-label。", flush=True)
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
        clean_prediction = scores.argmax(dim=1).item()
        if clean_prediction != int(sample["predicted_label"]):
            raise RuntimeError("本次正常预测类别与先前记录不一致")

        clean_correct = clean_prediction == label
        # 4. 使用原库组合流程生成对抗图片，只攻击原本预测正确的图片。
        if clean_correct:
            adv_images = atk(images, labels)
            remaining = list(atk.last_remaining_counts)
        else:
            adv_images = images.clone().detach()
            remaining = [0]
        if not remaining or remaining[0] != int(clean_correct):
            raise RuntimeError(f"组合阶段初始数量不符合预期：{remaining}")
        sample_stages = make_stage_records(remaining, stage_names)

        with torch.no_grad():
            adv_scores = model(adv_images)
            adv_probability = adv_scores.softmax(dim=1)[0, 1].item()
            adv_prediction = adv_scores.argmax(dim=1).item()

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
        if sum(stage["new_success_count"] for stage in sample_stages) != int(success):
            raise RuntimeError("阶段成功数量与最终预测不一致，请检查组合记录与分类阈值")
        successful_stage = next((stage["attack"] for stage in sample_stages
                                 if stage["new_success_count"]), "none")
        if clean_correct:
            messages = [f"{stage['attack']} {'成功' if stage['new_success_count'] else '未成功'}"
                        for stage in sample_stages if stage["executed"]]
            skipped = [stage["attack"] for stage in sample_stages if not stage["executed"]]
            if skipped:
                messages.append("跳过 " + "、".join(skipped))
            stage_message = "；".join(messages)
        else:
            stage_message = "原本预测错误；跳过攻击"
        stage_columns = {}
        for key, stage in zip(stage_keys, sample_stages):
            stage_columns[f"{key}_attempted"] = stage["attempted_count"]
            stage_columns[f"{key}_new_success"] = stage["new_success_count"]
        rows.append({
            "relative_path": sample["relative_path"], "true_label": label,
            "clean_logit": original_z.item(), "clean_probability": clean_probability,
            "clean_prediction": clean_prediction, "clean_correct": clean_correct,
            "clean_cross_entropy": clean_loss, "adv_probability": adv_probability,
            "adv_prediction": adv_prediction, "adv_correct": adv_correct,
            "attack_success": success, "linf_pixel": linf,
            "negligible_change": linf <= 1e-6,
            **stage_columns,
            "stage_remaining_counts": json.dumps(remaining),
            "successful_stage": successful_stage,
        })
        clean_images.append(clean_pixels.cpu())
        attacked_images.append(adv_pixels.cpu())
        print(f"[{index}/{len(samples)}] {sample['relative_path']} | "
              f"概率 {clean_probability:.6f} -> {adv_probability:.6f} | "
              f"类别 {clean_prediction} -> {adv_prediction} | "
              f"L_inf={linf:.8f} | {stage_message}", flush=True)

        # 每类保存一对裁剪后的图片供观看；准确评估以浮点张量为准。
        if label not in previewed_labels:
            stem = f"label{label}_{Path(sample['relative_path']).stem}"
            save_image(clean_pixels, preview_dir / f"{stem}_clean.png")
            save_image(adv_pixels, preview_dir / f"{stem}_ensemble.png")
            previewed_labels.add(label)

    with (args.output_dir / "ensemble_comparison.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean_correct_count = sum(row["clean_correct"] for row in rows)
    success_count = sum(row["attack_success"] for row in rows)
    stage_statistics = [
        {"attack": name,
         "attempted_count": sum(row[f"{key}_attempted"] for row in rows),
         "new_success_count": sum(row[f"{key}_new_success"] for row in rows)}
        for name, key in zip(stage_names, stage_keys)
    ]
    cumulative_success = 0
    for stage in stage_statistics:
        cumulative_success += stage["new_success_count"]
        stage["cumulative_success_count"] = cumulative_success
        stage["remaining_count"] = clean_correct_count - cumulative_success
        stage["executed"] = stage["attempted_count"] > 0
    if cumulative_success != success_count:
        raise RuntimeError("组合阶段汇总与最终成功数量不一致")
    summary = {
        "attack": "CustomBinaryEnsemble(" + ",".join(stage_names) + ")",
        "mode": "untargeted", "eps_pixel": args.eps,
        "attack_order": stage_names, "norm": "Linf", "seed": 2026,
        "component_parameters": component_parameters,
        "standard_autoattack": False,
        "omitted_component": (None if include_margin else
                              "APGD-T: targeted DLR 与当前二分类输出不兼容。"),
        "replacement_component": ({
            "original": "APGD-T with targeted DLR",
            "replacement": "APGD-T-Margin",
            "reason": "targeted DLR 需要至少四个类别分数；当前输出为两个分数。",
        } if include_margin else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "stage_statistics": stage_statistics,
        "output_adapter": "[-z/2,z/2]; softmax fake probability equals sigmoid(z)",
        "data_root": str(root), "baseline_manifest": str(args.baseline.resolve()),
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
        "notes": f"只对该固定子集成立；这是 {order} 二分类自定义组合，"
                 "并非完整标准 AutoAttack；组件、损失与预算以本次配置为准。"
                 "每种攻击都从尚未成功样本的原图重新搜索，只攻击原本预测正确的图片。"
                 "前面阶段全部成功时，后续阶段会跳过；跳过不等于后续组件已执行验证。"
                 "n_queries 为每轮搜索预算参数，并非实际已用查询次数。"
                 "PNG 为量化预览，浮点张量保存在 ensemble_images.pt。",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({
        "clean_images": torch.cat(clean_images),
        "adv_images": torch.cat(attacked_images),
        "labels": torch.tensor([int(sample["true_label"]) for sample in samples]),
        "relative_paths": [sample["relative_path"] for sample in samples],
        "pixel_range": [0.0, 1.0], "eps": args.eps,
        "attack_order": stage_names,
        "successful_stage": [row["successful_stage"] for row in rows],
    }, args.output_dir / "ensemble_images.pt")
    for stage in stage_statistics:
        stage_status = "" if stage["executed"] else "（未执行）"
        print(f"{stage['attack']} 阶段：尝试 {stage['attempted_count']} 张；"
              f"新增成功 {stage['new_success_count']} 张；"
              f"累计成功 {stage['cumulative_success_count']} 张；"
              f"剩余 {stage['remaining_count']} 张{stage_status}", flush=True)
    print(f"正常准确率：{summary['clean_accuracy']:.1%}；"
          f"攻击后准确率：{summary['adversarial_accuracy']:.1%}", flush=True)
    print(f"攻击成功：{success_count}/{clean_correct_count}（以原本判断正确的图片为分母）", flush=True)
    print(f"最大像素扰动：{summary['max_linf_pixel']:.8f}；结果目录：{args.output_dir}", flush=True)
    if args.eps > 0 and summary["negligible_change_count"]:
        print(f"说明：{summary['negligible_change_count']} 张返回图片的修改接近零；"
              "零修改本身不能证明未搜索，也不能证明模型稳健。", flush=True)


if __name__ == "__main__":
    main()
