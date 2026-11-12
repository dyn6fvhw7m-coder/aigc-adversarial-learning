import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn, idctn
from scipy.fftpack import dct
from skimage.metrics import structural_similarity
from torchvision.models import resnet50 as torchvision_resnet50


PROJECT = Path(__file__).resolve().parents[1]
SAMPLES = (
    PROJECT / "cnnspot_demo" / "ensemble_margin_eps8_400samples_seed2026"
    / "ensemble_images.pt"
)
BENCHMARK = PROJECT / "AIGCDetectBenchmark-main" / "AIGCDetectBenchmark-main"


def load_cpu(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_cnnspot(benchmark, device):
    sys.path.insert(0, str(benchmark))
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    checkpoint = load_cpu(benchmark / "weights" / "classifier" / "CNNSpot.pth")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def predict(images):
        pixels = torch.from_numpy(images).to(device, dtype=torch.float32)
        return model((pixels - mean) / std).flatten().cpu().numpy()

    return predict


def load_fredect(benchmark, device):
    weights = benchmark / "weights"
    model = torchvision_resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    checkpoint = load_cpu(weights / "classifier" / "DCTAnalysis.pth")
    model.load_state_dict(checkpoint["netC"], strict=True)
    model.to(device).eval()
    mean = load_cpu(weights / "auxiliary" / "dct_mean").numpy()
    std = np.sqrt(load_cpu(weights / "auxiliary" / "dct_var").numpy())

    @torch.inference_mode()
    def predict(pixels):
        # Same orthonormal DCT, log magnitude and statistics as the benchmark.
        coeff = dct(pixels, type=2, axis=-2, norm="ortho")
        coeff = dct(coeff, type=2, axis=-1, norm="ortho")
        features = (np.log(np.abs(coeff) + 1e-12) - mean) / std
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        return model(inputs).flatten().cpu().numpy()

    return predict


def to_uint8(images):
    return np.rint(np.clip(images, 0, 1) * 255).astype(np.uint8)


def evaluate(data, detector, predict):
    labels = data["labels"].numpy()
    rows = []
    for start in range(0, len(labels), 16):
        stop = min(start + 16, len(labels))
        clean = data["clean_images"][start:stop].double().numpy()
        adv = data["adv_images"][start:stop].double().numpy()
        coeff = dctn(adv, type=2, axes=(-2, -1), norm="ortho")
        restored = idctn(coeff, type=2, axes=(-2, -1), norm="ortho")

        if detector == "fredect":
            inputs = [to_uint8(images) for images in (clean, adv, restored)]
            changed_values = (inputs[1] != inputs[2]).sum(axis=(1, 2, 3))
        else:
            inputs = (clean, adv, restored)

        scores = np.stack([predict(images) for images in inputs], axis=1)
        predictions = (scores > 0).astype(int)
        for j, i in enumerate(range(start, stop)):
            row = [data["relative_paths"][i], labels[i], *predictions[j]]
            if detector == "fredect":
                row.extend([*scores[j].astype(float), changed_values[j]])
            else:
                pairs = [(clean[j], adv[j]), (adv[j], restored[j]),
                         (clean[j], restored[j])]
                row.extend(structural_similarity(a, b, data_range=1.0, channel_axis=0)
                           for a, b in pairs)
                row.append(np.abs(adv[j] - restored[j]).max())
            rows.append(row)
        print(f"Processed {stop}/{len(labels)}", flush=True)
    return rows


def print_results(rows, detector):
    label, clean, adv, restored = np.asarray([row[1:5] for row in rows]).T
    clean_correct = clean == label
    print(f"Clean correct: {clean_correct.sum()}/{len(label)}")
    groups = [("all", clean_correct)]
    if detector == "fredect":
        groups.extend([("real", clean_correct & (label == 0)),
                       ("fake", clean_correct & (label == 1))])
    for name, predicted in (("adv", adv), ("restored", restored)):
        for group, mask in groups:
            success = (mask & (predicted != label)).sum()
            total = mask.sum()
            rate = f"{success}/{total}" if total else "N/A"
            print(f"Success ({name}, {group}): {rate}")
    print(f"Prediction changes (adv -> restored): {(adv != restored).sum()}")

    if detector == "fredect":
        print(f"Images with any 8-bit pixel change: {sum(row[-1] > 0 for row in rows)}")
    else:
        metrics = np.asarray([row[5:] for row in rows])
        names = ("clean/adv", "adv/restored", "clean/restored")
        for name, score in zip(names, metrics[:, :3].mean(axis=0)):
            print(f"Mean SSIM ({name}): {score:.12f}")
        print(f"Max round-trip pixel error: {metrics[:, 3].max():.3e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved DCT round-trip samples.")
    parser.add_argument("detector", choices=("cnnspot", "fredect"))
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output", type=Path, help="Optional path for a new per-image CSV.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Choose a new output path: {args.output}")

    data = load_cpu(args.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.detector == "cnnspot":
        predict = load_cnnspot(args.benchmark, device)
        columns = ["ssim_clean_adv", "ssim_adv_restored", "ssim_clean_restored",
                   "roundtrip_max_pixel_error"]
    else:
        predict = load_fredect(args.benchmark, device)
        columns = ["clean_logit", "adv_logit", "restored_logit",
                   "changed_8bit_channel_values"]

    print(f"Device: {device} | samples: {len(data['labels'])}", flush=True)
    rows = evaluate(data, args.detector, predict)
    print_results(rows, args.detector)
    if args.output is not None:
        with args.output.open("x", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label", "clean_pred", "adv_pred",
                             "restored_pred", *columns])
            writer.writerows(rows)
        print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()