from pathlib import Path
import torch

folder = Path(__file__).resolve().parent

run1 = torch.load(
    folder / "pgd_deterministic_run1" / "pgd_images.pt",
    map_location="cpu", weights_only=True,
)
run2 = torch.load(
    folder / "pgd_deterministic_run2" / "pgd_images.pt",
    map_location="cpu", weights_only=True,
)

if run1["relative_paths"] != run2["relative_paths"]:
    raise ValueError("两次图片名单或顺序不同，不能直接比较")

images1 = run1["adv_images"]
images2 = run2["adv_images"]

max_difference = (images1 - images2).abs().max().item()

print("最大像素差值：", max_difference)
print("图片数据完全相同：", torch.equal(images1, images2))