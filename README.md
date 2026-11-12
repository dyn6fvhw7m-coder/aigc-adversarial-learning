# AIGC 图像检测与对抗攻击学习

使用 CNNSpot 预训练检测器学习 FGSM、PGD、APGD-CE、FAB-T、Square 和攻击组合，并保存可复查的参数与逐图结果。

## 目录

- `cnnspot_demo/`：学习脚本、固定图片名单、逐图 CSV、实验 summary.json 和 Markdown 记录。
- `adversarial-attacks-pytorch-master/`：下载的 torchattacks 源码；保留本地学习注释与修改。
- `AIGCDetectBenchmark-main/AIGCDetectBenchmark-main/`：下载的检测基准源码。
- `tools/check_upload.py`：提交前检查已暂存文件中的大文件和常见凭据。
- `sync-github.ps1`：把本地更改提交并推送到配置好的 GitHub 仓库。

原始第三方代码的来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 环境与本地资源

已验证的实验环境记录为 PyTorch 2.11.0+cu128、torchattacks 3.5.1。torchattacks 从仓库内源码导入。

模型权重、数据集、虚拟环境以及保存对抗图片的 `.pt` 张量不上传。使用前需在本机准备：

1. CNNSpot 权重：`AIGCDetectBenchmark-main/AIGCDetectBenchmark-main/weights/classifier/CNNSpot.pth`。
2. 直接包含 `0_real` 和 `1_fake` 的图片目录。
3. 安装匹配设备的 PyTorch、torchvision，以及 Pillow、NumPy、SciPy、tqdm 等实际依赖。上游依赖清单保留在各项目目录。

脚本的 `--help` 说明可用参数；数据目录、基线名单、扰动上限和结果位置使用命令行指定。部分早期脚本的默认路径对应原实验电脑，应按实际位置传入 `--data-root`、`--baseline` 和 `--output-dir`。

## 示例

```powershell
python .\cnnspot_demo\predict_samples.py --data-root "D:\2026lw\AIGI_test\progan\airplane" --max-per-class 10 --output .\cnnspot_demo\baseline_progan_airplane_10perclass.csv
```

执行攻击前，应核对脚本内的步数、步长、随机种子及归一化设置。历史运行参数以每个结果目录中的 `summary.json` 为准，不能用当前源文件的默认值反推旧实验。

## 实验口径

- 标签 0 为真实图，1 为生成图；CNNSpot 的单 logit 适配为 `[-z/2, z/2]`。
- 扰动上限在 `[0,1]` 像素空间度量，使用 L∞ 范数。
- 攻击成功率分母为原本判断正确的图片。新流程跳过原本判断错误的图片，并分别统计两个类别。
- PNG 是量化预览，精确攻击评估使用本地保存的浮点张量。
- `attack_ensemble.py` 是二分类自定义组合，省略 APGD-T，不能标为完整标准 AutoAttack。
- `attack_apgdt_margin.py` 是 APGD-T-Margin 自定义损失改造；它与原版 targeted DLR 不同，也不是完整标准 AutoAttack。
