# 第三方来源与本地修改

本仓库是学习工作区，包含下载的第三方源码与自行编写的实验脚本；不表示上游项目由本仓库作者原创。

## adversarial-attacks-pytorch

- 来源：https://github.com/Harry24k/adversarial-attacks-pytorch
- 本地目录：`adversarial-attacks-pytorch-master/`
- 原始 MIT 许可保留在该目录的 `LICENSE` 中。
- 下载包没有附带 Git 提交历史，本仓库不虚构下载时的上游 commit。部分源码含学习注释或已在本地修改的默认参数，实验以显式参数和 summary.json 为准。

## AIGCDetectBenchmark

- 来源：https://github.com/Ekko-zn/AIGCDetectBenchmark
- 本地目录：`AIGCDetectBenchmark-main/AIGCDetectBenchmark-main/`
- 保留上游 README 及源码中的作者信息。本地下载包中未找到根级 LICENSE，本仓库不为这部分代码另行授予许可。
- 为避免上传固定凭据，`eval_all.py` 与 `train.py` 的 Comet API key 改为读取环境变量 `COMET_API_KEY`。需要运行上游 Comet 流程时，由使用者在本机配置；不要把实际密钥提交。

## CNNDetection 示例图片

- 来源：https://github.com/peterwang512/CNNDetection
- 仅保留最初用于推理演示的两张小样例。精确来源 URL、文件大小和原始 SHA-256 记录在 `cnnspot_demo/sample_sources.json` 中。

## AutoAttack 方法依据

- 原作者实现：https://github.com/fra31/auto-attack
- 二分类 DLR 限制及 margin 替代方向：https://github.com/fra31/auto-attack/issues/70
- 本地改造及自定义组合均明确记录相对标准 AutoAttack 的差异。
