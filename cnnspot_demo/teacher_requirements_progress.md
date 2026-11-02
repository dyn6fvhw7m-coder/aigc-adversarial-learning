---
kind: teacher_requirements_progress
compiled_on: 2026-11-02
requirements_source: user_provided_teacher_screenshots
---

# 老师要求与当前学习进度

本表根据用户提供的老师聊天截图、当前本地代码及已保存实验记录整理。完成状态限定于已核验的模型、样本和参数。

## 一、老师截图明确提到的内容

- 对训练好的模型尝试对抗攻击。
- 使用 AIGCDetectBenchmark：集合多种检测模型，之后的对比实验需要用到。
- 使用 adversarial-attacks-pytorch 中的攻击算法；老师提到通常使用 FGSM、PGD 和 AutoAttack。
- 理解攻击参数；老师给出的 PGD 示例为 eps=8/255、alpha=2/255、steps=4，并提示传入标准化设置。

截图中的项目与文档：

- https://github.com/Ekko-zn/AIGCDetectBenchmark
- https://github.com/Harry24k/adversarial-attacks-pytorch
- https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd

## 二、完成情况

| 内容 | 当前状态 | 已有证据与限制 |
|---|---|---|
| 加载预训练检测器、正常预测 | 已完成 CNNSpot 单模型流程 | ProGAN airplane 固定 400 张正常准确率 100%；网络和权重来自本地基准仓库。尚未完成多个检测器的对比。 |
| 输入输出与标准化适配 | 已完成当前流程 | CNNSpot 单 logit 适配为 [-z/2,z/2] 两类分数；使用 ImageNet 标准化；脚本核对适配前后概率、基线一致性和像素扰动范围。 |
| FGSM | 已运行并评估 | 同批 400 张，eps=8/255，成功 306/400，成功率 76.5%。 |
| PGD | 已运行并评估 | 同批 400 张，老师示例的 eps=8/255、alpha=2/255、4 步，成功 400/400。另完成扰动上限、步数和步长对照。 |
| 随机种子与重复性 | 已进行学习和验证 | 明确 seed 的含义；固定 seed 与确定性设置后，小样本 PGD 重复运行对抗张量一致。该结论限定当时环境与配置。 |
| AutoAttack 单组件 | APGD-CE、FAB-T、Square 已分别运行 | 400 张、eps=8/255 时，分别成功 400、392、398 张；这些是单组件结果。 |
| 多攻击组合机制 | 已验证 | 20 张 FGSM→PGD：第一阶段成功 15 张，第二阶段接手并成功 5 张。 |
| 二分类自定义组合 | 已运行小样本流程 | 20 张 APGD-CE→FAB-T→Square：第一阶段全部成功，后两阶段跳过；不含 APGD-T，不能标作完整标准 AutoAttack。 |
| 完整标准 AutoAttack | 尚未完成 | 本地 standard 含 APGDT；其 targeted DLR 读取第 3、4 个类别分数，当前二分类输出不兼容。当前自定义组合的组件和预算也与 standard 不同。 |
| 多模型、系统性跨来源对比 | 尚未完成 | 当前核心实验只使用 CNNSpot。截图将 benchmark 用于之后对比，未具体规定本轮对比规模。 |

## 三、已完成的参数学习对照

以下均为同一 ProGAN airplane 固定 400 张，eps=1/255、seed=2026：

| 方法 | alpha | steps | 成功数量 | 成功率 |
|---|---|---:|---:|---:|
| PGD | 2/255 | 4 | 394/400 | 98.50% |
| PGD | 2/255 | 10 | 399/400 | 99.75% |
| PGD | 0.25/255 | 10 | 400/400 | 100.00% |
| APGD-CE | 内部自适应 | 10 | 400/400 | 100.00% |

结论：本次设置中增加 PGD 步数、调整步长提高了攻击成功率。一次攻击未成功不能证明扰动范围内不存在对抗样本。不能将这个结果推广为“步长越小永远越好”或“步数越多最终返回结果必然越强”。

详见 [400 张实验完整日志](实验日志/CNNSpot/对抗评估/AIGC-B-261102-001.md)。历史参数以各次 summary.json 为准；当前 attack_pgd.py 已调整到 alpha=0.25/255、steps=10，未来运行时需要核对参数。

## 四、已做的拓展及其边界

跨数据来源的 Stable Diffusion 1.4 实验属于本次学习中追加的拓展，老师截图未明确要求必须做这一轮。

- 固定 400 张：真实图 200 张中正确 182 张，生成图 200 张中正确 9 张，正常准确率 191/400=47.75%。
- PGD 使用 eps=1/255、alpha=0.25/255、steps=10、seed=2026，仅攻击原本正确的 191 张，跳过原本错误的 209 张。
- 新增成功 190/191，成功率约 99.48%；攻击后整份名单仅 1/400 正确，准确率 0.25%。
- 真实图方向：181/182；生成图方向：9/9。生成图方向的攻击成功率分母只有 9，不能将 9/9 描述成攻击了全部 200 张生成图。
- 不同来源的图片内容也可能变化，这些差异不能只归因于生成算法。

证据：[正常预测](baseline_sd14_200perclass.csv)、[攻击汇总](sd14_pgd_eps1_alpha0p25_steps10_400samples_seed2026/summary.json)。

## 五、当前主线剩余问题：标准 AutoAttack 与二分类

本地 [autoattack.py](../adversarial-attacks-pytorch-master/torchattacks/attacks/autoattack.py) 的 standard 顺序为 APGD-CE、APGDT、FAB-T、Square。

本地 [apgdt.py](../adversarial-attacks-pytorch-master/torchattacks/attacks/apgdt.py) 的 dlr_loss_targeted 使用 x_sorted[:, -3] 和 x_sorted[:, -4]，损失计算需要至少四个类别分数。当前适配器只有两个分数，设置 n_classes=2 不会重写这个损失公式。

### 原作者与官方实现核查（2026-11-02）

- 原作者 fra31 在 [Issue #70](https://github.com/fra31/auto-attack/issues/70) 于 2021-09-09 回复：少于四类时不能按原版方式运行；提出将损失退回 margin loss 作为可能替代，同时指出它不具有尺度不变性。这是可能的改造方向，不能称为已经验证并发布的二分类标准协议。
- 本次查看的官方 [checks.py](https://github.com/fra31/auto-attack/blob/master/autoattack/checks.py) 中，check_n_classes 会对二分类使用 DLR、三分类使用 targeted DLR 发出警告，与本地读到的限制一致。
- 官方 [README 的 Custom version](https://github.com/fra31/auto-attack#custom-version) 支持 version='custom' 并指定攻击组件。这证明官方提供自定义机制，不代表任意省略或替换组件后的方案与 standard 等价，也不证明我们使用的三组件组合已经成为官方二分类标准。
- API 需要区分：上述 version='custom' 属于原作者的 autoattack 包；当前老师指定的本地 torchattacks.AutoAttack 只接受 standard、plus、rand。我们现有自定义实验使用 torchattacks.MultiAttack，不能直接照抄官方包的 custom 参数。

当前可准确汇报为：已完成原库组件实验和二分类自定义组合；完整原版 standard 与当前二分类模型不兼容。可继续研究作者提出的 margin 替代损失，但需单独命名、验证和报告修改。若老师要求具体评测协议，应明确二分类场景采用的组件、损失、预算和评测口径。

两项容易误报的情况：

- 第一阶段已经全部成功会使后续组件跳过，整个调用正常结束不等于 APGDT 被执行且兼容。
- 省略或替换 APGDT、改变预算后，应准确记录为相应的自定义方案。

## 六、可以向老师汇报的进度摘要

已使用基准仓库的 CNNSpot 权重完成正常预测、FGSM 和 PGD 攻击，并进行了固定图片名单、随机种子、扰动范围及参数对照检查。已独立运行 APGD-CE、FAB-T、Square，并验证 MultiAttack 的阶段接力和提前停止。当前待解决的是标准 AutoAttack 中 APGDT 的 targeted DLR 与二分类输出的兼容问题；目前跑通的三组件组合已标注为二分类自定义组合。现有结果针对一个检测器和有限子集，尚未完成基准中的多模型系统对比。

## 原始结果入口

- [FGSM 400 张](fgsm_eps8_400samples/summary.json)
- [PGD 示例参数 400 张](pgd_eps8_steps4_400samples_seed2026/summary.json)
- [APGD-CE 400 张](apgd_ce_eps8_steps10_400samples_seed2026/summary.json)
- [FAB-T 400 张](fab_t_eps8_steps10_400samples_seed2026/summary.json)
- [Square 400 张](square_eps8_queries1000_400samples_seed2026/summary.json)
- [FGSM→PGD 小样本组合](multi_fgsm_pgd_eps8_seed2026/summary.json)
- [三组件二分类自定义组合](ensemble_apgd_fab_square_eps8_seed2026/summary.json)
