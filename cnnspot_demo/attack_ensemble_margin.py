"""四组件二分类自定义组合入口；含 APGD-T-Margin，非标准 AutoAttack。

输入：与基线 CSV 一致的固定图片名单、CNNSpot 权重、像素扰动上限。
输出：逐图记录、每阶段统计、汇总 JSON、浮点对抗张量和量化预览。
复用三组件入口的评估流程，保留其原有默认组件与历史实验配置。
"""

from attack_ensemble import main


if __name__ == "__main__":
    main(include_margin=True)
