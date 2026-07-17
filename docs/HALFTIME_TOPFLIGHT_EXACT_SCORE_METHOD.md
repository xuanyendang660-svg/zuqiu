# 顶级联赛半场精准比分方法

## 最终状态

**通过生产闸门：validated_topflight_selective_halftime_exact_score_method**

这不是通用赛前精准比分模型，也不适用于低级别联赛、杯赛或半场非 0-0 的比赛。它是一个高选择性的半场状态方法：只有满足完整适用条件时输出一个终场精准比分，否则必须放弃预测。

## 唯一允许输出的比分

当比赛属于国内顶级联赛，半场比分为 0-0，并且赛前存在明显强队时：

- 强队主场：预测终场 **1-0**
- 强队客场：预测终场 **0-1**

换成“弱队-强队”方向，统一预测为 **0-1**。

## 强队触发条件

先把赛前 1X2 欧赔去水，得到主胜、平局、客胜公平概率。主客胜概率中较低的一方定义为弱队。

满足以下任意一项：

1. 弱队胜率不高于 **20%**；
2. 强队胜率减弱队胜率超过 **30 个百分点**。

同时必须满足：

- 国内顶级联赛；
- 半场精确比分 0-0；
- 赛前赔率完整且可去水；
- 主客胜率不能相同。

任意条件不满足：**不输出比分**。

## 验证链路

### 规则发现与筛选

- 五大联赛发现数据：42,430 场
- 二级联赛迁移筛选：8,154 场
- 核心规则只根据迁移集指标冻结，未使用后续盲测结果挑选

### 现代顶级联赛复制集

荷甲、葡超、比甲、苏超、土超、希腊超，2022-2025：

- 总比赛：6,885
- 触发：727
- 覆盖率：10.56%
- 精准比分命中：26.13%
- 同半场比分基线：19.53%
- 提升：**+6.60 个百分点**
- Bootstrap 95% 区间：**+1.65 至 +11.42 个百分点**
- McNemar 精确检验：**p=0.00979**
- 6 个联赛中 5 个联赛为正提升

### 独立历史顶级联赛盲测

相同六个联赛，2002-2017；该数据未用于发现、筛选或阈值设计：

- 总比赛：23,711
- 触发：2,929
- 覆盖率：12.35%
- 精准比分命中：25.64%
- 同半场比分基线：21.00%
- 提升：**+4.64 个百分点**
- Bootstrap 95% 区间：**+2.15 至 +7.10 个百分点**
- McNemar 精确检验：**p=0.000257**
- 6 个联赛全部为正提升

## 已明确否决的版本

- 通用赛前精准比分模型：未稳定击败市场。
- 9 条半场状态规则组合：影子集 p=0.0706，未通过显著性闸门。
- 低级别联赛版本：仅提升 0.99 个百分点，p=0.638，明确失败。
- 弱队 30 分钟 2-0 后继续打穿：Wyscout 样本有强信号，但独立数据触发过少，未通过跨源确认。

## 调用方式

Python：

```python
from football_v2.halftime_topflight_predictor import (
    predict_topflight_halftime_exact_score_from_odds,
)

result = predict_topflight_halftime_exact_score_from_odds(
    is_top_flight_domestic_league=True,
    home_odds=1.35,
    draw_odds=5.00,
    away_odds=10.00,
    halftime_home_score=0,
    halftime_away_score=0,
)

if result.accepted:
    print(result.home_score, result.away_score)
else:
    print("ABSTAIN", result.reason)
```

核心实现：

- `src/football_v2/halftime_topflight_predictor.py`
- `src/football_v2/halftime_core_predictor.py`
- `src/football_v2/halftime_topflight_audit.py`
- `scripts/run_halftime_topflight_domain_audit.py`

## 风险边界

- 25%-26% 的精准比分命中率不等于投注盈利。
- 验证基线是同半场比分状态的历史模态比分，不是实时正确比分市场赔率。
- 未验证杯赛、国家队比赛、女子赛事、青年赛事和低级别联赛。
- 不得把“半场 0-0 + 一般热门”扩张成该方法；强弱阈值必须满足。
- 不得在未触发时强行输出模板比分。
