# 盈利玩法 v1

本分支停用“唯一精准比分”作为默认入口，改为：

- 单关只选择保守概率下仍为正 EV 的亚洲让球或大小球。
- 同一场最多保留一个方向。
- 串关初期只允许不同比赛的 2 串 1。
- 每一腿必须先独立通过单关筛选。
- 不满足条件时输出 `NO_BET` 或 `NO_PARLAY`。
- 单关和串关的 ROI、CLV、回撤分开统计。

## 运行筛选

```powershell
$env:PYTHONPATH="src"
python -m profit_model.cli examples\profit_offers_sample.json --pretty
```

也可以双击 `打开模型.bat`，再输入候选盘口 JSON 路径。

输入字段：

- `decimal_odds`：实际可下注欧赔。
- `model_probability`：模型点估计概率。
- `model_probability_lower`：保守概率下限；系统优先使用它。
- `market_fair_probability`：去水后的市场公平概率，可选。
- `quoted_parlays`：庄家实际 2 串 1 报价，可选；不提供时按两腿欧赔乘积计算。

默认风控：

- 单关保守 EV 至少 3%。
- 单关概率优势至少 2.5 个百分点。
- 单注最高资金 0.5%。
- 2 串 1 保守 EV 至少 7%。
- 串关最高资金 0.25%。
- 同场串关默认禁止。

这些阈值只是初始研究参数，必须由封存的前向样本校准，不能视为盈利证明。

## 结算与回测

```powershell
$env:PYTHONPATH="src"
python -m profit_model.backtest_cli records.json
```

记录格式：

```json
{
  "records": [
    {
      "bet_id": "2026-001",
      "kind": "single",
      "stake": 1.0,
      "placed_odds": 2.02,
      "closing_odds": 1.94,
      "result": "win"
    }
  ]
}
```

报告分别输出：

- 单关与串关 ROI
- 命中率
- 平均 CLV
- 最大回撤
- 最长连败

## 测试

```powershell
$env:PYTHONPATH="src"
python -m unittest tests.test_profit_model tests.test_profit_backtest
```

本系统不保证盈利。只有长期封存样本显示正 CLV，并且样本外 ROI 超过基线，才允许从影子模式进入真钱验证。
