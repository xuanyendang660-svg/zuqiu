# 足球盈利价值引擎

这个分支彻底停止把“唯一精准比分”当盈利核心。生产入口只做：

- 赛前亚洲让球单关
- 赛前大小球单关
- 由两条已经独立通过单关门槛的选项组成的 2 串 1
- 没有价格优势时输出 `NO_BET` / `NO_PARLAY`

旧精准比分模型仍保留在历史分支中，但不进入本项目的运行入口。

## 核心原则

1. 先对市场赔率去水，得到市场公平概率。
2. 模型概率必须高于市场概率，并覆盖模型不确定性后仍为正期望。
3. 每一条串关腿必须单独具备下注资格，禁止加入“稳胆”凑赔率。
4. 默认禁止同场串关；没有联合概率模型时，不允许把同场概率直接相乘。
5. 默认影子模式，真实结算样本不足 500 笔时执行仓位为 0。
6. 单注默认上限为资金的 0.5%，2 串 1 默认上限为 0.25%。
7. 主要验证指标是 CLV、ROI、最大回撤和概率校准，不是单纯命中率。

## 输入格式

见 `examples/markets.json`。每个市场必须包含完整的互斥结果，例如两项大小球或两项亚洲让球：

```json
{
  "event_id": "match-001",
  "market_id": "match-001-ou-2.5",
  "market_type": "total",
  "outcomes": [
    {
      "selection": "over 2.5",
      "odds": 2.02,
      "model_probability": 0.54,
      "uncertainty": 0.015,
      "data_quality": 0.95
    },
    {
      "selection": "under 2.5",
      "odds": 1.86,
      "model_probability": 0.46,
      "uncertainty": 0.015,
      "data_quality": 0.95
    }
  ]
}
```

`model_probability` 必须来自独立、经过时间滚动验证的足球模型。大语言模型不能直接填写该字段作为正式下注依据。

## 运行

```bash
python -m pip install -e ".[dev]"
football-value select examples/markets.json --pretty
```

输出包括：

- 市场去水概率
- 模型公平赔率
- 原始 EV 与保守 EV
- 推荐影子仓位与实际执行仓位
- 合格单关
- 合格 2 串 1
- `deployment_allowed`

回测：

```bash
football-value backtest examples/settled.json --pretty
```

## 默认门槛

- 单关模型优势：至少 2.5 个百分点
- 单关原始 EV：至少 3%
- 保守概率下 EV：必须大于 0
- 数据质量：至少 0.80
- 2 串 1 保守 EV：至少 5%
- 单关：0.15 Kelly，最高 0.5% 资金
- 2 串 1：最高 0.25% 资金
- 每日总风险：最高 2% 资金

所有门槛都可以在输入 JSON 的 `policy` 中覆盖，但正式调整必须由历史样本外回测支持。

## 影子模式

默认：

```json
{
  "shadow_mode": true,
  "min_settled_bets": 500
}
```

影子模式仍会输出理论仓位，但实际执行仓位为 0。只有关闭影子模式且结算样本达到门槛时，`deployment_allowed` 才会变成 `true`。

## 不保证盈利

系统只负责把下注纪律、联合定价和风险控制写进代码。没有任何模型能保证盈利。若长期拿不到正 CLV，或者样本外 ROI 没有超过基线，正确输出就是停止下注，而不是降低门槛或增加串关腿数。
