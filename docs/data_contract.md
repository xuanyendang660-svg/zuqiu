# 输入字段口径

模型输入是一场比赛的 JSON。字段应尽量保持赛前可获得，不能混入赛后信息。

## 基础字段

| 字段 | 含义 |
| --- | --- |
| `match_id` | 比赛唯一 ID |
| `league` | 联赛名称或联赛 ID |
| `home_team` | 主队 |
| `away_team` | 客队 |

## 联赛画像

| 字段 | 建议范围 | 含义 |
| --- | --- | --- |
| `league_avg_goals` | `1.55..3.75` | 联赛平均总进球 |
| `league_home_advantage` | `-0.20..0.35` | 主场预期进球增量 |
| `league_draw_bias` | `-1..1` | 联赛平局倾向 |
| `league_volatility` | `0..1` | 大比分和节奏波动倾向 |

## 球队强度

| 字段 | 中性值 | 含义 |
| --- | --- | --- |
| `home_attack` | `1.0` | 主队进攻强度 |
| `away_attack` | `1.0` | 客队进攻强度 |
| `home_defense` | `1.0` | 主队防守失球压力，越高代表越容易被打穿 |
| `away_defense` | `1.0` | 客队防守失球压力，越高代表越容易被打穿 |

## 近期 xG

| 字段 | 含义 |
| --- | --- |
| `recent_home_xg` | 主队近期场均 xG |
| `recent_away_xg` | 客队近期场均 xG |
| `recent_home_xga` | 主队近期场均 xGA |
| `recent_away_xga` | 客队近期场均 xGA |

如果没有 xG，可以先用射门质量、禁区触球或进球期望替代；不要直接用赛果比分硬替代，比分噪声更大。

## 临场修正

| 字段 | 建议范围 | 含义 |
| --- | --- | --- |
| `home_form` / `away_form` | `-1..1` | 近期状态 |
| `home_absence_impact` / `away_absence_impact` | `0..1` | 关键伤停影响 |
| `home_rotation_risk` / `away_rotation_risk` | `0..1` | 轮换风险 |
| `home_rest_edge` / `away_rest_edge` | `-1..1` | 休息和赛程优势 |
| `home_motivation` / `away_motivation` | `-1..1` | 战意 |
| `weather_goal_drag` | `0..1` | 天气对进球的压制 |
| `referee_goal_bias` | `-1..1` | 裁判尺度对进球的影响 |

## 积分与战意字段

这些字段用于避免只看排名、不算真实战意。最后一轮、争冠、升级、保级、附加赛必须优先填。

| 字段 | 建议范围 | 含义 |
| --- | --- | --- |
| `home_rank` / `away_rank` | 整数 | 当前排名，只做展示和辅助校验 |
| `home_points` / `away_points` | 整数 | 当前积分 |
| `home_goal_difference` / `away_goal_difference` | 整数 | 当前净胜球 |
| `home_table_pressure` / `away_table_pressure` | `-1..1` | 积分形势压力，必须赢/保级/争冠填高，已无欲无求填低或负值 |
| `home_survival_pressure` / `away_survival_pressure` | `0..1` | 保级、附加赛、必须抢命的主动进攻压力，不等于普通战意 |
| `home_big_win_need` / `away_big_win_need` | `0..1` | 是否需要大胜刷净胜球 |
| `home_draw_sufficient` / `away_draw_sufficient` | `true/false` | 平局是否基本够用 |
| `home_settled` / `away_settled` | `true/false` | 是否已锁定名次/冠军/升级/降级，可能轮换或战意下降 |
| `is_final_round` | `true/false` | 是否末轮/收官轮/争冠组最后阶段，需要放大极端分支 |
| `endgame_chaos` | `0..1` | 收官战混乱度，包含庆典、抢命、无欲无求、盘口异常同场存在 |
| `home_celebration_risk` / `away_celebration_risk` | `0..1` | 已夺冠/已锁目标后的庆典、开放节奏、注意力下降风险 |
| `home_collapse_risk` / `away_collapse_risk` | `0..1` | 末轮心理、防线、战意或换人导致被连续打穿的风险 |

推荐口径：

- 必须赢争直升、争冠、保级：`table_pressure=0.70..1.00`
- 保级/附加赛抢命且必须主动赢球：`survival_pressure=0.65..1.00`
- 平局够用：`draw_sufficient=true`，并把 `table_pressure` 降到 `0.15..0.45`
- 已锁冠军/已升级/已降级/无排名收益：`settled=true`，`table_pressure=-0.30..0.10`
- 需要净胜球反超：`big_win_need=0.45..1.00`
- 末轮强队庆典但对手防线崩：`is_final_round=true`，强队 `celebration_risk=0.45..0.85`，弱队 `collapse_risk=0.45..0.90`
- 末轮名气队无目标且盘面偏热：`is_final_round=true`，名气队 `settled=true`、`collapse_risk=0.45..0.90`，同时提高 `bookmaker_trap_risk`
- 末轮双方都有开放动机：`endgame_chaos=0.55..0.90`，大比分平局和互爆比分必须入池
- 末轮弱队保级抢命，对手无目标/防线崩：弱队 `survival_pressure=0.80..1.00`，对手 `settled=true` 或 `collapse_risk=0.45..0.90`，允许 `3-0`、`4-1`、`5-1`、`6-1`

## 市场字段

```json
{
  "market": {
    "confidence": 0.62,
    "expected_home_goals": 1.38,
    "expected_away_goals": 1.31,
    "asian_handicap": -1.25,
    "total_goals_line": 2.75,
    "home_money_heat": 0.68,
    "draw_money_heat": 0.22,
    "away_money_heat": 0.10,
    "favorite_money_heat": 0.72,
    "over_money_heat": 0.55,
    "under_money_heat": 0.18,
    "bookmaker_trap_risk": 0.20,
    "one_x_two_odds": {
      "home": 2.45,
      "draw": 3.35,
      "away": 2.75
    },
    "exact_score_odds": {
      "1-1": 6.8,
      "1-2": 10.5,
      "1-3": 28.0
    }
  }
}
```

`confidence` 代表市场可信度。热门联赛、主流公司、临场深盘可以更高；冷门联赛、数据源质量差、盘口波动异常时应该更低。

`exact_score_odds` 可以不完整。模型会用已有比分赔率和市场预期进球混合推断市场分布。

盘口和资金字段口径：

- `asian_handicap` 是主队视角。主队让一球填 `-1.0`，主队受让一球填 `1.0`。
- `total_goals_line` 填大小球主流盘口，例如 `2.5`、`2.75`、`3.0`。
- `home_money_heat/draw_money_heat/away_money_heat` 不是胜率，是资金或大众热度，范围 `0..1`。
- `favorite_money_heat` 用于深盘强队热度。如果只能拿到一个热度数字，填这里。
- `bookmaker_trap_risk` 代表盘口与基本面不匹配、强队热但赔率不配合、平赔异常防范等风险。
