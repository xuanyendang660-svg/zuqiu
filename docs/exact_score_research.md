# 精准比分研究入口（独立于现有投注引擎）

## 目标与边界

每场输出一个比分：从训练后的完整联合比分分布中取最大概率格，不对热门、冷门、常见比分或尾部人为加减分。先保留历史时序与数据出处，开赛前冻结，赛后逐场核对方向、净胜差、败方进球/平局层级、总球带、停表状态。事件树和 31 格账本用于解释及诊断，不参与比分取舍。

该入口是**研究候选**。当前主分支的 `football-value` 盈利研究入口仍是单独的系统；这里不自动下注或推断收益。旧 `score_model` 分支的人工“防舒服比分”修正没有迁入。

## 数据合同

`history.csv` 需要 `match_id,date,league_code,home_team,away_team,actual_home_goals,actual_away_goals`。比赛日期格式 `YYYY-MM-DD`，赛果是 90 分钟常规时间主客进球。重复 ID、缺少队名或比分会阻断。程序按**整个日历日**分组，从此前比赛重建联赛均值、两队近八场进失球、Elo；一整天的比赛都完成赛前特征构建后才更新状态，不从同日任何赛果取未来信息。跨时区的比赛统一日历口径需要由数据供应方提供准确日期；当只有日期而缺少开球时刻时不能做更细的日内训练。

已存在于 `codex/football-complete-v3-training-pr` 分支的 `data/historical_football_dataset.csv` 可作为历史样本输入（14,305 场，2019-08-02 至 2025-05-25），只使用上面七个身份/结果字段，其余预计算特征会被重建。它源于 Football-Data.co.uk，但原文件没有盘口报价的实际发布和抓取时间，因此其中的市场赔率**不能**进入严格赛前回测。当前日期之后的比赛和用户此前已冻结的预测未包含在该数据里。初场没有前史时使用固定联赛伪计数（20 场，主 1.40、客 1.25），此先验需要未来时段检验，不宣称已普适校准。

如本地只有 `main`，可执行 `git fetch origin codex/football-complete-v3-training-pr`，再执行 `git show FETCH_HEAD:data/historical_football_dataset.csv > history.csv`，把旧数据文件只作为导入来源；不运行那个分支的人为比分选择器。

可选 `--quotes` 文件是 JSON 数组，每个快照至少包含 `match_id,kickoff_utc,published_at,observed_at,source,one_x_two_odds,over_under_odds,total_line`。所有时间都带时区；发布、抓取均须早于开球，来源和比赛 ID 必须一致。`one_x_two_odds` 包括 `home,draw,away`；`over_under_odds` 包括 `over,under`。可另附 `asian_handicap,asian_home_odds,asian_away_odds` 供结算校验。无有效时间戳的市场分支退出，记录 `MARKET_MISSING` 或 `MARKET_BLOCKED`，足球分布单独工作。不同公司/时刻不得在未核对时混成一个快照。

## 训练与运行

```sh
python -m pip install -e '.[score]'
football-score train history.csv --holdout-start 2024-08-01 \
  --model-out model_v1.json --report-out backtest_v1.json
# 可用经核验的报价：在 train 命令加 --quotes quotes.json
football-score freeze upcoming.json --model model_v1.json --out frozen_m1.json
football-score review freezes_week.json results_week.json --out review_week.json
```

`train` 按日期切成训练和时间外保留段，用训练段估计 (T=H+A\sim\mathrm{NB2}(\mu,\kappa)\)、(H\mid T\sim\mathrm{BetaBinomial}(T,p\phi,(1-p)\phi)\)。总球支持集一直扩到剩余概率 < `1e-8`。比较先前比赛加时间衰减的 Dixon–Coles 基线。市场基线把已同步的 1X2/大小球去水后拟合独立 Poisson 比分分布；亚洲让球按半球和四分之一球的输赢/走盘份额校验。只有时间先后无争议且样本数达门槛，才以训练段的日期分组时间外折数选择混合权重；否则权重为 1（只用足球端）。使用日期分组 bootstrap 估计相对基线的不确定性。保留段不参与任何系数或混合权重选择。

`freeze` 输入一场赛前 JSON：`match_id,home_team,away_team,kickoff_utc,feature_source,features_published_at,features_observed_at` 和 `league_avg_goals_pre,league_home_goals_pre,league_away_goals_pre,home_recent8_gf,home_recent8_ga,away_recent8_gf,away_recent8_ga,home_elo_pre,away_elo_pre`。这些特征必须是预先计算、具有来源和早于开球的发布时间、抓取时间的赛前值，不接受事后修订；可选 `market` 同报价合同。输出保存唯一主比分、概率、整张稀疏联合分布和时间戳；文件存在则拒绝覆盖。`stop_state` 默认未知，不能由终场比分或未来事件推断。将多个冻结合并为 `{"predictions":[...]}`，结果用 `{"results":[{"match_id":"...","home_team":"...","away_team":"...","score":[1,0],"status":"FT_90","stop_state":"lead_protected"},...]}`。没有可核实比赛时间线时，不填结果的 `stop_state`。

`review` 逐场判读五轴；净胜差和败方进球仅在方向一致时评分，汇总命中率的分母是方向正确的场次，停表状态缺少赛前声明或实际时间线时保持未知。若要声明赛前停表状态，输入还须有 `stop_state,stop_state_evidence,stop_state_published_at`，这不会改变模型比分。某轴至少三场错误、跨至少两日才提出**待验证**假设，每周最多三项；即使重复也不会据此自动修改参数。应结合赛前资料版本和可预测/偶发事件验证具体模块，对下周冻结原版与候选版成对比较。少于门槛仅列事实差异。

## 当前回测的决策

同一历史数据 14,305 场，以 2019-08-02 至 2024-06-02 的 11,941 场训练，2024-08-09 至 2025-05-25 的 2,364 场作为时间外保留期：联合模型精准比分命中 `12.394%`，Dixon–Coles `12.606%`；平均对数损失 `2.91090` 对 `2.90670`（越低越好）。按比赛日分组的 400 次配对 bootstrap 显示新模型减去基线的精准比分改善区间 `[-0.963, +0.530]` 个百分点；对数损失改善区间 `[-0.00739, -0.00066]`。当前新模型在该期没有证明优势，仍是研究候选，不是精度升级。

回测还报告方向、条件净胜差、条件败方进球、总球带、Brier、联赛切片与预测/实际各带校准。没有任何有效时间戳的盘口参与该次实验，融合权重为 `1`；不能声称完成双基线融合验证。赛前首发、事件时间线及 2025-26/2026-27 数据也未进入这次回测。手工“修大球”“修冷门”没有加入优化器。
