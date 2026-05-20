"""Tkinter app for analyzing 12 football matches at once."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .engine import ExactScoreModel
from .schema import MatchInput


MATCH_COUNT = 12

SHARED_DEFAULTS = {
    "league": "默认联赛",
    "league_avg_goals": "2.70",
    "league_home_advantage": "0.12",
    "league_volatility": "0.55",
}

ROW_FIELDS = [
    ("home_team", "主队", 12),
    ("away_team", "客队", 12),
    ("home_attack", "主攻", 6),
    ("away_attack", "客攻", 6),
    ("home_defense", "主防压", 6),
    ("away_defense", "客防压", 6),
    ("home_absence_impact", "主伤停", 6),
    ("away_absence_impact", "客伤停", 6),
    ("home_motivation", "主战意", 6),
    ("away_motivation", "客战意", 6),
]

ROW_DEFAULTS = {
    "home_attack": "1.00",
    "away_attack": "1.00",
    "home_defense": "1.00",
    "away_defense": "1.00",
    "home_absence_impact": "0.00",
    "away_absence_impact": "0.00",
    "home_motivation": "0.00",
    "away_motivation": "0.00",
}

HELP_TEXT = (
    "用法：改主队、客队，其他数字不会填就先用默认值，然后点「分析12场」。\n"
    "进攻：1.00普通，1.20偏强，0.80偏弱。防守压力：1.20更容易丢球，0.80更稳。\n"
    "伤停：0无影响，0.30明显，0.60严重。战意：0普通，0.30偏强，-0.30偏弱。"
)


class ScoreModelApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("足球12场唯一精准比分模型")
        self.geometry("1180x760")
        self.minsize(1080, 680)
        self.shared_entries: dict[str, ttk.Entry] = {}
        self.row_entries: list[dict[str, ttk.Entry]] = []
        self.model = ExactScoreModel(
            mode="adaptive",
            max_total_goals=10,
            max_team_goals=9,
            score_profile="scorebook",
        )
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="足球12场唯一精准比分模型", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(root, text=HELP_TEXT, justify="left").pack(anchor="w", pady=(8, 12))

        shared = ttk.LabelFrame(root, text="公共设置")
        shared.pack(fill="x", pady=(0, 12))
        self._add_shared_field(shared, "league", "联赛", 0)
        self._add_shared_field(shared, "league_avg_goals", "平均总进球", 1)
        self._add_shared_field(shared, "league_home_advantage", "主场优势", 2)
        self._add_shared_field(shared, "league_volatility", "波动", 3)

        table_wrap = ttk.Frame(root)
        table_wrap.pack(fill="x")
        self._build_match_table(table_wrap)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=12)
        ttk.Button(buttons, text="分析12场", command=self._predict_all).pack(side="left")
        ttk.Button(buttons, text="恢复默认", command=self._reset).pack(side="left", padx=8)
        ttk.Button(buttons, text="清空队名", command=self._clear_team_names).pack(side="left")

        self.output = tk.Text(root, height=18, wrap="none", font=("Consolas", 11))
        self.output.pack(fill="both", expand=True)
        self._write_output("填好12场比赛后，点击「分析12场」。不会填的数字先保持默认。")

    def _add_shared_field(self, parent: ttk.Frame, key: str, label: str, col: int) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="w", padx=(10, 4), pady=8)
        entry = ttk.Entry(parent, width=16)
        entry.insert(0, SHARED_DEFAULTS[key])
        entry.grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 10), pady=8)
        self.shared_entries[key] = entry

    def _build_match_table(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="场次", width=5).grid(row=0, column=0, padx=2, pady=3)
        for col, (_, label, width) in enumerate(ROW_FIELDS, start=1):
            ttk.Label(parent, text=label, width=width).grid(row=0, column=col, padx=2, pady=3)

        for row_index in range(MATCH_COUNT):
            ttk.Label(parent, text=str(row_index + 1), width=5).grid(row=row_index + 1, column=0, padx=2, pady=3)
            entries: dict[str, ttk.Entry] = {}
            for col, (key, _, width) in enumerate(ROW_FIELDS, start=1):
                entry = ttk.Entry(parent, width=width)
                entry.insert(0, self._row_default(key, row_index))
                entry.grid(row=row_index + 1, column=col, padx=2, pady=3)
                entries[key] = entry
            self.row_entries.append(entries)

    def _row_default(self, key: str, row_index: int) -> str:
        if key == "home_team":
            return f"主队{row_index + 1}"
        if key == "away_team":
            return f"客队{row_index + 1}"
        return ROW_DEFAULTS[key]

    def _reset(self) -> None:
        for key, entry in self.shared_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, SHARED_DEFAULTS[key])

        for row_index, row in enumerate(self.row_entries):
            for key, entry in row.items():
                entry.delete(0, tk.END)
                entry.insert(0, self._row_default(key, row_index))

    def _clear_team_names(self) -> None:
        for row in self.row_entries:
            row["home_team"].delete(0, tk.END)
            row["away_team"].delete(0, tk.END)

    def _predict_all(self) -> None:
        try:
            matches = self._read_matches()
        except ValueError as exc:
            messagebox.showerror("输入有问题", str(exc))
            return

        picks = self.model.predict_many(matches)
        lines = [
            "12场唯一精准比分结果",
            "",
            "场次 | 比赛 | 唯一比分 | 信号 | xG估计 | 每层冠军",
            "-" * 118,
        ]
        for index, pick in enumerate(picks, start=1):
            layer_summary = " ".join(f"T{item.total_goals}:{item.score}" for item in pick.layer_winners)
            lines.append(
                f"{index:02d} | {pick.home_team} vs {pick.away_team} | "
                f"{pick.score} | {pick.confidence_band} | "
                f"{pick.expected_home_goals:.2f}-{pick.expected_away_goals:.2f} | {layer_summary}"
            )

        lines.extend(["", "最终只看「唯一比分」这一列；后面的 T0-T10 是模型内部每个总球层的冠军，用来复盘。"])
        self._write_output("\n".join(lines))

    def _read_matches(self) -> list[MatchInput]:
        shared = self._read_shared_values()
        matches: list[MatchInput] = []
        for index, row in enumerate(self.row_entries, start=1):
            values = {key: entry.get().strip() for key, entry in row.items()}
            if not values["home_team"] or not values["away_team"]:
                raise ValueError(f"第{index}场：主队和客队都要填写。")

            numeric = self._read_numeric_row(index, values)
            matches.append(
                MatchInput.from_dict(
                    {
                        "match_id": f"gui-match-{index:02d}",
                        "league": shared["league"],
                        "home_team": values["home_team"],
                        "away_team": values["away_team"],
                        "league_avg_goals": shared["league_avg_goals"],
                        "league_home_advantage": shared["league_home_advantage"],
                        "league_volatility": shared["league_volatility"],
                        **numeric,
                        "market": {"confidence": 0.35},
                    }
                )
            )
        return matches

    def _read_shared_values(self) -> dict[str, str | float]:
        league = self.shared_entries["league"].get().strip()
        if not league:
            raise ValueError("公共设置里的联赛不能为空。")

        values: dict[str, str | float] = {"league": league}
        for key in ("league_avg_goals", "league_home_advantage", "league_volatility"):
            try:
                values[key] = float(self.shared_entries[key].get().strip())
            except ValueError as exc:
                raise ValueError(f"公共设置里的「{key}」要填数字。") from exc
        return values

    def _read_numeric_row(self, row_index: int, values: dict[str, str]) -> dict[str, float]:
        numeric: dict[str, float] = {}
        labels = {key: label for key, label, _ in ROW_FIELDS}
        for key in values:
            if key in {"home_team", "away_team"}:
                continue
            try:
                numeric[key] = float(values[key])
            except ValueError as exc:
                raise ValueError(f"第{row_index}场：{labels[key]} 要填数字。") from exc
        return numeric

    def _write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def main() -> None:
    app = ScoreModelApp()
    app.mainloop()


if __name__ == "__main__":
    main()
