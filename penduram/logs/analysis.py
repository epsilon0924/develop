from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "target_steps": 500,
    "max_force": 10.0,
    "latest_window": 100,
    "weights": {
        "success_rate": 0.45,
        "episode_duration": 0.30,
        "duration_consistency": 0.15,
        "force_efficiency": 0.10
    }
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.write_text(
            json.dumps(
                DEFAULT_CONFIG,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return DEFAULT_CONFIG.copy()

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()


def result_series(
    df: pd.DataFrame,
) -> pd.Series:
    if "result" not in df.columns:
        return pd.Series("", index=df.index)

    return (
        df["result"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def format_number(
    value: float,
    digits: int = 3,
) -> str:
    if value is None or math.isnan(value):
        return "N/A"

    return f"{value:,.{digits}f}"


def format_percent(
    value: float,
) -> str:
    if value is None or math.isnan(value):
        return "N/A"

    return f"{value * 100:.3f}%"


def safe_mean(
    series: pd.Series,
) -> float:
    return (
        float(series.mean())
        if not series.empty
        else float("nan")
    )


def safe_std(
    series: pd.Series,
) -> float:
    if series.empty:
        return float("nan")

    if len(series) == 1:
        return 0.0

    return float(series.std(ddof=1))


def safe_min(
    series: pd.Series,
) -> float:
    return (
        float(series.min())
        if not series.empty
        else float("nan")
    )


def safe_max(
    series: pd.Series,
) -> float:
    return (
        float(series.max())
        if not series.empty
        else float("nan")
    )


def safe_median(
    series: pd.Series,
) -> float:
    return (
        float(series.median())
        if not series.empty
        else float("nan")
    )


def safe_quantile(
    series: pd.Series,
    quantile: float,
) -> float:
    return (
        float(series.quantile(quantile))
        if not series.empty
        else float("nan")
    )


def markdown_table(
    headers: list[str],
    rows: list[list[Any]],
) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for row in rows:
        output.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|")
                for value in row
            )
            + " |"
        )

    return "\n".join(output)


def calculate_components(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    target_steps = float(
        config["target_steps"]
    )

    max_force = float(
        config["max_force"]
    )

    results = result_series(df)
    steps = numeric_series(df, "steps")

    mean_absolute_force = numeric_series(
        df,
        "mean_absolute_force",
    )

    if len(df) == 0:
        success_rate = 0.0
    else:
        success_rate = float(
            results.eq("success").mean()
        )

    if steps.empty or target_steps == 0:
        duration_score = 0.0
        consistency_score = 0.0
    else:
        normalized_steps = (
            steps / target_steps
        ).clip(
            lower=0.0,
            upper=1.0,
        )

        duration_score = float(
            normalized_steps.mean()
        )

        step_error = (
            target_steps
            - steps.clip(
                upper=target_steps
            )
        )

        root_mean_squared_step_error = float(
            np.sqrt(
                np.mean(
                    np.square(step_error)
                )
            )
        )

        consistency_score = clamp(
            1.0
            - root_mean_squared_step_error
            / target_steps
        )

    if (
        mean_absolute_force.empty
        or max_force == 0
    ):
        force_efficiency = 0.0
    else:
        normalized_force = (
            mean_absolute_force.abs()
            / max_force
        ).clip(
            lower=0.0,
            upper=1.0,
        )

        force_efficiency = clamp(
            1.0
            - float(
                normalized_force.mean()
            )
        )

    return {
        "success_rate": success_rate,
        "episode_duration": duration_score,
        "duration_consistency": (
            consistency_score
        ),
        "force_efficiency": (
            force_efficiency
        ),
    }


def calculate_score(
    components: dict[str, float],
    weights: dict[str, float],
) -> dict[str, Any]:
    weight_sum = sum(
        float(value)
        for value in weights.values()
    )

    contributions: dict[str, float] = {}

    if weight_sum == 0:
        normalized_weights = {
            key: 0.0
            for key in weights
        }
    else:
        normalized_weights = {
            key: float(value) / weight_sum
            for key, value in weights.items()
        }

    for name, component in components.items():
        weight = normalized_weights.get(
            name,
            0.0,
        )

        contributions[name] = (
            100.0
            * weight
            * component
        )

    total = sum(
        contributions.values()
    )

    return {
        "total": total,
        "normalized_weights": (
            normalized_weights
        ),
        "contributions": contributions,
    }


def calculate_statistics(
    df: pd.DataFrame,
) -> list[list[str]]:
    columns = [
        "steps",
        "total_reward",
        "initial_cart_position",
        "initial_cart_velocity",
        "initial_angle_degrees",
        "initial_angular_velocity",
        "final_cart_position",
        "final_cart_velocity",
        "final_angle_degrees",
        "final_angular_velocity",
        "mean_force",
        "mean_absolute_force",
        "maximum_absolute_force",
        "environment_steps_per_second",
    ]

    rows: list[list[str]] = []

    for column in columns:
        values = numeric_series(
            df,
            column,
        )

        if values.empty:
            continue

        rows.append(
            [
                column,
                f"{len(values):,}",
                format_number(
                    safe_mean(values)
                ),
                format_number(
                    safe_std(values)
                ),
                format_number(
                    safe_min(values)
                ),
                format_number(
                    safe_quantile(
                        values,
                        0.25,
                    )
                ),
                format_number(
                    safe_median(values)
                ),
                format_number(
                    safe_quantile(
                        values,
                        0.75,
                    )
                ),
                format_number(
                    safe_max(values)
                ),
            ]
        )

    return rows


def result_distribution(
    df: pd.DataFrame,
) -> list[list[str]]:
    results = result_series(df)

    if results.empty:
        return []

    counts = results.value_counts()

    rows: list[list[str]] = []

    for result, count in counts.items():
        rate = (
            float(count) / len(results)
            if len(results) > 0
            else 0.0
        )

        rows.append(
            [
                result,
                f"{count:,}",
                format_percent(rate),
            ]
        )

    return rows


def compare_windows(
    first: pd.DataFrame,
    latest: pd.DataFrame,
) -> list[list[str]]:
    rows: list[list[str]] = []

    metrics = [
        "steps",
        "total_reward",
        "mean_absolute_force",
        "maximum_absolute_force",
        "environment_steps_per_second",
    ]

    for metric in metrics:
        first_values = numeric_series(
            first,
            metric,
        )

        latest_values = numeric_series(
            latest,
            metric,
        )

        first_mean = safe_mean(
            first_values
        )

        latest_mean = safe_mean(
            latest_values
        )

        if (
            math.isnan(first_mean)
            or math.isnan(latest_mean)
            or first_mean == 0
        ):
            relative_change = float("nan")
        else:
            relative_change = (
                latest_mean - first_mean
            ) / abs(first_mean)

        rows.append(
            [
                metric,
                format_number(first_mean),
                format_number(latest_mean),
                format_percent(
                    relative_change
                ),
            ]
        )

    first_success = float(
        result_series(first)
        .eq("success")
        .mean()
    )

    latest_success = float(
        result_series(latest)
        .eq("success")
        .mean()
    )

    rows.append(
        [
            "success_rate",
            format_percent(
                first_success
            ),
            format_percent(
                latest_success
            ),
            format_percent(
                latest_success
                - first_success
            ),
        ]
    )

    return rows


def correlation_table(
    df: pd.DataFrame,
) -> list[list[str]]:
    columns = [
        "steps",
        "total_reward",
        "initial_angle_degrees",
        "initial_cart_velocity",
        "initial_angular_velocity",
        "mean_force",
        "mean_absolute_force",
        "maximum_absolute_force",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if len(available) < 2:
        return []

    numeric = df[
        available
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlations = numeric.corr()

    rows: list[list[str]] = []

    for row_name in available:
        row = [row_name]

        for column_name in available:
            value = correlations.loc[
                row_name,
                column_name,
            ]

            row.append(
                format_number(
                    float(value)
                )
            )

        rows.append(row)

    return rows


def generate_formula_markdown(
    weights: dict[str, float],
) -> list[str]:
    weight_sum = sum(
        float(value)
        for value in weights.values()
    )

    if weight_sum == 0:
        normalized = {
            key: 0.0
            for key in weights
        }
    else:
        normalized = {
            key: float(value) / weight_sum
            for key, value in weights.items()
        }

    return [
        "## Evaluation Function",
        "",
        "The total evaluation value is defined as:",
        "",
        "$$",
        "E = 100 \\left("
        f"{normalized.get('success_rate', 0.0):.6f}S"
        " + "
        f"{normalized.get('episode_duration', 0.0):.6f}D"
        " + "
        f"{normalized.get('duration_consistency', 0.0):.6f}C"
        " + "
        f"{normalized.get('force_efficiency', 0.0):.6f}F"
        "\\right)",
        "$$",
        "",
        "where:",
        "",
        "$$",
        "S = \\frac{N_{success}}{N}",
        "$$",
        "",
        "$$",
        "D = \\frac{1}{N}"
        "\\sum_{i=1}^{N}"
        "\\min\\left("
        "\\frac{t_i}{T}, 1"
        "\\right)",
        "$$",
        "",
        "$$",
        "RMSE_t = "
        "\\sqrt{"
        "\\frac{1}{N}"
        "\\sum_{i=1}^{N}"
        "\\left("
        "T - \\min(t_i,T)"
        "\\right)^2"
        "}",
        "$$",
        "",
        "$$",
        "C = \\max\\left("
        "0,"
        "1 - \\frac{RMSE_t}{T}"
        "\\right)",
        "$$",
        "",
        "$$",
        "F = \\max\\left("
        "0,"
        "1 - "
        "\\frac{1}{N}"
        "\\sum_{i=1}^{N}"
        "\\min\\left("
        "\\frac{|\\bar{f}_i|}{f_{max}}, 1"
        "\\right)"
        "\\right)",
        "$$",
        "",
        "Definitions:",
        "",
        "- $E$: total evaluation value from 0 to 100",
        "- $N$: number of analyzed episodes",
        "- $N_{success}$: number of successful episodes",
        "- $t_i$: number of steps in episode $i$",
        "- $T$: target number of steps",
        "- $\\bar{f}_i$: mean absolute force in episode $i$",
        "- $f_{max}$: configured maximum force",
        "- $S$: success component",
        "- $D$: episode-duration component",
        "- $C$: duration-consistency component",
        "- $F$: force-efficiency component",
        "",
        "The score is a project-specific measurement. "
        "It is not a standardized reinforcement-learning benchmark.",
        "",
    ]


def analyze(
    csv_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    config = load_config(config_path)

    df = pd.read_csv(csv_path)

    if df.empty:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            "# Training Analysis\n\n"
            "No rows were found in the CSV file.\n",
            encoding="utf-8",
        )
        return

    if "episode" in df.columns:
        df["episode"] = pd.to_numeric(
            df["episode"],
            errors="coerce",
        )

        df = (
            df.dropna(subset=["episode"])
            .sort_values("episode", kind="stable")
            .reset_index(drop=True)
        )

    if df.empty:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            "# Training Analysis\n\n"
            "No valid episode records were found.\n",
            encoding="utf-8",
        )
        return

    window_size = max(
        1,
        int(config["latest_window"]),
    )

    actual_window = min(
        window_size,
        len(df),
    )

    first_window = df.head(actual_window)
    latest_window = df.tail(actual_window)

    overall_components = calculate_components(
        df,
        config,
    )

    latest_components = calculate_components(
        latest_window,
        config,
    )

    weights = config["weights"]

    overall_score = calculate_score(
        overall_components,
        weights,
    )

    latest_score = calculate_score(
        latest_components,
        weights,
    )

    first_episode = (
        int(df["episode"].min())
        if "episode" in df.columns
        else 1
    )

    final_episode = (
        int(df["episode"].max())
        if "episode" in df.columns
        else len(df)
    )

    lines = [
        "# Training Analysis",
        "",
        "## Dataset",
        "",
        f"- Source: `{csv_path}`",
        f"- Configuration: `{config_path}`",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{len(df.columns):,}**",
        f"- Episode range: **{first_episode:,} to {final_episode:,}**",
        f"- Latest-window size: **{actual_window:,}**",
        "",
        "## Evaluation Summary",
        "",
        markdown_table(
            ["Metric", "Overall", "Latest Window"],
            [
                [
                    "Total score",
                    format_number(
                        overall_score["total"],
                    ),
                    format_number(
                        latest_score["total"],
                    ),
                ],
                [
                    "Success rate",
                    format_percent(
                        overall_components["success_rate"],
                    ),
                    format_percent(
                        latest_components["success_rate"],
                    ),
                ],
                [
                    "Episode duration",
                    format_percent(
                        overall_components["episode_duration"],
                    ),
                    format_percent(
                        latest_components["episode_duration"],
                    ),
                ],
                [
                    "Duration consistency",
                    format_percent(
                        overall_components["duration_consistency"],
                    ),
                    format_percent(
                        latest_components["duration_consistency"],
                    ),
                ],
                [
                    "Force efficiency",
                    format_percent(
                        overall_components["force_efficiency"],
                    ),
                    format_percent(
                        latest_components["force_efficiency"],
                    ),
                ],
            ],
        ),
        "",
        "Scores range from 0 to 100. Higher values indicate better "
        "performance according to the configured evaluation function.",
        "",
        "## Score Contributions",
        "",
        markdown_table(
            [
                "Component",
                "Weight",
                "Overall Score",
                "Latest-Window Score",
            ],
            [
                [
                    name,
                    format_percent(
                        overall_score["normalized_weights"].get(
                            name, 0.0
                        ),
                    ),
                    format_number(
                        overall_score["contributions"].get(
                            name, 0.0
                        ),
                    ),
                    format_number(
                        latest_score["contributions"].get(
                            name, 0.0
                        ),
                    ),
                ]
                for name in overall_components
            ],
        ),
        "",
        "## Result Distribution",
        "",
        markdown_table(
            ["Result", "Count", "Rate"],
            result_distribution(df),
        ) if result_distribution(df) else "No result data available.",
        "",
        "## Overall Statistics",
        "",
        markdown_table(
            [
                "Metric",
                "Count",
                "Mean",
                "Std",
                "Min",
                "Q1",
                "Median",
                "Q3",
                "Max",
            ],
            calculate_statistics(df),
        ) if calculate_statistics(df) else "No numeric statistics available.",
        "",
        "## First Window vs. Latest Window",
        "",
        f"The first and latest windows each contain up to "
        f"**{actual_window:,} episodes**.",
        "",
        markdown_table(
            [
                "Metric",
                "First Window",
                "Latest Window",
                "Relative Change",
            ],
            compare_windows(
                first_window,
                latest_window,
            ),
        ),
        "",
        "For the success rate, the change is reported in percentage "
        "points. For other metrics, relative change is calculated "
        "against the absolute first-window mean.",
        "",
        "## Correlation Matrix",
        "",
    ]

    correlation_rows = correlation_table(df)

    if correlation_rows:
        correlation_headers = [
            "Metric",
            *[
                column
                for column in [
                    "steps",
                    "total_reward",
                    "initial_angle_degrees",
                    "initial_cart_velocity",
                    "initial_angular_velocity",
                    "mean_force",
                    "mean_absolute_force",
                    "maximum_absolute_force",
                ]
                if column in df.columns
            ],
        ]

        lines.extend(
            [
                markdown_table(
                    correlation_headers,
                    correlation_rows,
                ),
                "",
                "Correlations describe linear associations and do not "
                "establish causation.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Not enough numeric columns to calculate correlations.",
                "",
            ]
        )

    lines.extend(
        generate_formula_markdown(weights)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Analysis completed: {output_path}")
    print(
        f"Overall score: "
        f"{format_number(overall_score['total'])} / 100"
    )
    print(
        f"Latest-window score: "
        f"{format_number(latest_score['total'])} / 100"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze reinforcement-learning training results "
            "and generate a Markdown report."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Path to the training-results CSV file.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the JSON configuration file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_analysis.md"),
        help="Path to the output Markdown report.",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(
            f"CSV file does not exist: {args.csv}"
        )

    try:
        analyze(
            csv_path=args.csv,
            config_path=args.config,
            output_path=args.output,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as exc:
        parser.exit(
            status=1,
            message=f"Error: {exc}\n",
        )


if __name__ == "__main__":
    main()