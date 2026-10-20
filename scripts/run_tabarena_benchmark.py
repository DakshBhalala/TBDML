"""Run official TabArena (Suite 457) benchmark evaluation comparing CompactTab against baselines."""

import os

import pandas as pd
from rich.console import Console
from rich.table import Table

from compacttab.data.tabarena import load_tabarena_task
from compacttab.evaluation.runner import BenchmarkRunner

console = Console()


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table."""
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.itertuples(index=False)]
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_str = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    row_strs = [
        "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows
    ]
    return "\n".join([header_str, sep_str] + row_strs)


def main():
    output_dir = "experiments/tabarena"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    console.print("[bold green]=====================================================[/bold green]")
    console.print("[bold green]        Official TabArena (Suite 457) Benchmark       [/bold green]")
    console.print("[bold green]=====================================================[/bold green]\n")

    runner = BenchmarkRunner(output_dir=output_dir, random_seed=42)

    # 4 Representative TabArena Benchmark Tasks
    tabarena_tasks = [
        (363612, "airfoil_self_noise"),
        (363621, "blood_transfusion"),
        (363614, "anneal"),
        (363619, "bank_churn"),
    ]

    models_to_test = [
        "Dummy",
        "RandomForest",
        "HistGradientBoosting",
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "TabularMLP",
        "TabularResNet",
        "CompactTab-Tiny",
        "CompactTab-Small",
    ]

    all_results = []

    for task_id, task_label in tabarena_tasks:
        console.print(f"\n[bold cyan]>>> Fetching TabArena Task {task_id}: {task_label} <<<[/bold cyan]")
        try:
            X_tr, X_val, X_te, y_tr, y_val, y_te, schema = load_tabarena_task(
                task_id=task_id, random_state=42
            )
            console.print(
                f"Task type: {schema.task_type.value} | Train N: {len(X_tr)} | Test N: {len(X_te)} | Num: {len(schema.numerical_cols)} | Cat: {len(schema.categorical_cols)}"
            )

            res_df = runner.run_dataset_benchmark(
                dataset_name=f"TabArena_{task_label}",
                df_train=X_tr,
                df_val=X_val,
                df_test=X_te,
                y_train=y_tr,
                y_val=y_val,
                y_test=y_te,
                schema=schema,
                model_names=models_to_test,
                max_epochs=30,
                batch_size=128,
            )
            all_results.append(res_df)

            # Print rich table summary
            t = Table(title=f"TabArena Results: {task_label} ({schema.task_type.value})")
            t.add_column("Model", style="cyan")
            t.add_column("Params", justify="right")
            t.add_column("Fit Time (s)", justify="right")
            t.add_column("Lat (ms)", justify="right")

            if schema.is_classification:
                t.add_column("ROC-AUC", style="green", justify="right")
                t.add_column("Accuracy", justify="right")
                for _, r in res_df.iterrows():
                    t.add_row(
                        str(r.get("model")),
                        f"{int(r.get('params', 0)):,}",
                        f"{r.get('fit_time_sec', 0.0):.2f}",
                        f"{r.get('latency_ms_per_sample', 0.0):.3f}",
                        f"{r.get('roc_auc', float('nan')):.4f}",
                        f"{r.get('accuracy', float('nan')):.4f}",
                    )
            else:
                t.add_column("RMSE", style="green", justify="right")
                t.add_column("R2", justify="right")
                for _, r in res_df.iterrows():
                    t.add_row(
                        str(r.get("model")),
                        f"{int(r.get('params', 0)):,}",
                        f"{r.get('fit_time_sec', 0.0):.2f}",
                        f"{r.get('latency_ms_per_sample', 0.0):.3f}",
                        f"{r.get('rmse', float('nan')):.4f}",
                        f"{r.get('r2', float('nan')):.4f}",
                    )

            console.print(t)

        except Exception as e:
            console.print(f"[bold red]Failed on TabArena task {task_id}: {e}[/bold red]")

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_csv = os.path.join(output_dir, "tabarena_summary.csv")
        master_df.to_csv(master_csv, index=False)
        console.print(f"\n[bold green]Saved TabArena Summary to: {master_csv}[/bold green]")

        # Generate TabArena report
        tabarena_report = "reports/tabarena_report.md"
        with open(tabarena_report, "w", encoding="utf-8") as f:
            f.write("# Official TabArena (Suite 457) Benchmark Report\n\n")
            f.write("**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) / 16GB RAM\n\n")
            for dataset_name, group in master_df.groupby("dataset"):
                task = group["task_type"].iloc[0]
                f.write(f"## {dataset_name} ({task.capitalize()})\n\n")
                f.write(df_to_markdown_table(group) + "\n\n---\n\n")

        console.print(f"[bold green]Saved TabArena Benchmark Report to: {tabarena_report}[/bold green]")


if __name__ == "__main__":
    main()
