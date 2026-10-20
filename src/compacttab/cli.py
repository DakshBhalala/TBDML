"""Command Line Interface for CompactTab."""

import os
import sys

import pandas as pd
import psutil
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from compacttab.data.loaders import (
    load_breast_cancer_dataset,
    load_california_housing_dataset,
    load_diabetes_dataset,
    load_wine_dataset,
)

app = typer.Typer(
    name="compacttab",
    help="CompactTab: State-of-the-Art Compact Tabular Deep Learning System",
    add_completion=False,
)
console = Console()


@app.command("env-info")
def env_info() -> None:
    """Display environment, CPU, RAM, and GPU hardware resources."""
    table = Table(title="CompactTab Hardware & Runtime Environment", show_header=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Python
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", sys.platform)

    # CPU & RAM
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    ram_avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    table.add_row("CPU Logical Cores", str(cpu_count))
    table.add_row("System RAM (Total / Avail)", f"{ram_gb} GB / {ram_avail_gb} GB")

    # PyTorch & CUDA
    table.add_row("PyTorch Version", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    table.add_row("CUDA Acceleration Available", str(cuda_avail))

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        cuda_version = torch.version.cuda
        table.add_row("GPU Device Name", device_name)
        table.add_row("Total Dedicated VRAM", f"{vram_gb} GB")
        table.add_row("CUDA Runtime Version", str(cuda_version))
    else:
        table.add_row("GPU Device Name", "N/A (CPU execution mode)")

    console.print(table)


@app.command("smoke-benchmark")
def smoke_benchmark(
    output_dir: str = typer.Option("experiments", help="Directory to save experiment outputs"),
    max_epochs: int = typer.Option(25, help="Maximum epochs for neural training"),
    fast: bool = typer.Option(True, help="Run on compact benchmark subset"),
) -> None:
    """Run reproducible local smoke benchmark on classification and regression datasets."""
    from compacttab.evaluation.runner import BenchmarkRunner

    console.print(
        Panel.fit(
            "[bold green]Starting Local Reproducible Smoke Benchmark[/bold green]\n"
            "Evaluating CompactTab against strong tree-based and neural baselines.",
            border_style="green",
        )
    )

    runner = BenchmarkRunner(output_dir=output_dir, random_seed=42)

    # Select datasets
    tasks = [
        ("BreastCancer", load_breast_cancer_dataset()),
        ("CaliforniaHousing", load_california_housing_dataset()),
    ]
    if not fast:
        tasks.extend(
            [
                ("Wine", load_wine_dataset()),
                ("Diabetes", load_diabetes_dataset()),
            ]
        )

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

    all_dfs = []

    for name, data_tuple in tasks:
        df_tr, df_val, df_te, y_tr, y_val, y_te, schema = data_tuple
        console.print(f"\n[bold blue]==> Evaluating on {name} ({schema.task_type.value}) <==[/bold blue]")
        res_df = runner.run_dataset_benchmark(
            dataset_name=name,
            df_train=df_tr,
            df_val=df_val,
            df_test=df_te,
            y_train=y_tr,
            y_val=y_val,
            y_test=y_te,
            schema=schema,
            model_names=models_to_test,
            max_epochs=max_epochs,
        )
        all_dfs.append(res_df)

        # Print summary table
        res_table = Table(title=f"Results: {name} ({schema.task_type.value})")
        res_table.add_column("Model", style="cyan")
        res_table.add_column("Params", justify="right")
        res_table.add_column("Fit Time (s)", justify="right")
        res_table.add_column("Lat (ms)", justify="right")

        if schema.is_classification:
            res_table.add_column("ROC-AUC", style="green", justify="right")
            res_table.add_column("Accuracy", justify="right")
            for _, row in res_df.iterrows():
                res_table.add_row(
                    str(row.get("model")),
                    f"{int(row.get('params', 0)):,}",
                    f"{row.get('fit_time_sec', 0.0):.2f}",
                    f"{row.get('latency_ms_per_sample', 0.0):.3f}",
                    f"{row.get('roc_auc', float('nan')):.4f}",
                    f"{row.get('accuracy', float('nan')):.4f}",
                )
        else:
            res_table.add_column("RMSE", style="green", justify="right")
            res_table.add_column("R2", justify="right")
            for _, row in res_df.iterrows():
                res_table.add_row(
                    str(row.get("model")),
                    f"{int(row.get('params', 0)):,}",
                    f"{row.get('fit_time_sec', 0.0):.2f}",
                    f"{row.get('latency_ms_per_sample', 0.0):.3f}",
                    f"{row.get('rmse', float('nan')):.4f}",
                    f"{row.get('r2', float('nan')):.4f}",
                )

        console.print(res_table)

    # Save aggregated table
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(output_dir, "smoke_benchmark_summary.csv")
    combined_df.to_csv(combined_csv, index=False)
    console.print(f"\n[bold green]Smoke benchmark complete! Summary saved to {combined_csv}[/bold green]")


@app.command("compare")
def compare_experiments(
    experiments_dir: str = typer.Option("experiments", help="Path to experiments directory"),
) -> None:
    """Compare and display registered experiment results from CSVs."""
    if not os.path.exists(experiments_dir):
        console.print(f"[red]Directory {experiments_dir} does not exist.[/red]")
        return

    csv_files = [f for f in os.listdir(experiments_dir) if f.endswith("_results.csv")]
    if not csv_files:
        console.print(f"[yellow]No experiment CSV results found in {experiments_dir}.[/yellow]")
        return

    for csv_file in csv_files:
        path = os.path.join(experiments_dir, csv_file)
        df = pd.read_csv(path)
        table = Table(title=f"Experiment Summary: {csv_file}")
        for col in df.columns:
            table.add_column(col)
        for _, row in df.iterrows():
            table.add_row(*[str(val) for val in row])
        console.print(table)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate."""
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.itertuples(index=False)]
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]

    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_str = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    row_strs = [
        "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows
    ]

    return "\n".join([header_str, sep_str] + row_strs)


@app.command("report")
def generate_report(
    experiments_dir: str = typer.Option("experiments", help="Directory containing experiment CSVs"),
    output_path: str = typer.Option("reports/benchmark_report.md", help="Destination markdown report path"),
) -> None:
    """Generate a formal markdown benchmark report from recorded experiments."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    csv_files = [f for f in os.listdir(experiments_dir) if f.endswith("_results.csv")]

    content = [
        "# CompactTab Benchmark Report",
        "\n*Generated automatically from registered experiment results.*",
        "\n---\n",
    ]

    for csv_file in csv_files:
        path = os.path.join(experiments_dir, csv_file)
        df = pd.read_csv(path)
        dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else csv_file
        task_type = df["task_type"].iloc[0] if "task_type" in df.columns else "unknown"

        content.append(f"## Dataset: {dataset_name} ({task_type})\n")
        content.append(df_to_markdown_table(df))
        content.append("\n\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    console.print(f"[bold green]Report successfully written to {output_path}[/bold green]")


if __name__ == "__main__":
    app()
