from pathlib import Path

from metrics import format_summary_cell


def add_model_configuration_lines(lines, aggregate):
    if not aggregate["runs"]:
        return

    model_config = aggregate["runs"][0].get("config", {})
    if not model_config:
        return

    lines.extend([
        "",
        "### Configuration",
        "",
    ])
    for key, value in model_config.items():
        lines.append(f"- {key}: `{value}`")


def build_final_report_markdown(session_name, session_dir, config, aggregated_results):
    lines = [
        "# Final Report",
        "",
        f"- Session: `{session_name}`",
        f"- Created at: `{config['created_at']}`",
        f"- Updated at: `{config['updated_at']}`",
        f"- Dataset path: `{config['dataset_path']}`",
        f"- Output directory: `{session_dir}`",
    ]

    if config.get("resume_session"):
        lines.append(f"- Resume mode: `yes` (`{config['resume_session']}`)")

    if config.get("requested_models"):
        lines.append(f"- Models trained in this invocation: `{', '.join(config['requested_models'])}`")

    lines.extend([
        "- Model-specific hyperparameters can differ; see each model section.",
        "",
        "## Overall comparison",
        "",
        "| Model | Runs | Test accuracy | Kappa | Macro F1 | Weighted F1 | Test loss | Final val accuracy | Training time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for aggregate in aggregated_results.values():
        metrics = aggregate["aggregate_metrics"]
        lines.append(
            "| "
            f"{aggregate['display_name']} | "
            f"{aggregate['num_runs']} | "
            f"{format_summary_cell(metrics['test_accuracy'], scale=100.0, suffix='%')} | "
            f"{format_summary_cell(metrics['kappa'])} | "
            f"{format_summary_cell(metrics['macro_f1'])} | "
            f"{format_summary_cell(metrics['weighted_f1'])} | "
            f"{format_summary_cell(metrics['test_loss'])} | "
            f"{format_summary_cell(metrics['final_val_accuracy'], scale=100.0, suffix='%')} | "
            f"{format_summary_cell(metrics['total_training_time_seconds'], suffix=' s')} |"
        )

    for model_key, aggregate in aggregated_results.items():
        lines.extend([
            "",
            f"## {aggregate['display_name']}",
            "",
            f"- Number of runs: `{aggregate['num_runs']}`",
            f"- Aggregate files: `{Path(model_key)}` inside the session output directory",
        ])

        add_model_configuration_lines(lines, aggregate)

        lines.extend([
            "",
            "### Run details",
            "",
            "| Run | Seed | Test accuracy | Kappa | Test loss | Final val accuracy | Best val accuracy | Total training time |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])

        for index, summary in enumerate(aggregate["runs"], start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{summary['seed']} | "
                f"{summary['test']['accuracy_percent']:.4f}% | "
                f"{summary['test']['kappa']:.4f} | "
                f"{summary['test']['loss']:.4f} | "
                f"{summary['validation']['final_accuracy'] * 100.0:.4f}% | "
                f"{summary['validation']['best_accuracy'] * 100.0:.4f}% | "
                f"{summary['timing']['total_training_time_seconds']:.2f} s |"
            )

        lines.extend([
            "",
            "### Mean classification report",
            "",
            "| Class | Precision | Recall | F1-score | Support |",
            "| --- | --- | --- | --- | --- |",
        ])

        report = aggregate["aggregate_classification_report"]
        for key, values in report.items():
            if key == "accuracy":
                continue
            if not isinstance(values, dict):
                continue

            precision = format_summary_cell(values.get("precision"))
            recall = format_summary_cell(values.get("recall"))
            f1_score = format_summary_cell(values.get("f1-score"))
            support = values.get("support", "n/a")
            lines.append(f"| {key} | {precision} | {recall} | {f1_score} | {support} |")

    return "\n".join(lines) + "\n"
