from pathlib import Path

from experiments import (
    build_expected_model_config,
    run_custom_experiment,
    run_transfer_experiment,
    validate_existing_runs,
)
from metrics import (
    aggregate_classification_reports,
    aggregate_histories,
    ensure_dir,
    plot_learning_curves,
    read_json,
    summarize_scalars,
    write_json,
)


def build_model_aggregate(run_results):
    summaries = [result["summary"] for result in run_results]
    histories = [result["history"] for result in run_results]
    report_dicts = [result["classification_report"] for result in run_results]

    return {
        "display_name": summaries[0]["display_name"],
        "num_runs": len(run_results),
        "runs": summaries,
        "aggregate_metrics": {
            "test_accuracy": summarize_scalars([summary["test"]["accuracy"] for summary in summaries]),
            "kappa": summarize_scalars([summary["test"]["kappa"] for summary in summaries]),
            "macro_f1": summarize_scalars([summary["test"]["macro_f1"] for summary in summaries]),
            "weighted_f1": summarize_scalars([summary["test"]["weighted_f1"] for summary in summaries]),
            "test_loss": summarize_scalars([summary["test"]["loss"] for summary in summaries]),
            "final_val_accuracy": summarize_scalars([summary["validation"]["final_accuracy"] for summary in summaries]),
            "best_val_accuracy": summarize_scalars([summary["validation"]["best_accuracy"] for summary in summaries]),
            "final_val_loss": summarize_scalars([summary["validation"]["final_loss"] for summary in summaries]),
            "total_training_time_seconds": summarize_scalars(
                [summary["timing"]["total_training_time_seconds"] for summary in summaries]
            ),
            "mean_epoch_time_seconds": summarize_scalars(
                [summary["timing"]["mean_epoch_time_seconds"] for summary in summaries]
            ),
        },
        "aggregate_history": aggregate_histories(histories),
        "aggregate_classification_report": aggregate_classification_reports(report_dicts),
    }


def save_model_aggregate(model_dir, aggregate):
    ensure_dir(model_dir)
    write_json(model_dir / "aggregate_metrics.json", aggregate["aggregate_metrics"])
    write_json(model_dir / "aggregate_history.json", aggregate["aggregate_history"])
    write_json(
        model_dir / "aggregate_classification_report.json",
        aggregate["aggregate_classification_report"],
    )
    write_json(model_dir / "runs_summary.json", aggregate["runs"])

    mean_history = {
        "train_loss": aggregate["aggregate_history"]["train_loss"]["mean"],
        "train_accuracy": aggregate["aggregate_history"]["train_accuracy"]["mean"],
        "val_loss": aggregate["aggregate_history"]["val_loss"]["mean"],
        "val_accuracy": aggregate["aggregate_history"]["val_accuracy"]["mean"],
        "epoch_time_seconds": aggregate["aggregate_history"]["epoch_time_seconds"]["mean"],
    }
    plot_learning_curves(
        mean_history,
        model_dir / "mean_learning_curves.png",
        title_prefix=f"{aggregate['display_name']} Mean",
    )


def load_run_result(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.json"
    report_path = run_dir / "classification_report.json"

    required_paths = [metrics_path, history_path, report_path]
    if not all(path.exists() for path in required_paths):
        return None

    return {
        "summary": read_json(metrics_path),
        "history": read_json(history_path),
        "classification_report": read_json(report_path),
    }


def load_existing_run_results(model_dir):
    model_dir = Path(model_dir)
    run_results = []

    for run_dir in sorted(model_dir.glob("run_*")):
        run_result = load_run_result(run_dir)
        if run_result is None:
            print(f"Skipping incomplete run directory: {run_dir}")
            continue
        run_results.append(run_result)

    return run_results


def load_existing_model_aggregate(model_dir):
    model_dir = Path(model_dir)
    run_results = load_existing_run_results(model_dir)
    if run_results:
        return build_model_aggregate(run_results)

    aggregate_paths = {
        "aggregate_metrics": model_dir / "aggregate_metrics.json",
        "aggregate_history": model_dir / "aggregate_history.json",
        "aggregate_classification_report": model_dir / "aggregate_classification_report.json",
        "runs": model_dir / "runs_summary.json",
    }
    if not all(path.exists() for path in aggregate_paths.values()):
        return None

    runs = read_json(aggregate_paths["runs"])
    return {
        "display_name": runs[0]["display_name"] if runs else model_dir.name,
        "num_runs": len(runs),
        "runs": runs,
        "aggregate_metrics": read_json(aggregate_paths["aggregate_metrics"]),
        "aggregate_history": read_json(aggregate_paths["aggregate_history"]),
        "aggregate_classification_report": read_json(aggregate_paths["aggregate_classification_report"]),
    }


def run_requested_model(model_key, args, dataset_path, device, session_dir):
    model_dir = ensure_dir(Path(session_dir) / model_key)
    existing_run_results = load_existing_run_results(model_dir)

    if existing_run_results:
        validate_existing_runs(model_key, existing_run_results, build_expected_model_config(model_key, args))
        print(f"Found {len(existing_run_results)} completed existing runs for {model_key}")

    if len(existing_run_results) > args.runs:
        raise ValueError(
            f"Session already contains {len(existing_run_results)} completed runs for {model_key}, "
            f"which is greater than the requested {args.runs} runs."
        )

    run_results = list(existing_run_results)
    run_function = run_transfer_experiment if model_key == "transfer_learning" else run_custom_experiment

    for run_index in range(len(run_results), args.runs):
        seed = args.seed_base + run_index
        run_dir = model_dir / f"run_{run_index + 1:02d}"
        print(f"Running {model_key} - run {run_index + 1}/{args.runs} with seed {seed}")
        run_results.append(run_function(args, dataset_path, device, seed, run_dir))

    if not run_results:
        raise ValueError(f"No completed runs available for {model_key}")

    aggregate = build_model_aggregate(run_results)
    save_model_aggregate(model_dir, aggregate)
    return aggregate
