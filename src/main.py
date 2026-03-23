from datetime import datetime
from cli import MODEL_DIR_BY_ARG, parse_args, validate_args


def build_session_config(args, device, dataset_path, now, existing_config):
    return {
        **existing_config,
        "created_at": existing_config.get("created_at", now),
        "updated_at": now,
        "device": str(device),
        "dataset_path": str(dataset_path),
        "resume_session": str(args.resume_session) if args.resume_session else None,
        "requested_models": args.models,
        "runs": args.runs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "transfer_epochs": args.transfer_epochs,
        "custom_epochs": args.custom_epochs,
        "transfer_lr": args.transfer_lr,
        "custom_lr": args.custom_lr,
        "custom_weight_decay": args.custom_weight_decay,
        "seed_base": args.seed_base,
        "custom_model": args.custom_model,
    }


def main():
    args = parse_args()
    validate_args(args)

    from dataset import load_dataset
    from metrics import ensure_dir, read_json, write_json, write_text
    from reporting import build_final_report_markdown
    from runtime import get_device, resolve_project_path, resolve_session_dir
    from session_results import load_existing_model_aggregate, run_requested_model

    device = get_device()
    download_dir = resolve_project_path(args.download_dir)
    output_root = resolve_project_path(args.output_dir)
    dataset_path = load_dataset(str(download_dir))

    now = datetime.now().isoformat(timespec="seconds")
    existing_config = {}

    if args.resume_session:
        session_dir = resolve_session_dir(output_root, args.resume_session)
        if not session_dir.exists():
            raise FileNotFoundError(f"Resume session not found: {session_dir}")
        session_name = session_dir.name
        config_path = session_dir / "config.json"
        if config_path.exists():
            existing_config = read_json(config_path)
    else:
        session_name = args.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = ensure_dir(output_root / session_name)

    config = build_session_config(args, device, dataset_path, now, existing_config)
    write_json(session_dir / "config.json", config)

    print(f"Dataset path: {dataset_path}")
    print(f"Device: {device}")
    print(f"Saving outputs to: {session_dir}")

    requested_model_keys = [MODEL_DIR_BY_ARG[model_arg] for model_arg in args.models]
    aggregated_results = {}

    if args.resume_session:
        for model_key in MODEL_DIR_BY_ARG.values():
            if model_key in requested_model_keys:
                continue
            aggregate = load_existing_model_aggregate(session_dir / model_key)
            if aggregate is not None:
                aggregated_results[model_key] = aggregate
                print(f"Reusing existing results for {model_key}")

    for model_arg in args.models:
        model_key = MODEL_DIR_BY_ARG[model_arg]
        aggregated_results[model_key] = run_requested_model(model_key, args, dataset_path, device, session_dir)

    final_summary = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "config": config,
        "models": aggregated_results,
    }
    write_json(session_dir / "final_report.json", final_summary)
    write_text(
        session_dir / "final_report.md",
        build_final_report_markdown(session_name, session_dir, config, aggregated_results),
    )

    print(f"Final report written to: {session_dir / 'final_report.md'}")


if __name__ == "__main__":
    main()
