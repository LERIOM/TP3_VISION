import argparse


DEFAULT_RUNS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRANSFER_EPOCHS = 15
DEFAULT_CUSTOM_EPOCHS = 15
DEFAULT_SEED_BASE = 42
DEFAULT_NUM_WORKERS = 8
MODEL_DIR_BY_ARG = {
    "transfer": "transfer_learning",
    "custom": "custom_cnn",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated CNN experiments and save metrics.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for new runs.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--download-dir", default="data", help="Dataset download directory.")
    parser.add_argument("--output-dir", default="output", help="Root directory for experiment outputs.")
    parser.add_argument("--session-name", default=None, help="Optional name for a new output session directory.")
    parser.add_argument(
        "--resume-session",
        default=None,
        help="Existing session name or path to resume and merge with new runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["transfer", "custom"],
        default=["transfer", "custom"],
        help="Models to train in this invocation.",
    )
    parser.add_argument(
        "--custom-model",
        choices=["resnet_like", "fast_food"],
        default="resnet_like",
        help="Custom CNN architecture to use.",
    )
    parser.add_argument("--transfer-epochs", type=int, default=DEFAULT_TRANSFER_EPOCHS, help="Epochs for transfer.")
    parser.add_argument("--custom-epochs", type=int, default=DEFAULT_CUSTOM_EPOCHS, help="Epochs for custom CNN.")
    parser.add_argument("--transfer-lr", type=float, default=1e-3, help="Learning rate for transfer learning.")
    parser.add_argument("--custom-lr", type=float, default=1e-3, help="Learning rate for the custom CNN.")
    parser.add_argument(
        "--custom-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the custom CNN optimizer.",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Base seed for repeated runs.")
    return parser.parse_args()


def validate_args(args):
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.transfer_epochs < 1:
        raise ValueError("--transfer-epochs must be at least 1.")
    if args.custom_epochs < 1:
        raise ValueError("--custom-epochs must be at least 1.")
    if args.resume_session and args.session_name:
        raise ValueError("--resume-session and --session-name cannot be used together.")
