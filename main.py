"""EcoSort AI ♻️ - Unified Command-Line Interface (CLI).

Provides a single master entrypoint for all platform operations:
    python main.py app                      # Launch Streamlit web platform
    python main.py classify <path/to/img>   # Classify a single waste item
    python main.py evaluate                 # Benchmark model on held-out test set
    python main.py train [--quick]          # Train MobileNetV2 transfer learning
    python main.py prepare [--force]        # Split data/raw into stratified splits
    python main.py download [--from-zip]    # Extract or download dataset
    python main.py test                     # Run automated quality test suite
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def cmd_classify(args):
    """Run model inference on single image."""
    from PIL import Image

    from src.predict import load_disposal_guide, load_model_and_classes, predict_image

    img_path = Path(args.image_path)
    if not img_path.is_file():
        print(f"Error: Image file not found at: {img_path}")
        sys.exit(1)

    model, class_names = load_model_and_classes()
    print(f"Loaded model ({model.backend} backend)")
    res = predict_image(model, class_names, Image.open(img_path), top_k=args.top_k)
    guide = load_disposal_guide().get(res["label"], {})

    print("\n==================================================")
    print(f"Prediction : {res['label'].upper()}  ({res['confidence']:.1%})")
    print("--------------------------------------------------")
    for label, p in res["top_k"]:
        print(f"  {label:<14} {p:.1%}")
    print("--------------------------------------------------")
    if guide:
        print(f"Disposal Bin : {guide.get('bin', 'N/A')}")
        print(f"Recyclable   : {'Yes' if guide.get('recyclable', True) else 'No'}")
        print("Handling Steps:")
        for step in guide.get("steps", []):
            print(f"  - {step}")
    print("==================================================\n")


def cmd_evaluate(args):
    """Evaluate model on held-out test dataset."""
    from src.evaluate import evaluate_test_set
    evaluate_test_set(batch_size=args.batch_size)


def cmd_train(args):
    """Execute model training and fine-tuning."""
    from src.train import train_pytorch
    train_pytorch(
        epochs=args.epochs,
        fine_tune=args.fine_tune,
        fine_epochs=args.fine_epochs,
        quick=args.quick,
    )


def cmd_prepare(args):
    """Split dataset into stratified train, val, and test subsets."""
    from src.prepare_data import split_dataset
    split_dataset(force=args.force, split_ratios=tuple(args.split) if args.split else None)


def cmd_download(args):
    """Download or extract raw dataset."""
    from src.download_dataset import main as dl_main
    dl_main()


def cmd_app(args):
    """Launch Streamlit web platform."""
    app_path = BASE_DIR / "app.py"
    port_arg = ["--server.port", str(args.port)] if args.port else []
    print("Launching EcoSort AI Streamlit Platform...", flush=True)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + port_arg
    subprocess.run(cmd)


def cmd_test(args):
    """Execute pytest test suite."""
    print("Running EcoSort AI automated test suite...", flush=True)
    cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
    if args.coverage:
        try:
            import pytest_cov  # noqa: F401
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        except ImportError:
            print(
                "Notice: 'pytest-cov' is not installed. To view coverage, install it via: pip install pytest-cov",
                flush=True,
            )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="ecosort",
        description="EcoSort AI - Enterprise Circular Material Intelligence Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. app
    p_app = subparsers.add_parser("app", help="Launch Streamlit Web Platform")
    p_app.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server")
    p_app.set_defaults(func=cmd_app)

    # 2. classify
    p_cls = subparsers.add_parser("classify", help="Classify a single waste item photo")
    p_cls.add_argument("image_path", type=str, help="Path to image file")
    p_cls.add_argument("--top-k", type=int, default=3, help="Number of top candidates to display")
    p_cls.set_defaults(func=cmd_classify)

    # 3. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark model on test dataset")
    p_eval.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    p_eval.set_defaults(func=cmd_evaluate)

    # 4. train
    p_train = subparsers.add_parser("train", help="Train MobileNetV2 transfer learning model")
    p_train.add_argument("--epochs", type=int, default=3, help="Phase 1 head training epochs")
    p_train.add_argument("--fine-tune", action="store_true", help="Run Phase 2 fine-tuning")
    p_train.add_argument("--fine-epochs", type=int, default=2, help="Phase 2 fine-tuning epochs")
    p_train.add_argument("--quick", action="store_true", help="Quick mode (balanced subset)")
    p_train.set_defaults(func=cmd_train)

    # 5. prepare
    p_prep = subparsers.add_parser("prepare", help="Partition data/raw into stratified splits")
    p_prep.add_argument("--force", action="store_true", help="Redo split even if data/processed exists")
    p_prep.add_argument("--split", nargs=3, type=float, default=None, help="Train, val, test ratios (e.g. 0.70 0.15 0.15)")
    p_prep.set_defaults(func=cmd_prepare)

    # 6. download
    p_dl = subparsers.add_parser("download", help="Download or unpack Kaggle dataset")
    p_dl.add_argument("--from-zip", type=Path, default=None, help="Path to archive .zip")
    p_dl.add_argument("--from-dir", type=Path, default=None, help="Folder with 12 class dirs")
    p_dl.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_dl.set_defaults(func=cmd_download)

    # 7. test
    p_test = subparsers.add_parser("test", help="Run automated test suite")
    p_test.add_argument("--coverage", action="store_true", help="Calculate test coverage")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
