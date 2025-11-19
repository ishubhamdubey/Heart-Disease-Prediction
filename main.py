"""Entry point for Heart Attack Prediction from Retinal Images.

Usage (basic):
    python main.py --data data/retina --epochs 5

Directory structure expected for real images:
data/retina/
    healthy/*.jpg|png
    risk/*.jpg|png

If folders are absent, the pipeline falls back to simulated tabular data.
"""
import argparse
import os
from src.training.train import train_all


def parse_args():
    parser = argparse.ArgumentParser(description="Heart Attack Risk Prediction from Retinal Images")
    parser.add_argument("--data", default="data", help="Path to dataset root (contains healthy/ and risk/)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for CNN")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Starting training with data dir={args.data}, epochs={args.epochs}")
    artifacts = train_all(args.data, epochs=args.epochs)
    # Basic inference demo if CNN present
    if artifacts.cnn is not None:
        print("[INFO] CNN model summary:")
        artifacts.cnn.summary()
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
