import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import PreprocessingFilters


CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASELINE_MODELS = [
    "VGG16",
    "ResNet50V2",
    "MobileNetV2",
    "EfficientNetB0",
    "DenseNet121_vanilla",
]


def configure_gpu():
    """Enable memory growth to avoid OOM on GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"GPU(s) found: {[g.name for g in gpus]} — memory growth enabled")
    else:
        print("No GPU found, running on CPU")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on TestA and TestB."
    )
    parser.add_argument(
        "--data-root",
        default="data/raabin-wbc-data",
        help="Root directory containing TestA and TestB folders.",
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/baseline_results",
        help="Directory containing saved baseline .keras files.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a single baseline model. Overrides --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all baseline models found in --models-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline_results/evaluations",
        help="Root directory for reports and confusion matrices.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per split (quick sanity check).",
    )
    parser.add_argument(
        "--tta",
        choices=["none", "light"],
        default="light",
        help="'light': 8-augment average. 'none': single pass.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU even when a GPU is available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def resolve_model_paths(args):
    """Return list of (model_name, model_path) to evaluate."""
    if args.model_path:
        p = Path(args.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return [(p.stem, p)]

    if args.all:
        models_dir = Path(args.models_dir)
        found = []
        for name in BASELINE_MODELS:
            candidate = models_dir / f"{name}_best.keras"
            if candidate.exists():
                found.append((name, candidate))
            else:
                print(f"Warning: {candidate} not found, skipping.")
        if not found:
            raise FileNotFoundError(f"No baseline models found in {models_dir}")
        return found

    raise ValueError("Specify --model-path <path> or --all")


def load_model(model_path):
    """Load a standard Keras model with no custom objects."""
    return tf.keras.models.load_model(str(model_path))


def collect_samples(split_dir, limit=None):
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    available = [c for c in CLASS_NAMES if (split_path / c).exists()]
    unknown = [
        p.name for p in split_path.iterdir()
        if p.is_dir() and p.name not in CLASS_NAMES
    ]
    if unknown:
        print(f"Warning: skipped unrecognized folders: {', '.join(sorted(unknown))}")
    if not available:
        raise ValueError(f"No class folders found in: {split_path}")

    samples = []
    for class_name in available:
        for image_path in sorted((split_path / class_name).iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append((image_path, class_name))
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def preprocess_image(image_path):
    """Load, resize, and apply medical_enhanced — same pipeline as training."""
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot read image: {image_path}") from exc

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))
    return PreprocessingFilters.medical_enhanced(image_np)


def build_tta_batch(image):
    """8-variant deterministic TTA batch identical to class.py."""
    return np.array([
        image,
        np.fliplr(image),
        np.flipud(image),
        np.rot90(image, 1),
        np.rot90(image, 2),
        np.rot90(image, 3),
        np.clip(image * 1.1, 0.0, 1.0),
        np.clip(image * 0.9, 0.0, 1.0),
    ], dtype=np.float32)


def predict_samples(model, samples, batch_size, tta_mode):
    label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_true, y_pred, probabilities, file_paths = [], [], [], []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_images, batch_true, batch_paths = [], [], []

        for image_path, class_name in batch:
            try:
                batch_images.append(preprocess_image(image_path))
                batch_true.append(label_to_index[class_name])
                batch_paths.append(str(image_path))
            except ValueError as exc:
                print(f"Warning: {exc}")

        if not batch_images:
            continue

        if tta_mode == "none":
            preds = model.predict(
                np.array(batch_images, dtype=np.float32), verbose=0
            )
            preds = np.array(preds, dtype=np.float32)
        else:
            agg = []
            for img in batch_images:
                tta_batch = build_tta_batch(img)
                tta_preds = model.predict(tta_batch, verbose=0)
                agg.append(np.mean(np.array(tta_preds, dtype=np.float32), axis=0))
            preds = np.array(agg, dtype=np.float32)

        predicted_indices = np.argmax(preds, axis=1)
        y_true.extend(batch_true)
        y_pred.extend(predicted_indices.tolist())
        probabilities.extend(preds.tolist())
        file_paths.extend(batch_paths)

    return y_true, y_pred, probabilities, file_paths


def compute_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def build_classification_report(y_true, y_pred):
    matrix = compute_confusion_matrix(y_true, y_pred)
    supports = matrix.sum(axis=1)
    total = int(supports.sum())

    rows, precisions, recalls, f1_scores = [], [], [], []
    for idx, name in enumerate(CLASS_NAMES):
        tp = matrix[idx, idx]
        pp = matrix[:, idx].sum()
        ap = supports[idx]
        prec = tp / pp if pp else 0.0
        rec = tp / ap if ap else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        rows.append((name, prec, rec, f1, int(ap)))

    accuracy = float(np.trace(matrix) / total) if total else 0.0
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    w_p = float(np.average(precisions, weights=supports)) if total else 0.0
    w_r = float(np.average(recalls, weights=supports)) if total else 0.0
    w_f1 = float(np.average(f1_scores, weights=supports)) if total else 0.0

    lines = [f"{'class':<15}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>10}", ""]
    for name, p, r, f, s in rows:
        lines.append(f"{name:<15}{p:>12.4f}{r:>12.4f}{f:>12.4f}{s:>10d}")
    lines.append("")
    lines.append(f"{'accuracy':<15}{'':>12}{'':>12}{accuracy:>12.4f}{total:>10d}")
    lines.append(f"{'macro avg':<15}{macro_p:>12.4f}{macro_r:>12.4f}{macro_f1:>12.4f}{total:>10d}")
    lines.append(f"{'weighted avg':<15}{w_p:>12.4f}{w_r:>12.4f}{w_f1:>12.4f}{total:>10d}")

    return "\n".join(lines), matrix


def save_confusion_matrix(y_true, y_pred, output_path, title):
    matrix = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(img, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(
                c, r, matrix[r, c],
                ha="center", va="center",
                color="white" if matrix[r, c] > threshold else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(output_path, file_paths, y_true, y_pred, probabilities):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        header = ["file_path", "true_label", "pred_label"] + [
            f"prob_{n}" for n in CLASS_NAMES
        ]
        f.write(",".join(header) + "\n")
        for fp, ti, pi, probs in zip(file_paths, y_true, y_pred, probabilities):
            row = [fp, CLASS_NAMES[ti], CLASS_NAMES[pi]] + [
                f"{float(p):.6f}" for p in probs
            ]
            f.write(",".join(row) + "\n")


def evaluate_split(model, split_name, split_dir, output_dir, batch_size, tta_mode, limit=None):
    samples = collect_samples(split_dir, limit=limit)
    if not samples:
        print(f"{split_name}: no images found.")
        return None

    print(f"{split_name}: {len(samples)} images...")
    y_true, y_pred, probabilities, file_paths = predict_samples(
        model, samples, batch_size, tta_mode
    )

    report_text, _ = build_classification_report(y_true, y_pred)
    split_out = output_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    (split_out / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_predictions_csv(split_out / "predictions.csv", file_paths, y_true, y_pred, probabilities)
    save_confusion_matrix(y_true, y_pred, split_out / "confusion_matrix.png", f"{split_name} — {output_dir.name}")

    print(f"\n===== {split_name} =====")
    print(report_text)

    return {"y_true": y_true, "y_pred": y_pred, "probabilities": probabilities, "file_paths": file_paths}


def evaluate_combined(results, output_dir):
    combined_true, combined_pred, combined_probs, combined_paths = [], [], [], []
    for r in results:
        if not r:
            continue
        combined_true.extend(r["y_true"])
        combined_pred.extend(r["y_pred"])
        combined_probs.extend(r["probabilities"])
        combined_paths.extend(r["file_paths"])

    if not combined_true:
        return

    report_text, _ = build_classification_report(combined_true, combined_pred)
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_predictions_csv(combined_dir / "predictions.csv", combined_paths, combined_true, combined_pred, combined_probs)
    save_confusion_matrix(combined_true, combined_pred, combined_dir / "confusion_matrix.png", f"TestA+TestB — {output_dir.name}")

    print(f"\n===== COMBINED =====")
    print(report_text)


def evaluate_one_model(model_name, model_path, args, output_root):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    model = load_model(model_path)
    output_dir = output_root / model_name
    data_root = Path(args.data_root)

    results = []
    for split_name in ["TestA", "TestB"]:
        split_dir = data_root / split_name
        results.append(
            evaluate_split(
                model=model,
                split_name=split_name,
                split_dir=split_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                tta_mode=args.tta,
                limit=args.limit,
            )
        )

    evaluate_combined(results, output_dir)

    # Free GPU memory before next model
    del model
    tf.keras.backend.clear_session()


def main():
    args = parse_args()

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU disabled by --no-gpu flag")
    else:
        configure_gpu()

    model_list = resolve_model_paths(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Models to evaluate: {[name for name, _ in model_list]}")
    print(f"TTA mode: {args.tta}")
    print(f"Output root: {output_root}")

    for model_name, model_path in model_list:
        evaluate_one_model(model_name, model_path, args, output_root)

    print(f"\nAll done. Results in: {output_root}")


if __name__ == "__main__":
    main()
