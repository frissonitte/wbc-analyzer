import argparse

import json

import random

import time

from pathlib import Path

import numpy as np

import tensorflow as tf

from src.preprocessing import PreprocessingFilters


keras = tf.keras

layers = keras.layers



# ─── CONFIG ────────────────────────────────────────────────────────────────

MODELS_TO_COMPARE = [

    "VGG16",

    "ResNet50V2",

    "MobileNetV2",

    "EfficientNetB0",

    "DenseNet121_vanilla",  # Attention/MedSwish olmadan

]


IMG_SIZE = 224

BATCH_SIZE = 32

PHASE1_EPOCHS = 12   # Backbone donduruldu

PHASE2_EPOCHS = 12   # Fine-tune

SEED = 42

VAL_FRACTION = 0.15



# ─── DATA LOADING ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_dataset(data_root: Path, split: str, class_names: list, val_fraction: float):

    """Görüntü yollarını ve etiketleri yükle, train/val olarak ayır."""

    split_dir = data_root / split

    all_paths, all_labels = [], []


    for idx, cls in enumerate(class_names):

        cls_dir = split_dir / cls

        if not cls_dir.exists():

            continue

        for p in cls_dir.iterdir():

            if p.suffix.lower() in IMAGE_EXTS:

                all_paths.append(str(p))

                all_labels.append(idx)


    combined = list(zip(all_paths, all_labels))

    random.shuffle(combined)

    all_paths, all_labels = zip(*combined)


    n_val = int(len(all_paths) * val_fraction)

    val_paths, val_labels = all_paths[:n_val], all_labels[:n_val]

    train_paths, train_labels = all_paths[n_val:], all_labels[n_val:]


    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")

    return list(train_paths), list(train_labels), list(val_paths), list(val_labels)



def make_tf_dataset(paths, labels, n_classes, batch_size, training=False):

    """

    tf.data.Dataset oluştur.


    Pipeline order:

      map(load+preprocess) → cache → [shuffle+augment if training] → batch → prefetch


    cache() ensures medical_enhanced runs once per image instead of once per epoch.

    """

    def load_and_preprocess(path, label):

        img = tf.io.read_file(path)

        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

        img = tf.cast(img, tf.uint8)

        # Inside tf.numpy_function the argument is already a numpy array.

        img = tf.numpy_function(PreprocessingFilters.medical_enhanced, [img], tf.float32)

        img.set_shape([IMG_SIZE, IMG_SIZE, 3])

        return img, tf.one_hot(label, n_classes)


    def augment(img, label):

        img = tf.image.random_flip_left_right(img)

        img = tf.image.random_flip_up_down(img)

        return img, label


    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache preprocessed images in RAM so epochs 2+ skip disk I/O and medical_enhanced.

    ds = ds.cache()


    if training:

        ds = ds.shuffle(len(paths), seed=SEED)

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)


    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# ─── MODEL BUILDERS ────────────────────────────────────────────────────────

def build_model(model_name: str, n_classes: int):

    """Her model için aynı head mimarisi — backbone farkı görmek istiyoruz."""

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    backbone_kwargs = dict(include_top=False, weights="imagenet", input_tensor=inp)


    if model_name == "VGG16":

        backbone = keras.applications.VGG16(**backbone_kwargs)

    elif model_name == "ResNet50V2":

        backbone = keras.applications.ResNet50V2(**backbone_kwargs)

    elif model_name == "MobileNetV2":

        backbone = keras.applications.MobileNetV2(**backbone_kwargs)

    elif model_name == "EfficientNetB0":

        backbone = keras.applications.EfficientNetB0(**backbone_kwargs)

    elif model_name == "DenseNet121_vanilla":

        backbone = keras.applications.DenseNet121(**backbone_kwargs)

    else:

        raise ValueError(f"Bilinmeyen model: {model_name}")


    backbone.trainable = False   # Phase 1: dondur


    x = backbone.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation="relu")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)


    model = keras.Model(inputs=inp, outputs=out)

    return model, backbone



def compile_model(model, lr=1e-3):

    model.compile(

        optimizer=keras.optimizers.Adam(lr),

        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),

        metrics=["accuracy"]

    )



# ─── METRICS ───────────────────────────────────────────────────────────────

def compute_metrics(model, val_ds, class_names):

    """Accuracy ve macro F1 hesapla."""

    from sklearn.metrics import classification_report, f1_score


    # Single pass with model.predict avoids per-batch Python overhead.

    y_pred_probs = model.predict(val_ds, verbose=0)

    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([

        np.argmax(batch_y.numpy(), axis=1) for _, batch_y in val_ds

    ])


    acc = float(np.mean(y_true == y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return acc, macro_f1, report



def measure_inference_time(model, n_runs=50) -> float:

    """Ortalama tek görüntü inference süresi (ms)."""

    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    # Warmup — ensures XLA/TF graph compilation happens before timing.

    for _ in range(5):

        model(dummy, training=False)

    times = []

    for _ in range(n_runs):

        t0 = time.perf_counter()

        model(dummy, training=False)

        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))



# ─── TRAINING ──────────────────────────────────────────────────────────────

def train_model(model_name, train_ds, val_ds, n_classes, phase1_epochs, phase2_epochs):

    print(f"\n{'='*60}")

    print(f"  Eğitim: {model_name}")

    print(f"{'='*60}")


    model, backbone = build_model(model_name, n_classes)

    total_params = model.count_params()

    results_dir = Path("outputs/baseline_results")

    checkpoint_path = results_dir / f"{model_name}_best.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_accuracy"),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), 
            save_best_only=True, 
            monitor="val_accuracy"
        )
    ]


    # Phase 1 — feature extraction

    print(f"\n  [Phase 1] Backbone donduruldu — {phase1_epochs} epoch")

    compile_model(model, lr=1e-3)

    model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs,

              callbacks=callbacks, verbose=1)


    # Phase 2 — fine-tune

    print(f"\n  [Phase 2] Fine-tune — {phase2_epochs} epoch")

    backbone.trainable = True

    compile_model(model, lr=5e-5)

    model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs,

              callbacks=callbacks, verbose=1)


    return model, total_params



# ─── LATEX TABLE ───────────────────────────────────────────────────────────

def generate_latex_table(results: list) -> str:

    """Karşılaştırma tablosu için LaTeX kodu üret."""

    display_names = {

        "VGG16": "VGG16~\\cite{simonyan2014very}",

        "ResNet50V2": "ResNet50V2~\\cite{he2016deep}",

        "MobileNetV2": "MobileNetV2~\\cite{howard2017mobilenets}",

        "EfficientNetB0": "EfficientNetB0~\\cite{tan2019efficientnet}",

        "DenseNet121_vanilla": "DenseNet121 (yalın)~\\cite{huang2017densely}",

    }


    header = r"""\begin{table}[htbp]

\centering

\caption{Transfer learning omurga mimarilerinin karşılaştırmalı analizi.

         Tüm modeller aynı veri ön işleme (Medical Enhanced Filter),

         aynı sınıflandırma başlığı ve iki aşamalı eğitim protokolüyle

         eğitilmiştir. En iyi değerler \textbf{kalın} ile gösterilmiştir.}

\label{tab:backbone_comparison}

\begin{tabular}{lrrrr}

\toprule

\textbf{Model} & \textbf{Param. (M)} & \textbf{Val. Acc.~(\%)} &

\textbf{Macro F1} & \textbf{Inf. (ms)} \\

\midrule"""


    rows = []

    all_accs = [r["val_accuracy"] for r in results]

    best_acc = max(all_accs) if all_accs else 0


    for r in results:

        dname = display_names.get(r["model"], r["model"])

        acc_str = f"{r['val_accuracy']*100:.2f}"

        if r["val_accuracy"] == best_acc:

            acc_str = f"\\textbf{{{acc_str}}}"

        f1_str = f"{r['macro_f1']:.4f}"

        params_m = r["total_params"] / 1e6

        rows.append(

            f"        {dname} & {params_m:.1f} & {acc_str} & {f1_str} & {r['inf_time_ms']:.1f} \\\\"

        )


    footer = r"""

\bottomrule

\end{tabular}

\end{table}"""


    return header + "\n" + "\n".join(rows) + footer



# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root",   default="data/raabin-wbc-data")

    parser.add_argument("--split",       default="Train")

    parser.add_argument("--results-dir", default="outputs/baseline_results")


    parser.add_argument("--fast",        action="store_true",

                        help="Epoch sayısını yarıya indir (hızlı deneme için)")

    parser.add_argument("--mixed-precision", action="store_true",

                        help="float16 mixed precision etkinleştir (uyumlu GPU gerektirir)")

    parser.add_argument("--models",      nargs="+", choices=MODELS_TO_COMPARE,

                        default=MODELS_TO_COMPARE,

                        help="Eğitilecek modelleri seç (varsayılan: hepsi)")

    parser.add_argument("--resume",      action="store_true",

                        help="Mevcut JSON'daki tamamlanmış modelleri atla")

    args = parser.parse_args()


    if args.mixed_precision:

        keras.mixed_precision.set_global_policy("mixed_float16")

        print("Mixed precision: float16 aktif")


    phase1_epochs = PHASE1_EPOCHS // 2 if args.fast else PHASE1_EPOCHS

    phase2_epochs = PHASE2_EPOCHS // 2 if args.fast else PHASE2_EPOCHS

    if args.fast:

        print(f"Fast mode: {phase1_epochs}+{phase2_epochs} epoch")


    random.seed(SEED)

    np.random.seed(SEED)

    tf.random.set_seed(SEED)


    data_root = Path(args.data_root)

    results_dir = Path(args.results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)


    # Sınıf isimlerini bul

    split_dir = data_root / args.split

    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    n_classes = len(class_names)

    print(f"Sınıflar ({n_classes}): {class_names}")


    train_paths, train_labels, val_paths, val_labels = load_dataset(

        data_root, args.split, class_names, VAL_FRACTION

    )


    train_ds = make_tf_dataset(train_paths, train_labels, n_classes, BATCH_SIZE, training=True)

    val_ds   = make_tf_dataset(val_paths, val_labels, n_classes, BATCH_SIZE, training=False)


    # Resume: load completed results from previous run

    out_json = results_dir / "comparison_results.json"

    if args.resume and out_json.exists():

        with open(out_json, encoding="utf-8") as f:

            all_results = json.load(f)

        completed = {r["model"] for r in all_results}

        print(f"Resume: {completed} atlanıyor")

    else:

        all_results = []

        completed = set()


    for model_name in args.models:

        if model_name in completed:

            print(f"  ↷ {model_name} zaten tamamlandı, atlanıyor")

            continue


        try:

            model, total_params = train_model(

                model_name, train_ds, val_ds, n_classes, phase1_epochs, phase2_epochs

            )


            acc, macro_f1, report = compute_metrics(model, val_ds, class_names)

            inf_time = measure_inference_time(model)


            result = {

                "model": model_name,

                "val_accuracy": acc,

                "macro_f1": macro_f1,

                "total_params": total_params,

                "inf_time_ms": inf_time,

                "per_class": report,

            }

            all_results.append(result)


            print(f"\n  ✓ {model_name}: acc={acc:.4f}  F1={macro_f1:.4f}  "

                  f"params={total_params/1e6:.2f}M  inf={inf_time:.1f}ms")


            # Save after each model so a crash doesn't lose completed results.

            with open(out_json, "w", encoding="utf-8") as f:

                json.dump(all_results, f, indent=2, ensure_ascii=False)

            tf.keras.backend.clear_session()


        except Exception as e:

            print(f"  ✗ {model_name} hata: {e}")


    print(f"\nSonuçlar kaydedildi: {out_json}")


    # LaTeX tablosu

    latex_code = generate_latex_table(

        all_results,

    )

    out_tex = results_dir / "backbone_comparison_table.tex"

    with open(out_tex, "w", encoding="utf-8") as f:

        f.write(latex_code)

    print(f"LaTeX tablosu kaydedildi: {out_tex}")


    # Terminalde özet

    print("\n" + "="*60)

    print("ÖZET")

    print("="*60)

    print(f"{'Model':<35} {'Acc':>8} {'F1':>8} {'Params(M)':>10} {'Inf(ms)':>8}")

    print("-"*60)

    for r in sorted(all_results, key=lambda x: x["val_accuracy"], reverse=True):

        print(f"{r['model']:<35} {r['val_accuracy']*100:>7.2f}% "

              f"{r['macro_f1']:>8.4f} {r['total_params']/1e6:>10.2f} {r['inf_time_ms']:>8.1f}")

    print("="*60)

    print(f"\nJSON: {out_json}")

    print(f"LaTeX: {out_tex}")



if __name__ == "__main__":

    main()