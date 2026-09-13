"""
Two-Stage Transfer Learning Training Pipeline.

Implements:
- Stage 1: Feature Extraction with frozen CNN backbone.
- Stage 2: Fine-Tuning with unfrozen deeper layers and reduced learning rate.
- Class-weight computation for imbalance handling.
- Automated callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger).
- Metric plotting and model persistence.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.config import config
from src.utils import (
    set_random_seeds,
    setup_logger,
    save_json,
    plot_training_history,
    get_model_parameter_counts
)
from src.data_loader import (
    validate_dataset_structure,
    get_class_names_from_directory,
    compute_imbalance_weights,
    load_dataset_pipelines
)
from src.augmentation import build_augmentation_pipeline, augment_training_dataset
from src.model import build_disease_model, unfreeze_for_fine_tuning

logger = setup_logger("train", config.LOGS_PATH / "train.log")


def train_model(
    model_name: str = config.DEFAULT_MODEL,
    epochs_stage1: int = config.STAGE1_EPOCHS,
    epochs_stage2: int = config.STAGE2_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    initial_lr: float = config.INITIAL_LEARNING_RATE,
    fine_tune_lr: float = config.FINE_TUNE_LEARNING_RATE,
    use_class_weights: bool = True,
    use_augmentation: bool = True
) -> Path:
    """
    Execute complete two-stage transfer learning procedure.

    Args:
        model_name: Name of CNN architecture to train.
        epochs_stage1: Maximum epochs for Stage 1 (Feature Extraction).
        epochs_stage2: Maximum epochs for Stage 2 (Fine-Tuning).
        batch_size: Batch size for training.
        initial_lr: Learning rate for Stage 1.
        fine_tune_lr: Learning rate for Stage 2.
        use_class_weights: Whether to apply class weighting for imbalance.
        use_augmentation: Whether to apply biological data augmentation.

    Returns:
        Path to best saved model (.keras).
    """
    set_random_seeds(config.RANDOM_SEED)
    config.ensure_directories()

    logger.info(f"=== Initiating Training Workflow for '{model_name}' ===")

    # 1. Validate dataset & inspect class distributions
    summary = validate_dataset_structure(config)
    class_names = get_class_names_from_directory(config.TRAIN_DIR)
    num_classes = len(class_names)
    logger.info(f"Target classes ({num_classes}): {class_names}")

    # 2. Compute class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = compute_imbalance_weights(config.TRAIN_DIR, class_names)
        logger.info(f"Using class weights: {class_weights}")

    # 3. Load tf.data pipelines
    train_ds, val_ds, _, _ = load_dataset_pipelines(
        cfg=config,
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE
    )

    if use_augmentation:
        logger.info("Applying data augmentation to training pipeline.")
        aug_layer = build_augmentation_pipeline()
        train_ds = augment_training_dataset(train_ds, aug_layer)

    # 4. Construct initial model (Stage 1: Frozen Backbone)
    model = build_disease_model(
        model_name=model_name,
        num_classes=num_classes,
        input_shape=config.INPUT_SHAPE,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REGULARIZATION
    )

    model_checkpoint_path = config.MODELS_PATH / f"best_{model_name}.keras"
    csv_log_path = config.LOGS_PATH / f"training_{model_name}.csv"

    # Stage 1 Callbacks
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), separator=",", append=False)
    ]

    # Compile for Stage 1
    optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer_stage1,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    params_stage1 = get_model_parameter_counts(model)
    logger.info(f"Stage 1 Parameters: {params_stage1}")
    logger.info(f"--- Starting Stage 1 Training: Feature Extraction ({epochs_stage1} Epochs max) ---")

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1
    )

    history_stage1_dict = {k: [float(val) for val in v] for k, v in history_stage1.history.items()}

    # 5. Stage 2: Fine-Tuning
    history_stage2_dict = None
    if epochs_stage2 > 0:
        logger.info("--- Preparing Stage 2: Fine-Tuning Unfrozen Layers ---")
        model = unfreeze_for_fine_tuning(
            model,
            unfreeze_percent=config.FINE_TUNE_UNFREEZE_PERCENT
        )

        params_stage2 = get_model_parameter_counts(model)
        logger.info(f"Stage 2 Parameters: {params_stage2}")

        optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
        model.compile(
            optimizer=optimizer_stage2,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks_stage2 = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_checkpoint_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LEARNING_RATE,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(csv_log_path), separator=",", append=True)
        ]

        logger.info(f"--- Starting Stage 2 Training: Fine-Tuning ({epochs_stage2} Epochs max) ---")
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage2,
            class_weight=class_weights,
            callbacks=callbacks_stage2,
            verbose=1
        )
        history_stage2_dict = {k: [float(val) for val in v] for k, v in history_stage2.history.items()}

    # 6. Save combined training histories
    full_history = {
        "model_name": model_name,
        "stage1": history_stage1_dict,
        "stage2": history_stage2_dict
    }
    history_json_path = config.LOGS_PATH / f"history_{model_name}.json"
    save_json(full_history, history_json_path)

    # 7. Generate and save training curve visualizations
    curves_plot_path = config.RESULTS_PATH / f"training_curves_{model_name}.png"
    plot_training_history(
        history_stage1=history_stage1_dict,
        history_stage2=history_stage2_dict,
        save_path=curves_plot_path,
        model_name=model_name.replace("_", " ").title()
    )

    logger.info(f"Training completed. Best model saved to '{model_checkpoint_path}'.")
    logger.info(f"Training curves saved to '{curves_plot_path}'.")
    return model_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a lightweight CNN for leaf/fruit disease detection.")
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=config.SUPPORTED_MODELS,
        help="Model architecture"
    )
    parser.add_argument("--epochs_stage1", type=int, default=config.STAGE1_EPOCHS, help="Stage 1 epochs")
    parser.add_argument("--epochs_stage2", type=int, default=config.STAGE2_EPOCHS, help="Stage 2 epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--initial_lr", type=float, default=config.INITIAL_LEARNING_RATE, help="Stage 1 learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=config.FINE_TUNE_LEARNING_RATE, help="Stage 2 learning rate")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class imbalance weighting")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        initial_lr=args.initial_lr,
        fine_tune_lr=args.fine_tune_lr,
        use_class_weights=not args.no_class_weights,
        use_augmentation=not args.no_augmentation
    )
