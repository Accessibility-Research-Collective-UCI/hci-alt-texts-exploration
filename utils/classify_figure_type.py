"""
Use a fine-tuned Vision Transformer to classify figures as:
plot, image, table, or other_figure.

Supports local image paths and HTTP/HTTPS image URLs loaded from a
CSV or JSON file. Images that cannot be loaded are retained in the
output with label_1 set to "Error".

Usage:
uv run utils/classify_figure_type.py --input-file processed_papers/spreadsheets/ASSETS_2021-2022-2023-2024-2025_papers.csv --image-field local_img_path --weights model_checkpoints/acl-fig_plot-image-table-other/model.safetensors --batch-size 128 --num-workers 12
"""

import argparse
import math
import shutil
import sys
from collections import Counter
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

Image.MAX_IMAGE_PIXELS = 933120000

TERMINAL_COLUMNS, _ = shutil.get_terminal_size()

MODEL_ID = "google/vit-base-patch16-224-in21k"
NEW_LABELS = [
    "plot",
    "image",
    "table",
    "other_figure",
]
REQUEST_TIMEOUT_SECONDS = 30

FileType: TypeAlias = Literal["csv", "json"]
PredictionRow: TypeAlias = dict[str, str | float]
DatasetItem: TypeAlias = tuple[str, Image.Image | None, str | None]


def require_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


class ImageDataset(Dataset):
    """
    Dataset containing local image paths and/or remote image URLs.

    Loading failures are returned as data rather than raised. This prevents
    one bad image from terminating the DataLoader and the full inference run.
    """

    def __init__(self, image_sources: list[str]):
        if not image_sources:
            raise ValueError("At least one image path or URL is required.")

        self.image_sources = [str(source) for source in image_sources]

    def __len__(self) -> int:
        return len(self.image_sources)

    def __getitem__(self, index: int) -> DatasetItem:
        source = self.image_sources[index]

        try:
            image = load_image(source)
            return source, image, None
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            return source, None, error_message


def get_local_path(source: str) -> Path | None:
    """
    Return a local Path for ordinary paths and file:// URLs.

    Return None for remote URLs.
    """
    parsed = urlparse(source)

    if parsed.scheme == "file":
        return Path(unquote(parsed.path))

    if parsed.scheme == "":
        return Path(source)

    return None


def load_image(source: str) -> Image.Image:
    """
    Load an RGB image from a local path or HTTP/HTTPS URL.
    """
    local_path = get_local_path(source)

    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Local image does not exist or is not a file: {local_path}"
            )

        with Image.open(local_path) as image:
            return image.convert("RGB").copy()

    parsed = urlparse(source)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported image source scheme: {parsed.scheme!r}")

    response = requests.get(
        source,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    with Image.open(BytesIO(response.content)) as image:
        return image.convert("RGB").copy()


def collate_images(
    batch: list[DatasetItem],
) -> tuple[list[str], list[Image.Image | None], list[str | None]]:
    """
    Pickling-safe collate function defined at module scope.
    """
    sources, images, errors = zip(*batch)
    return list(sources), list(images), list(errors)


def strip_prefix_from_all_keys(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return state_dict

    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def remove_wrapper_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove prefixes commonly added by DataParallel, torch.compile,
    or training-framework wrappers.
    """
    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
    )

    for prefix in prefixes:
        state_dict = strip_prefix_from_all_keys(state_dict, prefix)

    return state_dict


def get_square_patch_size(config: ViTConfig) -> int:
    patch_size = config.patch_size

    if isinstance(patch_size, int):
        return patch_size

    if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
        raise ValueError("This script expects square ViT patches.")

    return patch_size[0]


def infer_checkpoint_image_size(
    state_dict: dict[str, torch.Tensor],
    default_image_size: int,
    patch_size: int,
) -> int:
    """
    Infer the checkpoint image size from its positional embeddings.

    For ViT:
        number of positions = number of patches + 1 CLS token
        image size = sqrt(number of patches) * patch size
    """
    position_key = next(
        (
            key
            for key in state_dict
            if key.endswith("vit.embeddings.position_embeddings")
            or key.endswith("embeddings.position_embeddings")
        ),
        None,
    )

    if position_key is None:
        return default_image_size

    position_embeddings = state_dict[position_key]
    number_of_positions = position_embeddings.shape[1]
    number_of_patches = number_of_positions - 1
    grid_size = math.isqrt(number_of_patches)

    if grid_size * grid_size != number_of_patches:
        raise ValueError(
            "Could not infer a square image size from positional "
            f"embeddings with shape {tuple(position_embeddings.shape)}."
        )

    return grid_size * patch_size


def load_model(
    weights_path: str | Path,
    labels: list[str],
) -> tuple[ViTForImageClassification, int]:
    """
    Load the ViT architecture and fine-tuned safetensors weights.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Safetensors file does not exist: {weights_path}")

    state_dict = load_file(
        str(weights_path),
        device="cpu",
    )
    state_dict = remove_wrapper_prefixes(state_dict)

    config = ViTConfig.from_pretrained(MODEL_ID)
    checkpoint_image_size = infer_checkpoint_image_size(
        state_dict=state_dict,
        default_image_size=config.image_size,
        patch_size=get_square_patch_size(config),
    )

    config.num_labels = len(labels)
    config.id2label = {index: label for index, label in enumerate(labels)}
    config.label2id = {label: index for index, label in enumerate(labels)}
    config.image_size = checkpoint_image_size

    model = ViTForImageClassification(config)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model, checkpoint_image_size


def select_device() -> torch.device:
    """
    Select CUDA, Apple Metal, or CPU in that order.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_autocast_context(device: torch.device):
    """
    Enable mixed-precision inference on CUDA.

    MPS and CPU use full precision for compatibility.
    """
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def make_prediction_row(
    source: str,
    labels: list[str],
    probabilities: list[float | str],
    top_k: int,
) -> PredictionRow:
    row: PredictionRow = {
        "image_path": source,
    }

    for rank in range(1, top_k + 1):
        index = rank - 1
        row[f"label_{rank}"] = labels[index] if index < len(labels) else ""
        row[f"probability_{rank}"] = (
            probabilities[index] if index < len(probabilities) else ""
        )

    return row


def make_error_row(
    source: str,
    top_k: int,
) -> PredictionRow:
    """
    Create a row for an image that could not be loaded.
    """
    return make_prediction_row(
        source=source,
        labels=["Error"],
        probabilities=[""],
        top_k=top_k,
    )


def make_model_prediction_row(
    source: str,
    indices: torch.Tensor,
    scores: torch.Tensor,
    id2label: dict[int, str],
    top_k: int,
) -> PredictionRow:
    """
    Convert one model prediction into an output row.
    """
    return make_prediction_row(
        source=source,
        labels=[id2label[class_index] for class_index in indices.tolist()],
        probabilities=scores.tolist(),
        top_k=top_k,
    )


def predict(
    model: ViTForImageClassification,
    processor: ViTImageProcessor,
    dataset: ImageDataset,
    checkpoint_image_size: int,
    target_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> list[PredictionRow]:
    """
    Run batched image classification.

    Failed images receive label_1="Error". Valid and failed rows remain in
    the same order as the original input.
    """
    require_at_least("batch_size", batch_size, 1)
    require_at_least("top_k", top_k, 1)
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    device = select_device()

    model.to(device)
    model.eval()

    top_k = min(top_k, model.config.num_labels)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_images,
    )

    interpolate_position_embeddings = (
        target_size[0] != checkpoint_image_size
        or target_size[1] != checkpoint_image_size
    )

    results: list[PredictionRow] = []

    print(f"Device: {device}")
    print(f"Checkpoint image size: {checkpoint_image_size}")
    print(f"Inference image size: {target_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num CPU workers for data loading: {num_workers}")
    print(f"Top K predictions to retain: {top_k}")
    print("-" * TERMINAL_COLUMNS)

    with torch.inference_mode():
        for sources, images, errors in tqdm(
            data_loader,
            desc="Predicting",
            unit="batch",
        ):
            # Keep a slot for every original item so output order is preserved.
            batch_rows: list[PredictionRow | None] = [None] * len(sources)

            valid_positions: list[int] = []
            valid_sources: list[str] = []
            valid_images: list[Image.Image] = []

            for position, (source, image, error) in enumerate(
                zip(sources, images, errors)
            ):
                if image is None or error is not None:
                    tqdm.write(
                        f"Error loading image {source}: "
                        f"{error or 'Unknown image-loading error'}",
                        file=sys.stderr,
                    )
                    batch_rows[position] = make_error_row(
                        source=source,
                        top_k=top_k,
                    )
                    continue

                valid_positions.append(position)
                valid_sources.append(source)
                valid_images.append(image)

            # A batch may contain only failed images.
            if valid_images:
                processed = processor(
                    images=valid_images,
                    return_tensors="pt",
                )

                pixel_values = processed["pixel_values"].to(
                    device,
                    non_blocking=device.type == "cuda",
                )

                with make_autocast_context(device):
                    outputs = model(
                        pixel_values=pixel_values,
                        interpolate_pos_encoding=(interpolate_position_embeddings),
                    )

                probabilities = outputs.logits.float().softmax(dim=-1)

                top_probabilities, top_indices = probabilities.topk(
                    k=top_k,
                    dim=-1,
                )

                top_probabilities = top_probabilities.cpu()
                top_indices = top_indices.cpu()

                for position, source, indices, scores in zip(
                    valid_positions,
                    valid_sources,
                    top_indices,
                    top_probabilities,
                ):
                    batch_rows[position] = make_model_prediction_row(
                        source=source,
                        indices=indices,
                        scores=scores,
                        id2label=model.config.id2label,
                        top_k=top_k,
                    )

            # Every slot should now contain either a prediction or Error row.
            results.extend(row for row in batch_rows if row is not None)

    return results


def write_predictions(
    input_data: list[dict],
    results: list[PredictionRow],
    results_field: str,
    output_type: FileType,
    output_path: str | Path,
    top_k: int,
) -> None:
    """
    Add predictions to the original input records and save them as CSV
    or JSON.

    Adds:
        results_field:
            The highest-probability class, or "Error".

        f"{results_field}_full":
            A string containing all top-k classes and probabilities, for
            example:

                "plot: 0.9123 | table: 0.0612 | image: 0.0265"
    """
    if len(input_data) != len(results):
        raise ValueError(
            "The number of input records does not match the number of "
            f"prediction results: {len(input_data)} inputs versus "
            f"{len(results)} results."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_results_field = f"{results_field}_full"
    output_data: list[dict] = []

    for input_record, result in zip(input_data, results):
        output_record = dict(input_record)
        top_prediction = result.get("label_1", "Error")

        output_record[results_field] = top_prediction

        if top_prediction == "Error":
            full_predictions = "Error"
        else:
            full_predictions = format_full_predictions(result, top_k)

        output_record[full_results_field] = full_predictions
        output_data.append(output_record)

    output_df = pd.DataFrame(output_data)

    if output_type == "csv":
        output_df.to_csv(
            output_path,
            index=False,
        )
    else:
        output_df.to_json(
            output_path,
            orient="records",
            indent=2,
            force_ascii=False,
        )


def format_full_predictions(result: PredictionRow, top_k: int) -> str:
    predictions = []

    for rank in range(1, top_k + 1):
        label = result.get(f"label_{rank}")
        probability = result.get(f"probability_{rank}")

        if not label or probability in (None, ""):
            continue

        predictions.append(f"{label}: {float(probability):.4f}")

    return " | ".join(predictions)


def read_json_frame(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(input_path)
    except ValueError as error:
        if "If using all scalar values" not in str(error):
            raise

    return pd.read_json(input_path, typ="series").to_frame().T


def read_input_frame(input_path: Path) -> tuple[pd.DataFrame, FileType]:
    readers = {
        ".csv": (pd.read_csv, "csv"),
        ".json": (read_json_frame, "json"),
    }

    try:
        reader, file_type = readers[input_path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported input type {input_path.suffix.lower()!r}. "
            "Use a .csv or .json file."
        ) from None

    input_frame = reader(input_path)
    print(f"Input data from {input_path} loaded as {file_type.upper()}.")
    return input_frame, file_type


def load_data(
    input_file: str | Path,
    image_field: str,
) -> tuple[list[dict], list[str], FileType]:
    """
    Load image paths/URLs from a CSV or JSON file.
    """
    input_path = Path(input_file)
    input_frame, file_type = read_input_frame(input_path)

    if image_field not in input_frame.columns:
        raise KeyError(
            f"Image field {image_field!r} was not found. "
            f"Available fields: {list(input_frame.columns)}"
        )

    # Missing values become an empty source and will be emitted as Error rows.
    return (
        input_frame.to_dict(orient="records"),
        input_frame[image_field].fillna("").astype(str).tolist(),
        file_type,
    )


def default_output_path(input_file: str | Path) -> Path:
    input_path = Path(input_file)
    return input_path.with_name(f"{input_path.stem}_with-preds{input_path.suffix}")


def print_prediction_summary(predictions: list[PredictionRow]) -> None:
    labels = [result.get("label_1", "Error") for result in predictions]
    total = len(labels)
    count = Counter(labels)

    print("-" * TERMINAL_COLUMNS)
    print(f"Summary of predictions (Total = {total} figures)")

    for label in NEW_LABELS:
        value = count[label] if label in count else 0
        print(f"{label}: {value} ({100 * value / total:.2f}%)")
    print("-" * TERMINAL_COLUMNS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched ViT image classification on local "
            "images and remote image URLs."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="CSV or JSON input file.",
    )
    parser.add_argument(
        "--image-field",
        required=True,
        help="Field containing the image URL or local path.",
    )
    parser.add_argument(
        "--prediction-field",
        default="inferred_type",
        help="Output field for the top predicted figure type.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the fine-tuned .safetensors file.",
    )
    parser.add_argument(
        "--output",
        help="Output path. Defaults to <input>_with-preds.<ext>.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Number of parallel image-loading workers. "
            "Use 0 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Square inference image size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of predictions to save per image.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_data, images, file_type = load_data(
        input_file=args.input_file,
        image_field=args.image_field,
    )
    print(f"Total number of figures to identify: {len(images)}")

    require_at_least("image_size", args.image_size, 1)
    target_size = (
        args.image_size,
        args.image_size,
    )

    model, checkpoint_image_size = load_model(
        weights_path=args.weights,
        labels=NEW_LABELS,
    )

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID,
        size={
            "height": target_size[0],
            "width": target_size[1],
        },
    )

    dataset = ImageDataset(images)
    top_k = min(args.top_k, len(NEW_LABELS))

    results = predict(
        model=model,
        processor=processor,
        dataset=dataset,
        checkpoint_image_size=checkpoint_image_size,
        target_size=target_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=top_k,
    )
    print_prediction_summary(results)

    output_path = (
        Path(args.output) if args.output else default_output_path(args.input_file)
    )

    write_predictions(
        input_data=input_data,
        results=results,
        results_field=args.prediction_field,
        output_type=file_type,
        output_path=output_path,
        top_k=top_k,
    )

    error_count = sum(row.get("label_1") == "Error" for row in results)

    print(f"Processed rows: {len(results)}")
    print(f"Image-loading errors: {error_count}")
    print(f"Predictions saved to: {output_path}")
    print("-" * TERMINAL_COLUMNS)


if __name__ == "__main__":
    main()
