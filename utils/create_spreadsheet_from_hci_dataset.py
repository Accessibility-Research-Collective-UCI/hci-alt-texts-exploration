"""
This script is used to create a CSV to easily browse the HCI alt text dataset.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic import BaseModel
from s3_image_uploader import S3ImageUploader
from tqdm.auto import tqdm


class AltTextLevels(str, Enum):
    INVALID = "invalid"
    NONE = "none"
    LOGISTICS = "logistics"
    STATISTICS = "statistics"
    TRENDS = "trends"
    SEMANTICS = "semantics"


_INT_TO_LEVELS = {
    -1: AltTextLevels.INVALID,
    0: AltTextLevels.NONE,
    1: AltTextLevels.LOGISTICS,
    2: AltTextLevels.STATISTICS,
    3: AltTextLevels.TRENDS,
    4: AltTextLevels.SEMANTICS,
}


class HCIAltTextRaw(BaseModel):
    title: str
    pdf_hash: str
    venue: str
    year: int
    alt_text: str
    levels: list[list[AltTextLevels]]
    corpus_id: int
    sentences: list[str]
    caption: str
    local_uri: list[str]
    is_plot: bool = False
    annotated: bool
    compound: bool


class HCIAltText(BaseModel):
    corpus_id: int
    title: str
    pdf_hash: str
    image_url: str | None
    image_upload_error: str | None = None
    venue: str
    year: int
    caption: str
    alt_text: str
    sentences: str
    levels: list[list[str]]
    is_plot: bool = False
    annotated: bool
    compound: bool


@dataclass
class ImageUploadResult:
    index: int
    image_url: str | None
    error: str | None = None


def read_jsonl(file_path: str) -> list[HCIAltTextRaw]:
    """
    Loads ASSETS 2022 HCI alt text data into a pydantic model.

    Args:
        file_path (str): The path to the JSONL file containing the data.

    Returns:
        list[HCIAltTextRaw]: A list of HCIAltTextRaw objects.
    """
    output: list[HCIAltTextRaw] = []
    with open(file_path, "r") as f:
        for line in f:
            curr = json.loads(line)
            if curr["levels"] is None:
                curr["levels"] = []
            else:
                for sent_index, levels in enumerate(curr["levels"]):
                    curr["levels"][sent_index] = [
                        _INT_TO_LEVELS[level] for level in levels
                    ]
            output.append(HCIAltTextRaw(**curr))
    return output


def _batch_items(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _upload_image(
    index: int,
    image_path: str,
    s3_uploader: S3ImageUploader,
    sub_bucket: str = "hci-alt-text-assets2022",
) -> ImageUploadResult:
    image_url = s3_uploader.upload_image(
        image_path,
        sub_bucket=sub_bucket,
    )
    return ImageUploadResult(index=index, image_url=image_url)


def _format_item(
    item: HCIAltTextRaw,
    upload_result: ImageUploadResult,
) -> HCIAltText:
    sentences_with_levels = [
        (sentence, "; ".join([level.value for level in item.levels[i]]))
        if item.levels
        else (sentence, None)
        for i, sentence in enumerate(item.sentences)
    ]
    return HCIAltText(
        corpus_id=item.corpus_id,
        title=item.title,
        pdf_hash=item.pdf_hash,
        image_url=upload_result.image_url,
        image_upload_error=upload_result.error,
        venue=item.venue,
        year=item.year,
        caption=item.caption,
        alt_text=item.alt_text,
        sentences=f"\n{'-' * 100}\n".join(
            f"(Levels {levels}): {sentence} "
            for sentence, levels in sentences_with_levels
        ),
        levels=[[level.value for level in sent_levels] for sent_levels in item.levels],
        is_plot=item.is_plot,
        annotated=item.annotated,
        compound=item.compound,
    )


def format_data(
    data: list[HCIAltTextRaw],
    image_base_url: str,
    max_workers: int = 16,
    upload_batch_size: int = 64,
    allow_partial_uploads: bool = True,
    s3_uploader: S3ImageUploader | None = None,
) -> list[HCIAltText]:
    """
    Formats the raw HCI alt text data into a CSV-friendly format.

    Args:
        data (list[HCIAltTextRaw]): A list of raw HCI alt text data.
        image_base_url (str): The base URL for the images.
        max_workers (int): The maximum number of parallel upload workers.
        upload_batch_size (int): The number of uploads to schedule at a time.
        allow_partial_uploads (bool): Whether to keep rows whose image upload fails.
        s3_uploader (S3ImageUploader | None): Optional uploader to use for uploads.

    Returns:
        list[HCIAltText]: A list of formatted HCI alt text data.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if upload_batch_size < 1:
        raise ValueError("upload_batch_size must be at least 1")

    s3_uploader = s3_uploader or S3ImageUploader()
    upload_results: list[ImageUploadResult | None] = [None] * len(data)
    upload_tasks = [
        (index, os.path.join(image_base_url, item.local_uri[0]))
        for index, item in enumerate(data)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(upload_tasks), desc="Uploading images") as progress:
            for batch in _batch_items(upload_tasks, upload_batch_size):
                futures = {
                    executor.submit(
                        _upload_image,
                        index,
                        image_path,
                        s3_uploader,
                    ): (index, image_path)
                    for index, image_path in batch
                }
                for future in as_completed(futures):
                    index, image_path = futures[future]
                    try:
                        upload_results[index] = future.result()
                    except Exception as exc:
                        if not allow_partial_uploads:
                            raise
                        upload_results[index] = ImageUploadResult(
                            index=index,
                            image_url=None,
                            error=f"{type(exc).__name__}: {exc} ({image_path})",
                        )
                    progress.update(1)

    return [
        _format_item(
            item,
            upload_results[index]
            or ImageUploadResult(
                index=index,
                image_url=None,
                error="Upload did not complete",
            ),
        )
        for index, item in enumerate(data)
    ]


def convert_to_spreadsheet_format(data: list[HCIAltText], output_file: str):
    """
    Converts the formatted HCI alt text data into a CSV file.

    Args:
        data (list[HCIAltText]): A list of formatted HCI alt text data.
        output_file (str): The path to the output CSV file.
    """
    df = pd.DataFrame([item.model_dump() for item in data])
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    data = read_jsonl("./data/hci-alt-text-dataset-20220915.jsonl")
    data = format_data(data, "./data/images")

    # output spreadsheet
    os.makedirs("./data/formatted", exist_ok=True)
    output_path = "./data/formatted/hci-alt-text-dataset-20220915.csv"
    convert_to_spreadsheet_format(data, output_path)
