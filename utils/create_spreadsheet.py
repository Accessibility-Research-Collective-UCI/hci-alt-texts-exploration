"""
This script is used to create a CSV to easily browse the HCI alt text dataset.
"""

import json
import os
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
    image_url: str
    venue: str
    year: int
    caption: str
    alt_text: str
    sentences: str
    levels: list[list[str]]
    is_plot: bool = False
    annotated: bool
    compound: bool


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


def format_data(data: list[HCIAltTextRaw], image_base_url: str) -> list[HCIAltText]:
    """
    Formats the raw HCI alt text data into a CSV-friendly format.

    Args:
        data (list[HCIAltTextRaw]): A list of raw HCI alt text data.
        image_base_url (str): The base URL for the images.

    Returns:
        list[HCIAltText]: A list of formatted HCI alt text data.
    """
    s3_uploader = S3ImageUploader()
    output: list[HCIAltText] = []
    for item in tqdm(data):
        image_url = s3_uploader.upload_image(f"{image_base_url}/{item.local_uri[0]}")
        sentences_with_levels = [
            (sentence, "; ".join([level.value for level in item.levels[i]]))
            if item.levels
            else (sentence, None)
            for i, sentence in enumerate(item.sentences)
        ]
        output.append(
            HCIAltText(
                corpus_id=item.corpus_id,
                title=item.title,
                pdf_hash=item.pdf_hash,
                image_url=image_url,
                venue=item.venue,
                year=item.year,
                caption=item.caption,
                alt_text=item.alt_text,
                sentences=f"\n{'-' * 100}\n".join(
                    f"(Levels {levels}): {sentence} "
                    for sentence, levels in sentences_with_levels
                ),
                levels=[
                    [level.value for level in sent_levels]
                    for sent_levels in item.levels
                ],
                is_plot=item.is_plot,
                annotated=item.annotated,
                compound=item.compound,
            )
        )
    return output


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
