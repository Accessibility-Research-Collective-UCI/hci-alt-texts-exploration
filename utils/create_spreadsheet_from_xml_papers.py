"""
This script converts JSON files from scraped XML papers into a CSV format for easier browsing and analysis.

The JSON structure is:
[
    {
        "title": str,
        "doi": str,
        "venue": str,
        "year": int,
        "figures": [
            {
                "figure_num": int | null,
                "img_src": str,
                "caption": str,
                "alt_text": str,
                "referring_text": [str]
            },
        ]
    },
]

Usage:
# with uploading images
uv run utils/create_spreadsheet_from_xml_papers.py --input-dir xml_papers/ --image-base-url . --output-dir processed_papers/spreadsheets --upload-batch-size 256 --concurrency 16

# specify venue to include
uv run utils/create_spreadsheet_from_xml_papers.py --input-dir xml_papers/ --image-base-url . --include-venues assets --output-dir processed_papers/spreadsheets --upload-batch-size 256 --concurrency 16

# run without uploading images
uv run utils/create_spreadsheet_from_xml_papers.py --input-dir xml_papers/ --image-base-url . --output-dir processed_papers/spreadsheets --skip-upload
"""

import argparse
import json
import os
import re
import shutil
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from s3_image_uploader import S3ImageUploader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

TERMINAL_COLUMNS, _ = shutil.get_terminal_size()


@dataclass
class InputXMLPaper:
    title: str
    doi: str
    venue: str
    year: int
    figures: list[InputXMLFigure]


@dataclass
class InputXMLFigure:
    figure_num: int | None
    img_src: Path
    caption: str
    alt_text: str
    referring_text: list[str]


@dataclass
class ImageUploadResult:
    index: int
    image_url: str | None
    error: str | None = None


@dataclass
class OutputSpreadsheetRow:
    title: str
    doi: str
    paper_link: str
    venue: str
    year: int
    img_url: str | None
    figure_num: int | None
    caption: str
    alt_text: str
    jaccard_similarity: float
    cosine_similarity: float
    referring_text: str
    figure_preview: str = ""
    figure_type: str = ""
    example_qualty: str = ""


def read_json_file(file_path: Path) -> list[InputXMLPaper]:
    """
    Reads a JSON file and returns a list of InputXMLPaper objects.

    Args:
        file_path (Path): The path to the JSON file.

    Returns:
        list[InputXMLPaper]: A list of InputXMLPaper objects.
    """
    output = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for paper in data:
            figures = [
                InputXMLFigure(
                    figure_num=figure.get("figure_num", None),
                    alt_text=figure.get("alt_text", ""),
                    img_src=Path(figure.get("img_src", "")),
                    caption=figure.get("caption", ""),
                    referring_text=figure.get("referring_text", []),
                )
                for figure in paper.get("figures", [])
            ]
            output.append(
                InputXMLPaper(
                    title=paper.get("title", ""),
                    doi=paper.get("doi", ""),
                    venue=paper.get("venue", ""),
                    year=paper.get("year", 0),
                    figures=figures,
                )
            )
    return output


def clean_text_for_similarity_comparison(text: str) -> str:
    """
    Removes standard prefixes, punctuation, and converts text to lowercase. This is done to make comparing similarity between figure captions and alt text is more reliable.

    Args:
        text (str): Input string to clean

    Returns
        (str): Cleaned string.
    """
    text = text.lower()

    # Your updated pattern (added re.IGNORECASE to be safe with mixed casing)
    pattern = r"^(figure \d+:?|fig\.? \d+:?|table \d+:?|image of|photo of)\s*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Strip punctuation
    return re.sub(r"[^\w\s]", "", text)


def get_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculates unique word overlap percentage (0.0 to 1.0).

    Args:
        text1 (str): First cleaned text string.
        text2 (str): Second cleaned text string.

    Returns:
        (float): Jaccard similarity of the two strings, between 0.0 and 1.0.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def get_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculates word frequency similarity by creating a word count vector and computing the cosine similarity between them (0.0 to 1.0).

    Args:
        text1 (str): First cleaned text string.
        text2 (str): Second cleaned text string.

    Returns:
        (float): Cosine similarity of the two strings, between 0.0 and 1.0.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def convert_to_spreadsheet_format(
    data: list[OutputSpreadsheetRow], venue: str, years: list[int], output_dir: str
) -> None:
    """
    Converts a list of OutputSpreadsheetRow objects into a list of dictionaries suitable for CSV output.

    Args:
        data (list[OutputSpreadsheetRow]): The input data to convert.
        venue (str): The venue of the papers.
        years (list[int]): The years of the papers.
        output_dir (str): The directory where the CSV file will be saved.
    """
    output_path = os.path.join(
        output_dir, f"{venue}_{'-'.join(str(year) for year in years)}_papers.csv"
    )
    df = pd.DataFrame([row.__dict__ for row in data])
    df.insert(0, "id", range(1, len(df) + 1))
    column_order = [
        "id",
        "title",
        "doi",
        "paper_link",
        "venue",
        "year",
        "img_url",
        "figure_preview",
        "figure_num",
        "caption",
        "alt_text",
        "jaccard_similarity",
        "cosine_similarity",
        "referring_text",
        "figure_type",
        "example_qualty",
    ]
    df = df[column_order]

    df.to_csv(output_path, index=False)
    print(
        f"Papers for {venue} {', '.join(str(year) for year in years)}: CSV file saved to {output_path}"
    )


def collect_venue_year_jsons(
    folder_path: Path,
) -> dict[str, list[tuple[int, Path]]]:
    """
    Collects all JSON files in the given folder and extracts the venue and year from their filenames.

    Args:
        folder_path (Path): The path to the folder containing JSON files.

    Returns:
        list[tuple[str, int, Path]]: A list of tuples containing the venue, year, and path to each JSON file.
    """
    venue_year_jsons = {}
    for json_file in folder_path.glob("*.json"):
        parsed_filepath = json_file.stem.split("_")
        venue = parsed_filepath[0]
        year = parsed_filepath[1]

        if venue in venue_year_jsons:
            venue_year_jsons[venue].append((int(year), json_file))
        else:
            venue_year_jsons[venue] = [(int(year), json_file)]

    for venue in venue_year_jsons:
        venue_year_jsons[venue].sort(key=lambda x: x[0])
    return venue_year_jsons


def format_data(
    data: list[InputXMLPaper],
) -> list[OutputSpreadsheetRow]:
    """
    Formats the parsed XML paper data into a spreadsheet-friendly format and uploads images to S3.

    Args:
        data (list[InputXMLPaper]): The parsed XML paper data.
    Returns:
        list[OutputSpreadsheetRow]: The formatted data ready for CSV output.
    """
    output_data: list[OutputSpreadsheetRow] = []
    for paper in data:
        for figure in paper.figures:
            # compute similarity between caption and alt text
            caption_cleaned = clean_text_for_similarity_comparison(figure.caption)
            alt_text_cleaned = clean_text_for_similarity_comparison(figure.alt_text)
            jaccard_similarity = get_jaccard_similarity(
                caption_cleaned, alt_text_cleaned
            )
            cosine_similarity = get_cosine_similarity(caption_cleaned, alt_text_cleaned)

            dl_acm_paper_link = (
                f"https://dl.acm.org/doi/{paper.doi}" if paper.doi != "" else ""
            )
            image_url = str(figure.img_src) if figure.img_src else None
            referring_text = f"\n{'-' * 50}\n".join(figure.referring_text)

            output_data.append(
                OutputSpreadsheetRow(
                    title=paper.title,
                    doi=paper.doi,
                    paper_link=dl_acm_paper_link,
                    venue=paper.venue,
                    year=paper.year,
                    img_url=image_url,
                    figure_num=figure.figure_num,
                    caption=figure.caption,
                    alt_text=figure.alt_text,
                    jaccard_similarity=jaccard_similarity,
                    cosine_similarity=cosine_similarity,
                    referring_text=referring_text,
                )
            )
    return output_data


def _upload_image(
    index: int,
    image_path: str,
    s3_uploader: S3ImageUploader,
    sub_bucket: str,
) -> ImageUploadResult:
    image_url = s3_uploader.upload_image(
        image_path,
        sub_bucket=sub_bucket,
    )
    return ImageUploadResult(index=index, image_url=image_url)


def _batch_items(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def upload_images(
    data: list[OutputSpreadsheetRow],
    image_base_url: str,
    s3_uploader: S3ImageUploader,
    bucket_prefix: str = "hci-alt-text",
    upload_batch_size: int = 64,
    max_workers: int = 16,
    allow_partial_uploads: bool = True,
) -> list[OutputSpreadsheetRow]:
    """
    Uploads images for the given paper data to S3, returning the updated, formatted data with S3 URLs.

    Args:
        data (list[OutputSpreadsheetRow]): Formatted paper data to upload images for.
        image_base_url (str): Local folder where images are stored. This is used to construct the full path to the image for uploading.
        bucket_prefix (str, optional): Prefix for the S3 bucket where images will be uploaded. Defaults to "hci-alt-text".
        upload_batch_size (int, optional): Number of images to upload in a single batch. Defaults to 64.
        max_workers (int, optional): Maximum number of worker threads for uploading images. Defaults to 16.
        allow_partial_uploads (bool, optional): Whether to allow partial uploads in case of errors. Defaults to True.
        s3_uploader (S3ImageUploader | None, optional): The S3 image uploader instance. Defaults to None.

    Returns:
        list[OutputSpreadsheetRow]: _description_
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if upload_batch_size < 1:
        raise ValueError("upload_batch_size must be at least 1")

    upload_results: list[ImageUploadResult | None] = [None] * len(data)
    upload_tasks: list[tuple[int, str, str, int]] = [
        (idx, os.path.join(image_base_url, row.img_url), row.venue, row.year)
        for idx, row in enumerate(data)
        if row.img_url
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(upload_tasks), desc="Uploading images to S3") as progress:
            for batch in _batch_items(upload_tasks, upload_batch_size):
                futures = {
                    executor.submit(
                        _upload_image,
                        idx,
                        image_path,
                        s3_uploader,
                        f"{bucket_prefix}/{venue}_{year}",
                    ): (idx, image_path, venue, year)
                    for idx, image_path, venue, year in batch
                }
                for future in as_completed(futures):
                    idx, _, _, _ = futures[future]
                    try:
                        s3_url = future.result().image_url
                        upload_results[idx] = ImageUploadResult(
                            index=idx, image_url=s3_url, error=None
                        )
                    except Exception as e:
                        if not allow_partial_uploads:
                            raise
                        upload_results[idx] = ImageUploadResult(
                            index=idx, image_url=None, error=str(e)
                        )
                    progress.update(1)

    # update the original data with the S3 URLs
    for result in upload_results:
        if result is not None:
            data[result.index].img_url = result.image_url

    return data


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON files from scraped XML papers into a CSV format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JSON files to process. Each file should be named VENUE_YEAR_extracted_info.json",
    )
    parser.add_argument(
        "--image-base-url",
        type=str,
        required=True,
        help="Base URL for images to be included in the CSV.",
    )
    parser.add_argument(
        "--include-venues",
        nargs="+",
        default=["all"],
        help="Venues to process. If none are provided, all venues found in the input directory (specified by the JSON file names) will be processed.",
    )
    parser.add_argument(
        "--bucket-prefix",
        type=str,
        default="hci-alt-text",
        help="Prefix for the S3 bucket where images will be uploaded. Default is 'hci-alt-text'.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Skip the uploading of images to S3. This will provide a blank URL instead/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the CSV files will be saved. If none is provided, the script will save in the same directory as the input files.",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=64,
        help="Number of images to upload in a single batch. Default is 64.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Maximum number of worker threads for uploading images. Default is 16.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    folder_path = args.input_dir
    if not folder_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {folder_path}")
    venues_to_include = set([venue.upper() for venue in args.include_venues])

    print(f"Processing JSON files in {folder_path}")
    print(f"Venues to be processed: {', '.join(venues_to_include)}")
    print(f"Images will be uploaded from local URL: {args.image_base_url}")
    print(f"Images will be uploaded to S3 bucket prefix: {args.bucket_prefix}")
    print(f"CSV files will be saved to: {args.output_dir or folder_path}")
    print(
        f"Using {args.concurrency} threads with batch size {args.upload_batch_size} for uploading images.",
        end="\n" + "-" * TERMINAL_COLUMNS + "\n",
    )

    s3_uploader = S3ImageUploader()
    parsed_json_info = collect_venue_year_jsons(folder_path)
    for venue in parsed_json_info:
        if venue.upper() not in venues_to_include and "ALL" not in venues_to_include:
            print(
                f"Venue {venue} was not included in --include-venues ({', '.join(venues_to_include)}). Skipping...",
                end="\n" + "-" * TERMINAL_COLUMNS + "\n",
            )
            continue

        years = [year for year, _ in parsed_json_info[venue]]
        print(f"Processing {venue} {','.join(str(y) for y in years)}")

        # TODO: add an option for if files should be combined. if not, save separately
        formatted_data_for_venue: list[OutputSpreadsheetRow] = []
        for year, json_file in parsed_json_info[venue]:
            data: list[InputXMLPaper] = read_json_file(json_file)
            curr_formatted_data: list[OutputSpreadsheetRow] = format_data(data)
            print(
                f"---{year}: Number of papers {len(data)} | Number of figures: {len(curr_formatted_data)}"
            )
            formatted_data_for_venue.extend(curr_formatted_data)
        print(f"Total number of figures for {venue}: {len(formatted_data_for_venue)}.")

        if args.skip_upload:
            output_data_for_venue = formatted_data_for_venue
        else:
            output_data_for_venue = upload_images(
                formatted_data_for_venue,
                image_base_url=args.image_base_url,
                s3_uploader=s3_uploader,
                bucket_prefix=f"{args.bucket_prefix}",
                upload_batch_size=args.upload_batch_size,
                max_workers=args.concurrency,
            )

        # TODO: allow for saving as JSON
        # save output
        convert_to_spreadsheet_format(
            output_data_for_venue,
            venue=venue,
            years=years,
            output_dir=args.output_dir or folder_path,
        )
        print(
            f"Finished processing {venue} {', '.join(str(y) for y in years)}.",
            end="\n" + "-" * TERMINAL_COLUMNS + "\n",
        )


if __name__ == "__main__":
    main()
