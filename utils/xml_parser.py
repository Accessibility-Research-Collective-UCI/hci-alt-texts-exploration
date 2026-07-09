"""
XML parser for extracting paper metadata and figure information.

Usage:
uv run utils/xml_parser.py --folder xml_papers/ --output-dir data/
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

VENUE_YEAR_DIR_PATTERN = re.compile(r"^[A-Za-z]+_\d{4}$")
TITLE_DOI_FILENAME_PATTERN = re.compile(
    r"^(?P<title>.+?) \[(?P<doi_prefix>10\.\d{4,9})_(?P<doi_suffix>[-._;()/:A-Za-z0-9]+)\]\.xml$"
)
FIGURE_NUM_PATTERN = re.compile(r"\bFigure\s+(\d+)\s*:", re.IGNORECASE)
FIGURE_REF_PATTERN_TEMPLATE = r"\b(?:fig(?:ure)?\.?)\s*{figure_num}\b"
RAW_FIGURE_ALT_PATTERN = re.compile(
    r"""
    <Figure\b
    [^>]*?
    \bAlt="
    (?P<alt>.*?)
    (?=
        "\s+(?:[A-Za-z_:][\w:.-]*)=
        |
        "\s*/?>
    )
    """,
    re.DOTALL | re.VERBOSE,
)
EXCLUDED_ALT_TEXTS = {
    "cc-by logo",
    "cc-by-nc logo",
    "cc-by-nc-nd logo",
    "cc-by-nd logo",
}


@dataclass
class FigureData:
    figure_num: int | None
    alt_text: str
    img_src: str
    caption: str
    referring_text: list[str]


class PaperParser:
    """Extract paper metadata and figure alt text from one XML paper."""

    def __init__(
        self,
        xml_path: Path,
        title: str = "",
        doi: str = "",
        venue: str = "",
        year: int = 0,
    ):
        self.contents = xml_path.read_text(encoding="utf-8", errors="replace")
        self.soup: BeautifulSoup = BeautifulSoup(self.contents, "xml")

        if title == "" and doi == "":
            self.title, self.doi = self.extract_title_and_doi_from_filename(
                xml_path.name
            )
        else:
            self.title: str = title
            self.doi: str = doi

        self.venue: str = venue
        self.year: int = year
        self.figures: list[FigureData] = []

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse whitespace and strip leading/trailing space."""
        return " ".join(text.split())

    @staticmethod
    def get_text_or_empty(tag) -> str:
        """Safely extract normalized text from a BeautifulSoup tag."""
        return PaperParser.normalize_text(tag.get_text()) if tag is not None else ""

    @staticmethod
    def get_attr_or_empty(tag, attr: str) -> str:
        """Safely extract and strip an attribute from a BeautifulSoup tag."""
        return tag.get(attr, "").strip() if tag is not None else ""

    @staticmethod
    def extract_figure_num(text: str) -> int | None:
        """Extract figure number from text like 'Figure 4:'."""
        match = FIGURE_NUM_PATTERN.search(text)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_title_and_doi_from_filename(filename: str) -> tuple[str, str]:
        match = TITLE_DOI_FILENAME_PATTERN.fullmatch(filename)

        if not match:
            return "", ""

        title = match.group("title").strip().replace("_", ":")
        doi = f"{match.group('doi_prefix')}/{match.group('doi_suffix')}"

        return title, doi

    @staticmethod
    def paragraph_mentions_figure(paragraph_text: str, figure_num: int) -> bool:
        """
        Match references such as:
        - Figure 4
        - Fig 4
        - Fig. 4

        Uses a trailing word boundary so Figure 4 does not match Figure 40.
        """
        pattern = re.compile(
            FIGURE_REF_PATTERN_TEMPLATE.format(figure_num=figure_num),
            re.IGNORECASE,
        )
        return pattern.search(paragraph_text) is not None

    def find_referring_text(
        self,
        paragraphs: list[str],
        figure_num: int,
        alt_text: str,
        caption: str,
    ) -> list[str]:
        """Find paragraphs that refer to a figure, excluding its alt text and caption."""
        referring_text = []
        excluded_text = {alt_text, caption}

        for paragraph_text in paragraphs:
            if paragraph_text in excluded_text:
                continue

            if self.paragraph_mentions_figure(paragraph_text, figure_num):
                referring_text.append(paragraph_text)

        return referring_text

    def _extract_paper_info(self) -> None:
        if self.title == "":
            title_node = self.soup.find("dc:title")
            if title_node is not None:
                self.title = self.get_text_or_empty(
                    title_node.find(attrs={"xml:lang": "en"})
                )

        if self.doi == "":
            self.doi = self.get_text_or_empty(self.soup.find("prism:doi"))

    @staticmethod
    def extract_figure_alts_from_raw_xml(xml_text: str) -> list[str]:
        return [
            " ".join(match.group("alt").split())
            for match in RAW_FIGURE_ALT_PATTERN.finditer(xml_text)
        ]

    def _extract_single_figure(
        self,
        figure,
        idx: int,
        raw_figure_alts: list[str],
        paragraphs: list[str],
        parent_path: Path | None,
    ) -> FigureData | None:
        image_data = figure.find_next("ImageData")
        caption_tag = figure.find_next("P")

        alt_text = (
            raw_figure_alts[idx]
            if idx < len(raw_figure_alts)
            else self.get_attr_or_empty(figure, "Alt")
        )

        if alt_text.lower() in EXCLUDED_ALT_TEXTS:
            return None

        caption = self.get_text_or_empty(caption_tag)
        figure_num = self.extract_figure_num(caption)

        img_path = self.get_attr_or_empty(image_data, "src")
        img_src = str(parent_path / img_path) if img_path and parent_path else ""

        figure_info = FigureData(
            figure_num=figure_num,
            alt_text=alt_text,
            img_src=img_src,
            caption=caption,
            referring_text=[],
        )

        if figure_num is not None:
            figure_info.referring_text = self.find_referring_text(
                paragraphs=paragraphs,
                figure_num=figure_num,
                alt_text=alt_text,
                caption=caption,
            )

        return figure_info

    def _extract_figure_info(self, parent_path: Path | None) -> None:
        self.figures = []
        raw_figure_alts = self.extract_figure_alts_from_raw_xml(self.contents)
        figures = self.soup.find_all("Figure")
        paragraphs = [self.get_text_or_empty(p) for p in self.soup.find_all("P")]

        for idx, figure in enumerate(figures):
            figure_info = self._extract_single_figure(
                figure=figure,
                idx=idx,
                raw_figure_alts=raw_figure_alts,
                paragraphs=paragraphs,
                parent_path=parent_path,
            )

            if figure_info is not None:
                self.figures.append(figure_info)

    def extract(self, parent_path: Path | None = None) -> dict:
        self._extract_paper_info()
        self._extract_figure_info(parent_path)

        return {
            "title": self.title,
            "doi": self.doi,
            "venue": self.venue,
            "year": self.year,
            "figures": [asdict(figure) for figure in self.figures],
        }


def parse_venue_year_dir(directory: Path) -> tuple[str, int, Path]:
    venue, year = directory.name.split("_")
    return venue, int(year), directory


def collect_venue_year_dirs(folder_path: Path) -> list[tuple[str, int, Path]]:
    sub_dirs = [d for d in folder_path.iterdir() if d.is_dir()]

    if not all(VENUE_YEAR_DIR_PATTERN.match(d.name) for d in sub_dirs):
        raise ValueError(
            "All sub-directories must follow the pattern 'Venue_Year' (e.g., 'ASSETS_2025')."
        )

    venue_year_dir_list = sorted([parse_venue_year_dir(d) for d in sub_dirs])
    return venue_year_dir_list


def process_venue_year_dir(venue: str, year: int, sub_dir: Path) -> list[dict]:
    curr_output = []
    xml_files = list(sub_dir.glob("*.xml"))

    for xml_file in tqdm(xml_files, desc=f"Processing {venue} {year} in {sub_dir}"):
        try:
            parser = PaperParser(xml_file, venue=venue, year=year)
            extracted_info = parser.extract(parent_path=sub_dir)
            curr_output.append(extracted_info)
        except Exception as e:
            print(f"Error processing {venue} {year}, {xml_file}: {e}")

    print(
        f"Finished processing {venue} {year}. Extracted info for {len(curr_output)} papers.",
        end="\n" + "-" * 100 + "\n",
    )
    return curr_output


def write_extracted_info(
    folder_path: Path, venue: str, year: int, extracted_info: list[dict]
) -> None:
    output_path = folder_path / f"{venue}_{year}_extracted_info.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(extracted_info, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse directory for XML file for paper info."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Path to folder that contains sub-folders with XML files.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        default=None,
        help="Optional path to output JSON file. If not provided, will save in the same folder.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    folder_path = args.folder
    if not folder_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {folder_path}")

    for venue, year, sub_dir in collect_venue_year_dirs(folder_path):
        curr_output = process_venue_year_dir(venue, year, sub_dir)
        write_extracted_info(args.output_dir or folder_path, venue, year, curr_output)


if __name__ == "__main__":
    main()
