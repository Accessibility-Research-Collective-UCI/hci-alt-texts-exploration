"""
XML parser for extracting paper metadata and figure information.

The source XML may contain malformed Figure Alt attributes, including:
- unescaped double quotes inside Alt="..."
- bare ampersands inside attributes or text

Before parsing, Figure Alt values are replaced with safe placeholders and
stored separately. Bare ampersands in the remaining XML are then escaped.

Usage:
    uv run utils/xml_parser.py --folder xml_papers/ --output-dir xml_papers/
"""

import argparse
import html
import json
import re
import shutil
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import wordninja
from bs4 import BeautifulSoup, NavigableString, Tag
from ftfy import fix_text
from tqdm.auto import tqdm

TERMINAL_COLUMNS, _ = shutil.get_terminal_size()

BARE_AMPERSAND_PATTERN = re.compile(
    r"&(?!(?:amp|lt|gt|quot|apos);|#\d+;|#x[0-9A-Fa-f]+;)"
)
VENUE_YEAR_DIR_PATTERN = re.compile(r"^[A-Za-z]+_\d{4}$")
TITLE_DOI_FILENAME_PATTERN = re.compile(
    r"^(?P<title>.+?) "
    r"\[(?P<doi_prefix>10\.\d{4,9})_"
    r"(?P<doi_suffix>[-._;()/:A-Za-z0-9]+)\]\.xml$"
)
FIGURE_NUM_PATTERN = re.compile(
    r"\b(?:Fig(?:ure)?|Tab(?:le)?)\.?\s*(\d+)\s*[:.]",
    re.IGNORECASE,
)
FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Fig(?:ure)?|Tab(?:le)?)\.?\s*(?P<number>\d+)\s*[:.]",
    re.IGNORECASE,
)
FIGURE_REF_PATTERN_TEMPLATE = (
    r"\b(?:fig(?:ure)?\.?)\s*"
    r"{figure_num}"
    r"(?:\s*[a-z])?"
    r"(?=\W|$)"
)
SUBFIGURE_CAPTION_PATTERN = re.compile(
    r"\((?P<label>[A-Za-z])\)\s*(?P<text>.+)$",
    re.DOTALL,
)
FIGURE_NUMBER_BOUNDARY_PATTERN = re.compile(
    r"\b(Fig(?:ure)?|Tab(?:le)?)(\d+)\b",
    re.IGNORECASE,
)
CAMEL_CASE_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")
COMPRESSED_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")
PUNCTUATION_BOUNDARY_PATTERN = re.compile(r"(?<=[,:;.!?])(?=[A-Za-z])")
COMMON_JOINED_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
INLINE_FIGURE_ANCESTORS = {"P", "Lbl", "LBody", "LI"}

# Capture a complete Figure Alt value from the raw XML. Internal double quotes
# are allowed; a quote only ends the Alt value when it is followed by another
# XML attribute or by the end of the Figure opening tag.
FIGURE_ALT_ATTRIBUTE_PATTERN = re.compile(
    r"""
    (?P<prefix>
        <Figure\b
        [^>]*?
        \bAlt="
    )
    (?P<alt>.*?)
    (?=
        "\s+(?:[A-Za-z_:][\w:.-]*)=
        |
        "\s*/?>
    )
    """,
    re.DOTALL | re.VERBOSE,
)

FIGURE_ALT_PLACEHOLDER_PREFIX = "__FIGURE_ALT_"

EXCLUDE_CC_LICENSE_ALT_TEXT_REGEX = re.compile(
    r"""
    ^
    \s*
    (?:cc|bb)
    (?:
        # License terms, optionally followed by "logo" and/or "image"
        (?:[\s_-]+(?:by|bt|nc|nd|sa))+
        (?:[\s_-]+logo)?
        (?:[\s_-]+image)?
        |
        # Cases containing only "CC logo" or "CC logo image"
        [\s_-]+logo
        (?:[\s_-]+image)?
    )
    \s*
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
EXCLUDE_CC_LICENSE_CAPTION_REGEX = re.compile(
    r'\bCreative\s*Commons\s*["“”]?\s*Attribution["“”]?',
    re.IGNORECASE,
)


@dataclass
class FigureData:
    figure_num: int | None
    subfigure_label: str | None
    alt_text: str
    img_src: str
    caption: str
    subcaption: str
    referring_text: list[str]


@dataclass
class FigureCandidate:
    tag: Tag
    subfigure_label: str | None
    subcaption: str


class PaperParser:
    """Extract paper metadata and figure information from one XML paper."""

    def __init__(
        self,
        xml_path: Path,
        title: str = "",
        doi: str = "",
        venue: str = "",
        year: int = 0,
    ):
        self.contents = xml_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        parseable_contents, self.figure_alts = self.prepare_xml_for_parsing(
            self.contents
        )
        self.soup: BeautifulSoup = BeautifulSoup(
            parseable_contents,
            "xml",
        )

        if title == "" and doi == "":
            self.title, self.doi = self.extract_title_and_doi_from_filename(
                xml_path.name
            )
        else:
            self.title = title
            self.doi = doi

        self.venue = venue
        self.year = year
        self.figures: list[FigureData] = []

    @staticmethod
    def normalize_text(text: str) -> str:
        """Repair mojibake, collapse whitespace, and trim the result."""
        return " ".join(fix_text(text).split())

    @classmethod
    def repair_compressed_text(cls, text: str) -> str:
        """
        Conservatively restore spaces in heavily compressed figure text.

        This handles common PDF-to-XML artifacts such as:
        "Figure3:OpportunitiesandchallengesforEpidermalComputing..."

        Long tokens are only segmented when WordNinja finds a phrase containing
        a common joining word such as "and", "of", or "for". This avoids
        splitting most legitimate long technical terms.
        """
        text = cls.normalize_text(text)
        text = FIGURE_NUMBER_BOUNDARY_PATTERN.sub(r"\1 \2", text)
        text = CAMEL_CASE_BOUNDARY_PATTERN.sub(" ", text)

        def split_compressed_token(match: re.Match[str]) -> str:
            token = match.group(0)

            if len(token) < 24 or token.isupper():
                return token

            parts = wordninja.split(token)
            normalized_parts = [part.lower() for part in parts]

            if len(parts) < 2:
                return token
            if not any(part in COMMON_JOINED_WORDS for part in normalized_parts):
                return token

            return " ".join(parts)

        text = COMPRESSED_TOKEN_PATTERN.sub(split_compressed_token, text)
        text = PUNCTUATION_BOUNDARY_PATTERN.sub(" ", text)
        return cls.normalize_text(text)

    @classmethod
    def prepare_xml_for_parsing(
        cls,
        xml_text: str,
    ) -> tuple[str, dict[str, str]]:
        """
        Make malformed XML parseable without changing extracted alt text.

        Figure Alt values are removed from the XML before parsing and replaced
        with unique placeholders. This prevents unescaped quotes and ampersands
        inside Alt values from corrupting the XML tree. Existing XML/HTML
        character references in the original alt text are decoded for output.

        Bare ampersands in the rest of the document are escaped after the Alt
        values have been isolated.
        """
        figure_alts: dict[str, str] = {}

        def replace_alt(match: re.Match[str]) -> str:
            placeholder = f"{FIGURE_ALT_PLACEHOLDER_PREFIX}{len(figure_alts)}__"
            raw_alt = cls.normalize_text(match.group("alt"))
            figure_alts[placeholder] = cls.repair_compressed_text(
                html.unescape(raw_alt)
            )

            # The regex deliberately leaves the closing quote and the remainder
            # of the opening tag untouched.
            return f"{match.group('prefix')}{placeholder}"

        parseable_xml = FIGURE_ALT_ATTRIBUTE_PATTERN.sub(
            replace_alt,
            xml_text,
        )
        parseable_xml = BARE_AMPERSAND_PATTERN.sub(
            "&amp;",
            parseable_xml,
        )

        return parseable_xml, figure_alts

    @classmethod
    def get_text_or_empty(
        cls,
        tag,
        *,
        repair_compressed: bool = False,
    ) -> str:
        """Safely extract normalized text from a BeautifulSoup tag."""
        if tag is None:
            return ""

        # The separator prevents text from adjacent child tags from being
        # silently concatenated, such as "Figure" + "3" becoming "Figure3".
        text = tag.get_text(" ", strip=True)

        if repair_compressed:
            return cls.repair_compressed_text(text)
        return cls.normalize_text(text)

    @staticmethod
    def get_attr_or_empty(tag, attr: str) -> str:
        """Safely extract and strip an attribute from a BeautifulSoup tag."""
        if tag is None:
            return ""
        return tag.get(attr, "").strip()

    @staticmethod
    def extract_figure_num(text: str) -> int | None:
        """Extract a figure or table number from text such as 'Figure 4:'."""
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
        Match references such as Figure 4, Figure 4a, Fig 4, and Fig. 4b.

        The final lookahead prevents Figure 4 from matching Figure 40.
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
        subcaption: str,
    ) -> list[str]:
        """Find paragraphs referring to a figure, excluding its own text."""
        referring_text = []
        excluded_text = {alt_text, caption, subcaption, ""}

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

    def _get_figure_alt_text(self, figure: Tag) -> str:
        """Resolve a sanitized Alt placeholder back to its original text."""
        parsed_alt = self.get_attr_or_empty(figure, "Alt")
        return self.figure_alts.get(parsed_alt, parsed_alt)

    @staticmethod
    def _is_inline_figure(figure: Tag) -> bool:
        """Return whether a Figure is embedded inside prose or a list item."""
        return any(
            ancestor.name in INLINE_FIGURE_ANCESTORS
            for ancestor in figure.parents
            if isinstance(ancestor, Tag)
        )

    @classmethod
    def _extract_subfigure_caption(
        cls,
        text: str,
    ) -> tuple[str | None, str]:
        """Extract the last '(a) ...' style subcaption from a text block."""
        text = cls.repair_compressed_text(text)
        matches = list(SUBFIGURE_CAPTION_PATTERN.finditer(text))

        if not matches:
            return None, ""

        match = matches[-1]
        label = match.group("label")
        subcaption_text = cls.repair_compressed_text(match.group("text"))
        return label, f"({label}) {subcaption_text}"

    def _candidate_from_figure(self, figure: Tag) -> FigureCandidate:
        direct_text_parts = [
            str(child)
            for child in figure.children
            if isinstance(child, NavigableString)
        ]
        direct_text = " ".join(direct_text_parts)
        subfigure_label, subcaption = self._extract_subfigure_caption(direct_text)

        return FigureCandidate(
            tag=figure,
            subfigure_label=subfigure_label,
            subcaption=subcaption,
        )

    def _collect_figure_groups(
        self,
    ) -> list[tuple[list[FigureCandidate], str]]:
        """
        Group adjacent standalone figures with their local shared caption.

        Supported layouts include:
        - Figure -> "Figure 2: ..."
        - Figure (a) -> Figure (b) -> "Figure 1: ..."
        - Figure -> "(a) ..." -> Figure -> "(b) ..." -> "Figure 7: ..."

        Empty intermediary paragraphs and tags are ignored. Scanning stops at
        the first substantive unrelated element, preventing a figure from
        borrowing a caption from a later section.
        """
        standalone_figures = [
            figure
            for figure in self.soup.find_all("Figure")
            if not self._is_inline_figure(figure)
        ]
        standalone_ids = {id(figure) for figure in standalone_figures}
        processed_ids: set[int] = set()
        groups: list[tuple[list[FigureCandidate], str]] = []

        for figure in standalone_figures:
            if id(figure) in processed_ids:
                continue

            group = [self._candidate_from_figure(figure)]
            processed_ids.add(id(figure))
            shared_caption = ""

            for sibling in figure.next_siblings:
                if isinstance(sibling, NavigableString):
                    if not str(sibling).strip():
                        continue
                    break

                if not isinstance(sibling, Tag):
                    continue

                if sibling.name == "Figure" and id(sibling) in standalone_ids:
                    group.append(self._candidate_from_figure(sibling))
                    processed_ids.add(id(sibling))
                    continue

                sibling_text = self.get_text_or_empty(
                    sibling,
                    repair_compressed=True,
                )

                if not sibling_text:
                    continue

                if sibling.name == "P":
                    if FIGURE_CAPTION_PATTERN.match(sibling_text):
                        shared_caption = sibling_text
                        break

                    label, subcaption = self._extract_subfigure_caption(sibling_text)
                    if label is not None:
                        current = group[-1]
                        if not current.subcaption:
                            current.subfigure_label = label
                            current.subcaption = subcaption
                        continue

                # Do not search arbitrarily far forward for a caption.
                break

            groups.append((group, shared_caption))

        return groups

    def _extract_single_figure(
        self,
        candidate: FigureCandidate,
        caption: str,
        paragraphs: list[str],
        parent_path: Path | None,
    ) -> FigureData | None:
        figure = candidate.tag

        # ImageData should belong to this Figure. Using find() avoids silently
        # borrowing the ImageData element from a later Figure.
        image_data = figure.find("ImageData")
        alt_text = self._get_figure_alt_text(figure)

        # Creative Commons license logos are sometimes represented as figures.
        if EXCLUDE_CC_LICENSE_ALT_TEXT_REGEX.fullmatch(alt_text):
            return None
        if EXCLUDE_CC_LICENSE_CAPTION_REGEX.search(caption):
            return None

        figure_num = self.extract_figure_num(caption)

        img_path = self.get_attr_or_empty(image_data, "src")
        img_src = str(parent_path / img_path) if img_path and parent_path else ""

        figure_info = FigureData(
            figure_num=figure_num,
            subfigure_label=candidate.subfigure_label,
            alt_text=alt_text,
            img_src=img_src,
            caption=caption,
            subcaption=candidate.subcaption,
            referring_text=[],
        )

        if figure_num is not None:
            figure_info.referring_text = self.find_referring_text(
                paragraphs=paragraphs,
                figure_num=figure_num,
                alt_text=alt_text,
                caption=caption,
                subcaption=candidate.subcaption,
            )

        return figure_info

    def _extract_paragraphs(self) -> list[str]:
        """Extract nonempty outermost paragraphs with minimal deduplication."""
        paragraphs: list[str] = []
        seen: set[str] = set()

        for paragraph in self.soup.find_all("P"):
            # Avoid extracting both a malformed outer P and its nested P again.
            if paragraph.find_parent("P") is not None:
                continue

            paragraph_text = self.get_text_or_empty(
                paragraph,
                repair_compressed=True,
            )
            if not paragraph_text or paragraph_text in seen:
                continue

            seen.add(paragraph_text)
            paragraphs.append(paragraph_text)

        return paragraphs

    def _extract_figure_info(self, parent_path: Path | None) -> None:
        self.figures = []
        paragraphs = self._extract_paragraphs()

        for candidates, caption in self._collect_figure_groups():
            for candidate in candidates:
                figure_info = self._extract_single_figure(
                    candidate=candidate,
                    caption=caption,
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
    venue, year = directory.name.split("_", maxsplit=1)
    return venue, int(year), directory


def collect_venue_year_dirs(folder_path: Path) -> list[tuple[str, int, Path]]:
    sub_dirs = [directory for directory in folder_path.iterdir() if directory.is_dir()]

    invalid_dirs = [
        directory.name
        for directory in sub_dirs
        if not VENUE_YEAR_DIR_PATTERN.fullmatch(directory.name)
    ]
    if invalid_dirs:
        invalid_names = ", ".join(sorted(invalid_dirs))
        raise ValueError(
            "All subdirectories must follow the pattern 'Venue_Year' "
            f"(for example, 'ASSETS_2025'). Invalid: {invalid_names}"
        )

    return sorted(parse_venue_year_dir(directory) for directory in sub_dirs)


def process_venue_year_dir(venue: str, year: int, sub_dir: Path) -> list[dict]:
    extracted_papers = []
    xml_files = sorted(sub_dir.glob("*.xml"))

    for xml_file in tqdm(
        xml_files,
        desc=f"Processing {venue} {year} in {sub_dir}",
    ):
        try:
            parser = PaperParser(xml_file, venue=venue, year=year)
            extracted_papers.append(parser.extract(parent_path=sub_dir))
        except Exception as error:
            print(f"Error processing {venue} {year}, {xml_file}: {error}")

    print(
        f"Finished processing {venue} {year}. "
        f"Extracted info for {len(extracted_papers)} papers.",
        end="\n" + "-" * TERMINAL_COLUMNS + "\n",
    )
    return extracted_papers


def write_extracted_info(
    folder_path: Path,
    venue: str,
    year: int,
    extracted_info: list[dict],
) -> None:
    folder_path.mkdir(parents=True, exist_ok=True)
    output_path = folder_path / f"{venue}_{year}_extracted_info.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            extracted_info,
            file,
            indent=4,
            ensure_ascii=False,
        )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Parse XML papers and extract paper and figure information."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder containing Venue_Year subfolders with XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_path: Path = args.folder
    output_dir: Path = args.output_dir or folder_path

    if not folder_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {folder_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for venue, year, sub_dir in collect_venue_year_dirs(folder_path):
        extracted_info = process_venue_year_dir(venue, year, sub_dir)
        write_extracted_info(
            folder_path=output_dir,
            venue=venue,
            year=year,
            extracted_info=extracted_info,
        )


if __name__ == "__main__":
    main()
