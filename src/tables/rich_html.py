from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class RichCell:
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.grid: list[list[RichCell | None]] = []
        self.row: list[RichCell | None] = []
        self.row_index = -1
        self.spans: dict[int, dict[int, int]] = {}
        self.current: RichCell | None = None
        self.current_text: list[str] = []
        self.in_table = False
        self.in_row = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.row = []
            self.row_index = len(self.grid)
            if self.row_index > 0:
                prior_width = len(self.grid[-1])
                carried = self.spans.get(self.row_index, {})
                self.row = [None] * max(prior_width, max(carried.keys(), default=-1) + 1)
            self.in_row = True
        elif tag in {"td", "th"} and self.in_row:
            attributes = dict(attrs)
            row_span = max(1, int(attributes.get("rowspan", "1") or "1"))
            col_span = max(1, int(attributes.get("colspan", "1") or "1"))
            self.current = RichCell(
                text="",
                row_span=row_span,
                col_span=col_span,
                is_header=tag == "th",
            )
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current is not None:
            self.current.text = " ".join("".join(self.current_text).split())
            self._place_cell(self.current)
            self.current = None
            self.current_text = []
        elif tag == "tr" and self.in_row:
            self.grid.append(self.row)
            self.row = []
            self.in_row = False
        elif tag == "table":
            self.in_table = False

    def _place_cell(self, cell: RichCell) -> None:
        column = 0
        carried = self.spans.get(self.row_index, {})
        while column in carried or (column < len(self.row) and self.row[column] is not None):
            column += 1
        width = column + cell.col_span
        while len(self.row) < width:
            self.row.append(None)
        self.row[column] = cell
        for offset in range(1, cell.col_span):
            position = column + offset
            self.row[position] = None
        for covered_row in range(self.row_index + 1, self.row_index + cell.row_span):
            for offset in range(cell.col_span):
                self.spans.setdefault(covered_row, {})[column + offset] = 1


def parse_html_table_rich(html: str) -> list[list[RichCell | None]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    if parser.in_row:
        parser.grid.append(parser.row)
    width = max((len(row) for row in parser.grid), default=0)
    for row in parser.grid:
        while len(row) < width:
            row.append(None)
    return parser.grid
