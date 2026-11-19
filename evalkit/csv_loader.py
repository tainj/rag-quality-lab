"""Load CSV articles → full documents or chunks."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
import html2text
import pandas as pd


def _clean_html(raw_html: str) -> str:
    """Strip HTML to plain text, matching production html_to_text logic."""
    if not raw_html or pd.isna(raw_html):
        return ""
    text = str(raw_html)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all("img"):
        alt = tag.get("alt", "")
        if "Рисунок " in alt:
            tag.decompose()
        else:
            tag.replace_with(alt)
    for tag in soup.find_all("a", href=True):
        tag.replace_with(str(tag["href"]))
    for tag in soup.find_all("table"):
        rows = []
        for tr in tag.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        tag.replace_with("\n".join(rows))
    cleaned = html2text.html2text(soup.prettify())
    return re.sub(r"\n{2,}", "\n", cleaned).strip()


def clean_and_filter(df: pd.DataFrame, min_chars_per_article=50) -> pd.DataFrame:
    """Clean HTML and filter out empty / too-short articles."""
    df = df.copy()
    df["clean_text"] = df["IE_DETAIL_TEXT"].apply(_clean_html)
    df["article_name"] = df["IE_NAME"].fillna("").astype(str)
    df["xml_id"] = df["IE_XML_ID"].astype(str)
    df["version"] = (
        df.get("VERSION", pd.Series("unknown", index=df.index))
        .fillna("unknown")
        .astype(str)
    )
    df["link"] = df.get("LINK", pd.Series("", index=df.index)).fillna("").astype(str)

    before = len(df)
    df = df[df["clean_text"].str.len() >= min_chars_per_article].reset_index(drop=True)
    print(f"Filtered: {before} → {len(df)} articles (>= {min_chars_per_article} chars)")

    return df
