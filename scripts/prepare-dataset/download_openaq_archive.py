"""Download a date-bounded slice of one OpenAQ location from the public S3 archive."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BUCKET_URL = "https://openaq-data-archive.s3.amazonaws.com"
LOCATION_ID = 3276359
DEFAULT_START = dt.date(2024, 11, 19)
DEFAULT_END = dt.date(2026, 11, 10)
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def request_bytes(url: str, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CO3029-air-quality-project/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (OSError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Request retries exhausted")


def list_keys(year: int) -> list[str]:
    prefix = f"records/csv.gz/locationid={LOCATION_ID}/year={year}/"
    token: str | None = None
    keys: list[str] = []
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}"
        root = ET.fromstring(request_bytes(url))
        keys.extend(node.text for node in root.findall("s3:Contents/s3:Key", S3_NS) if node.text)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) == "true"
        if not truncated:
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        if not token:
            raise RuntimeError(f"S3 listing for {year} is truncated without a continuation token")


def date_from_key(key: str) -> dt.date:
    stem = Path(key).name
    date_token = stem.removeprefix(f"location-{LOCATION_ID}-").removesuffix(".csv.gz")
    return dt.datetime.strptime(date_token, "%Y%m%d").date()


def download(start: dt.date, end: dt.date, output: Path, workers: int = 8) -> dict[str, object]:
    if end < start:
        raise ValueError("--end must be on or after --start")

    output.mkdir(parents=True, exist_ok=True)
    all_keys: list[str] = []
    for year in range(start.year, end.year + 1):
        all_keys.extend(list_keys(year))
    selected = sorted(key for key in all_keys if start <= date_from_key(key) <= end)

    def fetch_one(key: str) -> dict[str, object]:
        destination = output / Path(key).name
        payload = destination.read_bytes() if destination.exists() else request_bytes(
            f"{BUCKET_URL}/{urllib.parse.quote(key, safe='/=')}"
        )
        # Read once to reject incomplete or corrupt gzip files before keeping them.
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            stream.read()
        if not destination.exists():
            destination.write_bytes(payload)
        return {
            "key": key,
            "file": destination.name,
            "date": date_from_key(key).isoformat(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    downloaded: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, key): key for key in selected}
        for index, future in enumerate(as_completed(futures), start=1):
            downloaded.append(future.result())
            if index % 25 == 0 or index == len(selected):
                print(f"Verified {index}/{len(selected)} files", file=sys.stderr)
    downloaded.sort(key=lambda item: str(item["date"]))

    manifest: dict[str, object] = {
        "source": BUCKET_URL,
        "location_id": LOCATION_ID,
        "start_date_inclusive": start.isoformat(),
        "end_date_inclusive": end.isoformat(),
        "listed_file_count_in_range": len(selected),
        "files": downloaded,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=DEFAULT_START,
        help=f"Inclusive start date (YYYY-MM-DD; default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=DEFAULT_END,
        help=f"Inclusive end date (YYYY-MM-DD; default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "openaq_cmt8_2y",
    )
    args = parser.parse_args()
    result = download(args.start, args.end, args.output)
    print(
        f"Saved {result['listed_file_count_in_range']} daily files to {args.output} "
        f"for {args.start} through {args.end}."
    )


if __name__ == "__main__":
    main()
