"""Pulls real NFIP flood-insurance claims from FEMA's public OpenFEMA API
(no key required) for the 8 coastal counties covered by the
surge-exposure / surge-exposure-agent buildings dataset.

Source: https://www.fema.gov/api/open/v2/FimaNfipClaims (the "FIMA NFIP
Redacted Claims v2" dataset). This is real, individual (redacted) claim
data -- not synthetic. Lat/lon are FEMA's own privacy-generalized
coordinates (see the grid-size probe this script prints), not exact
addresses.

NOTE: this v2 endpoint is deprecated 2026-10-15 in favor of a renamed v3
dataset ("NFIP Redacted Claims v3"). Fine for now; re-point ENTITY/BASE_URL
at the v3 endpoint before that date.
"""
import csv
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
PAGE_SIZE = 1000
MAX_PAGES_PER_COUNTY = 20  # cap at 20k claims/county -- some of these counties
                            # (e.g. Orleans Parish post-Katrina) have 100k+
                            # historical claims; 20k real claims per county is
                            # plenty of signal for a portfolio-scale model
                            # and keeps the fetch fast.

# region_slug -> (state postal code, 5-digit county FIPS code)
# Matched by hand from each region's bbox in data/regions.csv.
REGION_COUNTIES = {
    "south-beach-miami": ("FL", "12086"),        # Miami-Dade County
    "clearwater-beach": ("FL", "12103"),         # Pinellas County
    "fort-myers-beach": ("FL", "12071"),         # Lee County (same county the original validation study used)
    "french-quarter-nola": ("LA", "22071"),      # Orleans Parish
    "galveston-seawall": ("TX", "48167"),        # Galveston County
    "charleston-battery": ("SC", "45019"),       # Charleston County
    "outer-banks-nags-head": ("NC", "37055"),    # Dare County
    "ocean-city-md": ("MD", "24047"),            # Worcester County
}

FIELDS = [
    "id", "state", "countyCode", "censusTract", "reportedZipCode",
    "latitude", "longitude", "yearOfLoss", "dateOfLoss", "floodEvent",
    "waterDepth", "ratedFloodZone", "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim", "netBuildingPaymentAmount",
    "elevatedBuildingIndicator", "occupancyType",
]


def fetch_county(state: str, county_code: str) -> list[dict]:
    """Paginates through every claim for one county. No API key needed."""
    rows: list[dict] = []
    skip = 0
    for _ in range(MAX_PAGES_PER_COUNTY):
        filter_clause = f"state eq '{state}' and countyCode eq '{county_code}'"
        params = {
            "$top": PAGE_SIZE,
            "$skip": skip,
            "$filter": filter_clause,
            "$select": ",".join(FIELDS),
        }
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            import json
            data = json.loads(resp.read())
        batch = data.get("FimaNfipClaims", [])
        if not batch:
            break
        rows.extend(batch)
        skip += PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.2)  # be polite to a free, unauthenticated public API
    return rows


def main() -> None:
    out_path = Path(__file__).parent / "claims_raw.csv"
    all_rows: list[dict] = []

    for region_slug, (state, county_code) in REGION_COUNTIES.items():
        print(f"Fetching {region_slug} ({state}/{county_code})...", file=sys.stderr)
        rows = fetch_county(state, county_code)
        for r in rows:
            r["region_slug"] = region_slug
        all_rows.extend(rows)
        print(f"  -> {len(rows)} claims", file=sys.stderr)

    if not all_rows:
        print("No claims fetched -- aborting without writing an empty file.", file=sys.stderr)
        sys.exit(1)

    fieldnames = FIELDS + ["region_slug"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"\nWrote {len(all_rows)} real NFIP claims to {out_path}", file=sys.stderr)

    # Grid-size probe: NFIP generalizes lat/lon for privacy. Print the
    # smallest observed non-zero gap between distinct values so
    # prepare_training_data.py buckets onto FEMA's real precision instead
    # of an assumed one.
    lats = sorted({r["latitude"] for r in all_rows if r.get("latitude") is not None})
    lons = sorted({r["longitude"] for r in all_rows if r.get("longitude") is not None})
    lat_gaps = sorted({round(b - a, 4) for a, b in zip(lats, lats[1:]) if b - a > 0})
    lon_gaps = sorted({round(b - a, 4) for a, b in zip(lons, lons[1:]) if b - a > 0})
    print(f"Distinct latitudes: {len(lats)}, smallest gaps: {lat_gaps[:5]}", file=sys.stderr)
    print(f"Distinct longitudes: {len(lons)}, smallest gaps: {lon_gaps[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
