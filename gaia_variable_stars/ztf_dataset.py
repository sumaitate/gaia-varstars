
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

from gaia_variable_stars.config import (
    PROJECT_ROOT,
    RAW_ROOT,
    INTERIM_ROOT,
    SOURCE_FILE,
    MIN_POINTS,
)

RELEASE = "ztf_dr24"
RADIUS = 1 / 3600
BATCH_SIZE = 25

API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

SPLITS = PROJECT_ROOT / "splits"

ZTF_SOURCES = RAW_ROOT / "ztf_sources.csv"
ZTF_PHOTOMETRY = RAW_ROOT / "ztf_photometry.csv"
ZTF_LABELED = INTERIM_ROOT / "ztf_labeled.csv"
ZTF_CLEAN = INTERIM_ROOT / "ztf_clean.csv"


def get_sources():
    sources = pd.read_csv(SOURCE_FILE)

    splits = pd.concat(
        [
            pd.read_csv(SPLITS / f"{split}_source_ids.csv")
            .assign(split=split)
            for split in ["train", "validation", "test"]
        ],
        ignore_index=True,
    )

    return sources.merge(splits, on="source_id")


def match(sources):
    upload = Table.from_pandas(
        sources[["source_id", "ra", "dec"]]
    )

    query = f"""
    SELECT
        gaia.source_id,
        ztf.oid,
        ztf.ra AS ztf_ra,
        ztf.dec AS ztf_dec,
        ztf.nobs,
        DISTANCE(
            POINT(gaia.ra, gaia.dec),
            POINT(ztf.ra, ztf.dec)
        ) AS separation
    FROM TAP_UPLOAD.sources AS gaia,
         ztf_objects_dr24 AS ztf
    WHERE CONTAINS(
        POINT(ztf.ra, ztf.dec),
        CIRCLE(gaia.ra, gaia.dec, {RADIUS:.12f})
    ) = 1
    AND (
        ztf.fid = 2
        AND ztf.nobs >= {MIN_POINTS}
    )
    """

    result = Irsa.tap.run_async(
        query,
        uploads={"sources": upload},
    )

    matches = result.to_table().to_pandas()

    matches = (
        matches.sort_values(
            ["separation", "nobs", "oid"],
            ascending=[True, False, True],
        )
        .drop_duplicates("oid")
        .drop_duplicates("source_id")
    )

    matches["oid"] = matches["oid"].astype("Int64")

    print("Matched sources:", len(matches))

    return sources.merge(
        matches,
        on="source_id",
        how="left",
    )


def download(objects):
    params = {
        "ID": objects,
        "COLLECTION": RELEASE,
        "FORMAT": "csv",
    }

    url = f"{API}?{urlencode(params, doseq=True)}"

    data = pd.read_csv(url, dtype={"oid": "int64"})

    time.sleep(1)

    return data


def get_photometry(sources):
    objects = sources["oid"].dropna().astype("int64").tolist()

    photometry = pd.concat(
        [
            download(objects[start:start + BATCH_SIZE])
            for start in range(0, len(objects), BATCH_SIZE)
        ],
        ignore_index=True,
    )

    photometry.to_csv(ZTF_PHOTOMETRY, index=False)

    return photometry


def label(sources, photometry):
    return photometry.merge(
        sources.dropna(subset=["oid"]),
        on="oid",
        how="inner",
        suffixes=("_ztf", ""),
    )


def clean(data):
    data = data.rename(columns={
        "hjd": "time",
        "mag": "magnitude",
        "magerr": "magnitude_error",
    })

    for col in ["time", "magnitude", "magnitude_error"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    valid = (
        data["catflags"].eq(0)
        & data["filtercode"].eq("zr")
        & np.isfinite(data["time"])
        & np.isfinite(data["magnitude"])
        & np.isfinite(data["magnitude_error"])
        & data["magnitude_error"].gt(0)
    )

    cleaned = data.loc[valid].copy()

    cleaned["time"] -= 2400000.5

    cleaned = (
        cleaned.sort_values(["source_id", "time"])
        .drop_duplicates(["source_id", "time"])
        .reset_index(drop=True)
    )

    return cleaned


def get_data():
    sources = match(get_sources())
    photometry = get_photometry(sources)

    return sources, photometry


def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    INTERIM_ROOT.mkdir(parents=True, exist_ok=True)

    sources, photometry = get_data()

    labeled = label(sources, photometry)
    cleaned = clean(labeled)

    counts = cleaned.groupby("source_id").size()

    sources["n_r_epochs"] = (
        sources["source_id"]
        .map(counts)
        .fillna(0)
        .astype("int64")
    )

    cleaned = cleaned.loc[
        cleaned["source_id"].map(counts).ge(MIN_POINTS)
    ].copy()

    sources.to_csv(ZTF_SOURCES, index=False)
    labeled.to_csv(ZTF_LABELED, index=False)
    cleaned.to_csv(ZTF_CLEAN, index=False)

    print("Gaia sources:", len(sources))
    print("ZTF photometry:", len(photometry))
    print("Clean observations:", len(cleaned))
    print("Eligible sources:", cleaned["source_id"].nunique())


if __name__ == "__main__":
    main()