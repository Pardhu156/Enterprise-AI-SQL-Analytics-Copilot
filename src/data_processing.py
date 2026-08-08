"""Conservative cleaning pipeline for the Olist CSV datasets."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_NAMES = {
    "olist_customers_dataset.csv": "customers.csv",
    "olist_orders_dataset.csv": "orders.csv",
    "olist_order_items_dataset.csv": "order_items.csv",
    "olist_products_dataset.csv": "products.csv",
    "olist_sellers_dataset.csv": "sellers.csv",
    "olist_order_payments_dataset.csv": "payments.csv",
    "olist_order_reviews_dataset.csv": "reviews.csv",
    "olist_geolocation_dataset.csv": "geolocation.csv",
    "product_category_name_translation.csv": "product_category_translation.csv",
}

TIMESTAMP_COLUMNS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
}

# Pandas otherwise serializes nullable integer columns as values such as "42.0",
# which PostgreSQL INTEGER columns correctly reject during COPY.
INTEGER_COLUMNS = {
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "geolocation_zip_code_prefix",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "order_item_id",
    "payment_sequential",
    "payment_installments",
    "review_score",
}


def standardize_column_name(name: str) -> str:
    """Convert a source column label to lower snake_case."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def clean_dataframe(frame: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, int]:
    """Apply loss-minimizing, dataset-agnostic cleanup and return duplicate count."""
    cleaned = frame.copy()
    cleaned.columns = [standardize_column_name(column) for column in cleaned.columns]

    duplicate_columns = cleaned.columns[cleaned.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"{dataset_name}: column standardization created duplicate names: "
            f"{duplicate_columns}"
        )

    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(cleaned)

    for column in sorted(TIMESTAMP_COLUMNS.intersection(cleaned.columns)):
        original_non_null = cleaned[column].notna()
        parsed = pd.to_datetime(cleaned[column], errors="coerce")
        newly_invalid = int((original_non_null & parsed.isna()).sum())
        if newly_invalid:
            raise ValueError(
                f"{dataset_name}: {newly_invalid} non-null values in {column} "
                "could not be parsed as timestamps"
            )
        cleaned[column] = parsed

    for column in sorted(INTEGER_COLUMNS.intersection(cleaned.columns)):
        source_non_null = cleaned[column].notna()
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        newly_invalid = int((source_non_null & numeric.isna()).sum())
        non_integral = int((numeric.dropna() % 1 != 0).sum())
        if newly_invalid or non_integral:
            raise ValueError(
                f"{dataset_name}: {column} contains {newly_invalid} non-numeric and "
                f"{non_integral} non-integral values"
            )
        cleaned[column] = numeric.astype("Int64")

    # Missing values are deliberately retained. In Olist, null delivery and approval
    # timestamps often describe a real order state rather than bad data.
    return cleaned, duplicates_removed


def process_all(raw_dir: Path, processed_dir: Path) -> None:
    """Clean every CSV in raw_dir and write deterministic processed CSV files."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Place the Olist files there first."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    for source_path in csv_files:
        logging.info("Reading %s", source_path.name)
        frame = pd.read_csv(source_path, low_memory=False)
        loaded_rows = len(frame)
        cleaned, duplicates_removed = clean_dataframe(frame, source_path.name)

        output_name = OUTPUT_NAMES.get(source_path.name, source_path.name)
        output_path = processed_dir / output_name
        cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        logging.info(
            "%s: loaded=%d, exact_duplicates_removed=%d, saved=%d -> %s",
            source_path.name,
            loaded_rows,
            duplicates_removed,
            len(cleaned),
            output_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    process_all(args.raw_dir.resolve(), args.processed_dir.resolve())


if __name__ == "__main__":
    main()
