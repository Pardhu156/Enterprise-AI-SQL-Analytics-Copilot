"""Create the Olist PostgreSQL schema and bulk-load processed CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import psycopg2
from psycopg2 import sql

from db_config import DatabaseConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
VIEWS_PATH = PROJECT_ROOT / "database" / "views.sql"

# Dependency-safe order: referenced dimension/parent tables precede child tables.
LOAD_PLAN = (
    ("geolocation", "geolocation.csv"),
    ("product_category_translation", "product_category_translation.csv"),
    ("customers", "customers.csv"),
    ("sellers", "sellers.csv"),
    ("products", "products.csv"),
    ("orders", "orders.csv"),
    ("order_items", "order_items.csv"),
    ("payments", "payments.csv"),
    ("reviews", "reviews.csv"),
)


def require_input_files(processed_dir: Path) -> None:
    missing = [filename for _, filename in LOAD_PLAN if not (processed_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing processed CSV files: " + ", ".join(missing)
            + ". Run src/data_processing.py after adding all expected raw files."
        )


def execute_sql_file(cursor: "psycopg2.extensions.cursor", path: Path) -> None:
    logging.info("Executing %s", path)
    cursor.execute(path.read_text(encoding="utf-8"))


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as source:
        header = source.readline().rstrip("\r\n")
    if not header:
        raise ValueError(f"CSV has no header: {path}")
    # Olist column names contain no commas or embedded quotes after processing.
    columns = header.split(",")
    if not all(columns):
        raise ValueError(f"CSV contains an empty column name: {path}")
    return columns


def copy_csv(cursor: "psycopg2.extensions.cursor", table: str, path: Path) -> int:
    columns = csv_columns(path)
    copy_statement = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '', ENCODING 'UTF8')"
    ).format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
    )
    with path.open("r", encoding="utf-8") as source:
        cursor.copy_expert(copy_statement.as_string(cursor), source)
    cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
    return int(cursor.fetchone()[0])


def load_database(processed_dir: Path, truncate: bool, create_views: bool) -> None:
    require_input_files(processed_dir)
    config = DatabaseConfig.from_env()

    try:
        with psycopg2.connect(**config.as_connect_kwargs()) as connection:
            with connection.cursor() as cursor:
                execute_sql_file(cursor, SCHEMA_PATH)
                if truncate:
                    logging.warning("Truncating existing Olist table data before load")
                    cursor.execute(
                        "TRUNCATE TABLE reviews, payments, order_items, orders, products, "
                        "sellers, customers, product_category_translation, geolocation "
                        "RESTART IDENTITY CASCADE"
                    )

                for table, filename in LOAD_PLAN:
                    path = processed_dir / filename
                    cursor.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                    )
                    before = int(cursor.fetchone()[0])
                    total = copy_csv(cursor, table, path)
                    logging.info(
                        "%s: inserted=%d, table_total=%d", table, total - before, total
                    )

                if create_views:
                    execute_sql_file(cursor, VIEWS_PATH)
        logging.info("Database load committed successfully")
    except psycopg2.Error:
        logging.exception("PostgreSQL load failed; the transaction was rolled back")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Delete existing rows before loading (use for a clean, repeatable reload).",
    )
    parser.add_argument(
        "--skip-views", action="store_true", help="Do not apply database/views.sql."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_database(args.processed_dir.resolve(), args.truncate, not args.skip_views)


if __name__ == "__main__":
    main()
