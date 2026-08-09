"""Initialize the Compose PostgreSQL volume by reusing the Phase 1 pipeline."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import psycopg2
from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import OUTPUT_NAMES, process_all  # noqa: E402
from db_config import DatabaseConfig  # noqa: E402
from load_postgres import (  # noqa: E402
    LOAD_PLAN,
    SCHEMA_PATH,
    VIEWS_PATH,
    execute_sql_file,
    load_database,
    require_input_files,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def classify_counts(counts: dict[str, int]) -> str:
    """Classify the warehouse without treating partial data as safe to reload."""
    populated = [table for table, count in counts.items() if count > 0]
    if not populated:
        return "empty"
    if len(populated) == len(counts):
        return "populated"
    empty = [table for table, count in counts.items() if count == 0]
    raise RuntimeError(
        "Container database is only partially populated. Non-empty tables: "
        f"{', '.join(populated)}. Empty tables: {', '.join(empty)}. "
        "Reset the Docker volume or repair the database manually before restarting."
    )


def ensure_schema_and_counts() -> dict[str, int]:
    config = DatabaseConfig.from_env()
    with psycopg2.connect(**config.as_connect_kwargs()) as connection:
        with connection.cursor() as cursor:
            execute_sql_file(cursor, SCHEMA_PATH)
            counts: dict[str, int] = {}
            for table, _ in LOAD_PLAN:
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                )
                counts[table] = int(cursor.fetchone()[0])
    return counts


def refresh_views() -> None:
    config = DatabaseConfig.from_env()
    with psycopg2.connect(**config.as_connect_kwargs()) as connection:
        with connection.cursor() as cursor:
            execute_sql_file(cursor, VIEWS_PATH)


def processed_files_available(directory: Path) -> bool:
    try:
        require_input_files(directory)
    except FileNotFoundError:
        return False
    return True


def raw_files_available(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in OUTPUT_NAMES)


def initialize() -> None:
    state = classify_counts(ensure_schema_and_counts())
    if state == "populated":
        logging.info("Olist tables already contain data; skipping CSV load")
        refresh_views()
        logging.info("Analytical views are current")
        return

    if processed_files_available(PROCESSED_DIR):
        logging.info("Loading existing processed Olist CSV files")
        load_database(PROCESSED_DIR, truncate=False, create_views=True)
        return

    if not raw_files_available(RAW_DIR):
        expected = ", ".join(sorted(OUTPUT_NAMES))
        raise FileNotFoundError(
            "The database is empty and no complete processed or raw Olist dataset was found. "
            f"Place these source files in data/raw/: {expected}"
        )

    with tempfile.TemporaryDirectory(prefix="olist-processed-") as temporary:
        generated = Path(temporary)
        logging.info("Processed CSVs are absent; processing mounted raw Olist files")
        process_all(RAW_DIR, generated)
        load_database(generated, truncate=False, create_views=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    initialize()


if __name__ == "__main__":
    main()
