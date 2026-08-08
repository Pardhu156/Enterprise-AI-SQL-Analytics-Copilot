import pytest

from src.text_to_sql.schema_manager import ColumnInfo, RelationInfo, SchemaSnapshot
from src.text_to_sql.sql_generator import SQLGenerationError, extract_sql
from src.text_to_sql.sql_validator import SQLValidator


@pytest.fixture()
def schema() -> SchemaSnapshot:
    return SchemaSnapshot(
        relations=(
            RelationInfo(
                "customers",
                "TABLE",
                (
                    ColumnInfo("customer_id", "text", False, True),
                    ColumnInfo("customer_state", "text", False),
                ),
            ),
            RelationInfo(
                "orders",
                "TABLE",
                (
                    ColumnInfo("order_id", "text", False, True),
                    ColumnInfo("customer_id", "text", False),
                    ColumnInfo("order_status", "text", False),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c.customer_state, COUNT(o.order_id) AS order_count FROM customers AS c JOIN orders AS o ON o.customer_id = c.customer_id GROUP BY c.customer_state",
        "SELECT COUNT(*) AS customer_count FROM customers AS c",
        "WITH delivered AS (SELECT o.customer_id FROM orders AS o WHERE o.order_status = 'delivered') SELECT d.customer_id FROM delivered AS d",
        "SELECT CURRENT_DATE AS today",
        "SELECT recent.customer_id FROM (SELECT c.customer_id FROM customers AS c) AS recent",
    ],
)
def test_safe_selects_pass(sql: str, schema: SchemaSnapshot) -> None:
    result = SQLValidator().validate(sql, schema)
    assert result.valid, result.reason


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DELETE FROM orders",
        "UPDATE orders SET order_status = 'x'",
        "INSERT INTO customers (customer_id) VALUES ('x')",
        "ALTER TABLE customers ADD COLUMN unsafe TEXT",
        "CREATE TABLE unsafe (id INTEGER)",
        "SELECT c.customer_id FROM customers AS c; DROP TABLE customers",
        "WITH removed AS (DELETE FROM orders RETURNING order_id) SELECT r.order_id FROM removed AS r",
        "SELECT set_config('search_path', 'public', false) AS changed",
        "SELECT pg_read_file('/etc/passwd') AS contents",
    ],
)
def test_unsafe_statements_fail(sql: str, schema: SchemaSnapshot) -> None:
    result = SQLValidator().validate(sql, schema)
    assert not result.valid


def test_wildcard_comments_unknown_objects_and_malformed_sql_fail(schema: SchemaSnapshot) -> None:
    validator = SQLValidator()
    statements = (
        "SELECT * FROM customers",
        "SELECT c.customer_id FROM customers AS c -- hidden text",
        "SELECT x.value FROM invented AS x",
        "SELECT c.invented FROM customers AS c",
        "SELECT FROM",
    )
    for statement in statements:
        assert not validator.validate(statement, schema).valid


def test_markdown_sql_extraction() -> None:
    response = "Here is the query:\n```sql\nSELECT c.customer_id FROM customers AS c;\n```"
    assert extract_sql(response) == "SELECT c.customer_id FROM customers AS c;"


def test_empty_or_non_sql_llm_output_fails() -> None:
    with pytest.raises(SQLGenerationError):
        extract_sql("")
    with pytest.raises(SQLGenerationError):
        extract_sql("I cannot answer that question.")
