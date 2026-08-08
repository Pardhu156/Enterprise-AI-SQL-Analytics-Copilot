from src.text_to_sql.schema_manager import (
    ColumnInfo,
    ForeignKeyInfo,
    RelationInfo,
    SchemaSnapshot,
)


def test_schema_serialization_includes_keys_views_and_relationships() -> None:
    snapshot = SchemaSnapshot(
        relations=(
            RelationInfo(
                name="customers",
                kind="TABLE",
                columns=(
                    ColumnInfo("customer_id", "character varying", False, True),
                    ColumnInfo("customer_state", "character", False),
                ),
            ),
            RelationInfo(
                name="vw_customer_orders",
                kind="VIEW",
                columns=(ColumnInfo("customer_id", "character varying"),),
                description="Customer order metrics",
            ),
        ),
        foreign_keys=(ForeignKeyInfo("orders", "customer_id", "customers", "customer_id"),),
    )

    context = snapshot.to_prompt()

    assert "TABLE customers" in context
    assert "customer_id CHARACTER VARYING PRIMARY KEY" in context
    assert "VIEW vw_customer_orders" in context
    assert "orders.customer_id -> customers.customer_id" in context
    assert snapshot.relation_names == {"customers", "vw_customer_orders"}
