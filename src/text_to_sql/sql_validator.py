"""AST-based safety and schema validation for generated PostgreSQL queries."""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from .schema_manager import SchemaSnapshot


FORBIDDEN_NODE_KEYS = frozenset(
    {
        "alter",
        "analyze",
        "attach",
        "cache",
        "call",
        "command",
        "commit",
        "copy",
        "create",
        "delete",
        "detach",
        "do",
        "drop",
        "execute",
        "grant",
        "insert",
        "kill",
        "load_data",
        "lock",
        "merge",
        "refresh",
        "replace",
        "reset",
        "revoke",
        "rollback",
        "set",
        "transaction",
        "truncate",
        "uncache",
        "update",
        "use",
        "vacuum",
    }
)

FORBIDDEN_FUNCTIONS = frozenset(
    {
        "dblink_connect",
        "dblink_disconnect",
        "dblink_exec",
        "lo_import",
        "lo_unlink",
        "nextval",
        "pg_advisory_lock",
        "pg_advisory_lock_shared",
        "pg_cancel_backend",
        "pg_read_binary_file",
        "pg_read_file",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_sleep",
        "pg_terminate_backend",
        "pg_write_file",
        "set_config",
        "setval",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    normalized_sql: str | None = None
    referenced_relations: tuple[str, ...] = field(default_factory=tuple)


class SQLValidator:
    """Permit one explicit-column SELECT against the introspected public schema."""

    def validate(self, sql: str, schema: SchemaSnapshot) -> ValidationResult:
        if not sql or not sql.strip():
            return ValidationResult(False, "SQL is empty")
        if _contains_comment(sql):
            return ValidationResult(False, "SQL comments are not allowed")

        try:
            statements = [statement for statement in sqlglot.parse(sql, read="postgres") if statement]
        except ParseError as exc:
            return ValidationResult(False, f"Malformed PostgreSQL SQL: {exc}")
        if len(statements) != 1:
            return ValidationResult(False, "Exactly one SQL statement is required")

        statement = statements[0]
        if not _is_query_root(statement):
            return ValidationResult(False, "Only SELECT queries are allowed")

        forbidden = sorted(
            {node.key.upper() for node in statement.walk() if node.key in FORBIDDEN_NODE_KEYS}
        )
        if forbidden:
            return ValidationResult(False, f"Forbidden SQL operation: {', '.join(forbidden)}")

        if any(
            isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)
            for node in statement.walk()
        ):
            return ValidationResult(False, "SELECT * and wildcard columns are not allowed")

        forbidden_functions = sorted(
            {
                node.name.lower()
                for node in statement.find_all(exp.Anonymous)
                if node.name.lower() in FORBIDDEN_FUNCTIONS
            }
        )
        if forbidden_functions:
            return ValidationResult(
                False,
                f"Forbidden SQL function: {', '.join(forbidden_functions)}",
            )

        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(exp.CTE)
            if cte.alias_or_name
        }
        alias_to_relation: dict[str, str] = {}
        cte_qualifiers = set(cte_names)
        cte_qualifiers.update(
            subquery.alias_or_name.lower()
            for subquery in statement.find_all(exp.Subquery)
            if subquery.alias_or_name
        )
        referenced: set[str] = set()
        for table in statement.find_all(exp.Table):
            relation = table.name.lower()
            if relation in cte_names:
                cte_qualifiers.add((table.alias_or_name or relation).lower())
                continue
            schema_name = table.db.lower() if table.db else ""
            if schema_name and schema_name != "public":
                return ValidationResult(False, f"Only the public schema is allowed: {table.sql()}")
            if relation not in schema.relation_names:
                return ValidationResult(False, f"Unknown table or view: {relation}")
            referenced.add(relation)
            alias_to_relation[(table.alias_or_name or relation).lower()] = relation
            alias_to_relation[relation] = relation

        if not referenced and not list(statement.find_all(exp.CTE)):
            # Scalar SELECTs are read-only and useful (for example SELECT CURRENT_DATE).
            referenced_relations: tuple[str, ...] = ()
        else:
            referenced_relations = tuple(sorted(referenced))

        columns_by_relation = schema.columns_by_relation
        for column in statement.find_all(exp.Column):
            qualifier = column.table.lower() if column.table else ""
            if not qualifier or qualifier in cte_qualifiers:
                continue
            relation = alias_to_relation.get(qualifier)
            if relation is None:
                return ValidationResult(False, f"Unknown table alias: {qualifier}")
            if column.name.lower() not in columns_by_relation[relation]:
                return ValidationResult(
                    False,
                    f"Unknown column {column.name} on relation {relation}",
                )

        normalized = statement.sql(dialect="postgres", pretty=True)
        return ValidationResult(
            True,
            "Query is a single read-only SELECT",
            normalized_sql=normalized,
            referenced_relations=referenced_relations,
        )


def _is_query_root(statement: exp.Expression) -> bool:
    return isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except))


def _contains_comment(sql: str) -> bool:
    """Detect SQL comments outside strings and quoted identifiers."""
    index = 0
    quote: str | None = None
    dollar_tag: str | None = None
    while index < len(sql):
        if dollar_tag is not None:
            if sql.startswith(dollar_tag, index):
                index += len(dollar_tag)
                dollar_tag = None
            else:
                index += 1
            continue

        char = sql[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            elif char == "\\" and quote == "'":
                index += 2
                continue
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "$":
            end = sql.find("$", index + 1)
            if end != -1:
                candidate = sql[index : end + 1]
                if candidate == "$$" or candidate[1:-1].replace("_", "a").isalnum():
                    dollar_tag = candidate
                    index = end + 1
                    continue
        if sql.startswith("--", index) or sql.startswith("/*", index):
            return True
        index += 1
    return False
