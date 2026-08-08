"""Live PostgreSQL schema introspection and compact prompt serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import psycopg2

from src.db_config import DatabaseConfig


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False


@dataclass(frozen=True)
class ForeignKeyInfo:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class RelationInfo:
    name: str
    kind: str
    columns: tuple[ColumnInfo, ...]
    description: str | None = None


@dataclass(frozen=True)
class SchemaSnapshot:
    relations: tuple[RelationInfo, ...]
    foreign_keys: tuple[ForeignKeyInfo, ...] = field(default_factory=tuple)

    @property
    def relation_names(self) -> frozenset[str]:
        return frozenset(relation.name for relation in self.relations)

    @property
    def columns_by_relation(self) -> dict[str, frozenset[str]]:
        return {
            relation.name: frozenset(column.name for column in relation.columns)
            for relation in self.relations
        }

    def to_prompt(self) -> str:
        """Serialize only query-relevant metadata into a compact prompt."""
        blocks: list[str] = []
        for relation in sorted(self.relations, key=lambda item: (item.kind, item.name)):
            label = "VIEW" if relation.kind == "VIEW" else "TABLE"
            column_lines = []
            for column in relation.columns:
                attributes = [column.data_type.upper()]
                if column.primary_key:
                    attributes.append("PRIMARY KEY")
                elif not column.nullable:
                    attributes.append("NOT NULL")
                column_lines.append(f"  {column.name} {' '.join(attributes)}")
            description = f" -- {relation.description}" if relation.description else ""
            blocks.append(
                f"{label} {relation.name}({description}\n"
                + ",\n".join(column_lines)
                + "\n)"
            )

        if self.foreign_keys:
            relationship_lines = [
                f"{fk.source_table}.{fk.source_column} -> "
                f"{fk.target_table}.{fk.target_column}"
                for fk in sorted(
                    self.foreign_keys,
                    key=lambda item: (
                        item.source_table,
                        item.source_column,
                        item.target_table,
                    ),
                )
            ]
            blocks.append("RELATIONSHIPS:\n" + "\n".join(relationship_lines))
        return "\n\n".join(blocks)


ConnectionFactory = Callable[..., "psycopg2.extensions.connection"]


class SchemaManager:
    """Inspect the configured database and cache a schema snapshot per process."""

    def __init__(
        self,
        config: DatabaseConfig | None = None,
        connection_factory: ConnectionFactory = psycopg2.connect,
    ) -> None:
        self._config = config
        self._connection_factory = connection_factory
        self._snapshot: SchemaSnapshot | None = None

    def get_snapshot(self, refresh: bool = False) -> SchemaSnapshot:
        if self._snapshot is None or refresh:
            self._snapshot = self._introspect()
        return self._snapshot

    def get_schema_context(self, refresh: bool = False) -> str:
        return self.get_snapshot(refresh=refresh).to_prompt()

    def _introspect(self) -> SchemaSnapshot:
        config = self._config or DatabaseConfig.from_env()
        with self._connection_factory(**config.as_connect_kwargs()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        c.table_name,
                        t.table_type,
                        c.column_name,
                        CASE
                            WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name
                            ELSE c.data_type
                        END AS data_type,
                        c.is_nullable,
                        c.ordinal_position,
                        obj_description(
                            to_regclass(format('%I.%I', c.table_schema, c.table_name)),
                            'pg_class'
                        ) AS description
                    FROM information_schema.columns AS c
                    JOIN information_schema.tables AS t
                      ON t.table_schema = c.table_schema
                     AND t.table_name = c.table_name
                    WHERE c.table_schema = 'public'
                      AND t.table_type IN ('BASE TABLE', 'VIEW')
                    ORDER BY c.table_name, c.ordinal_position
                    """
                )
                column_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT kcu.table_name, kcu.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON kcu.constraint_name = tc.constraint_name
                     AND kcu.constraint_schema = tc.constraint_schema
                    WHERE tc.table_schema = 'public'
                      AND tc.constraint_type = 'PRIMARY KEY'
                    """
                )
                primary_keys = frozenset(cursor.fetchall())

                cursor.execute(
                    """
                    SELECT
                        source_usage.table_name,
                        source_usage.column_name,
                        target_usage.table_name,
                        target_usage.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS source_usage
                      ON source_usage.constraint_name = tc.constraint_name
                     AND source_usage.constraint_schema = tc.constraint_schema
                    JOIN information_schema.referential_constraints AS rc
                      ON rc.constraint_name = tc.constraint_name
                     AND rc.constraint_schema = tc.constraint_schema
                    JOIN information_schema.key_column_usage AS target_usage
                      ON target_usage.constraint_name = rc.unique_constraint_name
                     AND target_usage.constraint_schema = rc.unique_constraint_schema
                     AND target_usage.ordinal_position = source_usage.position_in_unique_constraint
                    WHERE tc.table_schema = 'public'
                      AND tc.constraint_type = 'FOREIGN KEY'
                    ORDER BY source_usage.table_name, source_usage.ordinal_position
                    """
                )
                foreign_key_rows = cursor.fetchall()

        grouped: dict[str, dict[str, object]] = {}
        for table, table_type, column, data_type, nullable, _, description in column_rows:
            relation = grouped.setdefault(
                table,
                {
                    "kind": "VIEW" if table_type == "VIEW" else "TABLE",
                    "description": description,
                    "columns": [],
                },
            )
            columns = relation["columns"]
            assert isinstance(columns, list)
            columns.append(
                ColumnInfo(
                    name=column,
                    data_type=data_type,
                    nullable=nullable == "YES",
                    primary_key=(table, column) in primary_keys,
                )
            )

        relations = tuple(
            RelationInfo(
                name=name,
                kind=str(values["kind"]),
                description=(str(values["description"]) if values["description"] else None),
                columns=tuple(_columns(values["columns"])),
            )
            for name, values in sorted(grouped.items())
        )
        foreign_keys = tuple(ForeignKeyInfo(*row) for row in foreign_key_rows)
        return SchemaSnapshot(relations=relations, foreign_keys=foreign_keys)


def _columns(value: object) -> Iterable[ColumnInfo]:
    if not isinstance(value, list) or not all(isinstance(item, ColumnInfo) for item in value):
        raise TypeError("Invalid introspected column collection")
    return value
