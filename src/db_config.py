"""PostgreSQL connection configuration loaded exclusively from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseConfig:
    """Validated PostgreSQL connection settings."""

    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from a local .env file and/or process environment."""
        load_dotenv()
        required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(
                "Missing required database environment variables: "
                + ", ".join(missing)
            )

        port_value = os.environ["DB_PORT"]
        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError("DB_PORT must be an integer") from exc

        return cls(
            host=os.environ["DB_HOST"],
            port=port,
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
        )

    def as_connect_kwargs(self) -> dict[str, str | int]:
        """Return keyword arguments accepted by psycopg2.connect."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
        }
