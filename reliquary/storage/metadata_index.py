from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MetadataIndex(ABC):
    @abstractmethod
    def record_artifact(
        self,
        artifact: dict[str, Any],
        *,
        local_path: str | None = None,
        storage_kind: str = "local",
        storage_uri: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_records(
        self,
        artifact_type: str,
        *,
        window_id: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_record(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def latest_policy_record(self) -> dict[str, Any] | None:
        raise NotImplementedError


class SqliteMetadataIndex(MetadataIndex):
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    producer_role TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    window_id INTEGER NOT NULL,
                    commodity_use TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    local_path TEXT,
                    storage_kind TEXT NOT NULL,
                    storage_uri TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_type_window ON artifacts(artifact_type, window_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_policy ON artifacts(policy_id, window_id)"
            )

    def record_artifact(
        self,
        artifact: dict[str, Any],
        *,
        local_path: str | None = None,
        storage_kind: str = "local",
        storage_uri: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_type, lane, producer_id, producer_role,
                    policy_id, window_id, commodity_use, created_at,
                    local_path, storage_kind, storage_uri
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    local_path = excluded.local_path,
                    storage_kind = excluded.storage_kind,
                    storage_uri = excluded.storage_uri
                """,
                (
                    artifact["artifact_id"],
                    artifact["artifact_type"],
                    artifact.get("lane", "shared"),
                    artifact["producer_id"],
                    artifact.get("producer_role", "unknown"),
                    artifact["policy_id"],
                    artifact["window_id"],
                    artifact.get("commodity_use", "dual_use"),
                    artifact["created_at"],
                    local_path,
                    storage_kind,
                    storage_uri,
                ),
            )

    def list_records(
        self,
        artifact_type: str,
        *,
        window_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts WHERE artifact_type = ?"
        params: list[Any] = [artifact_type]
        if window_id is not None:
            query += " AND window_id = ?"
            params.append(window_id)
        query += " ORDER BY window_id ASC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_record(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_type = ? AND artifact_id = ?",
                (artifact_type, artifact_id),
            ).fetchone()
        return dict(row) if row else None

    def latest_policy_record(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM artifacts
                WHERE artifact_type = 'policy'
                ORDER BY window_id DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None


class PostgresMetadataIndex(MetadataIndex):
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._init_db()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def _init_db(self) -> None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    producer_role TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    window_id INTEGER NOT NULL,
                    commodity_use TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    local_path TEXT,
                    storage_kind TEXT NOT NULL,
                    storage_uri TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_type_window ON artifacts(artifact_type, window_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_policy ON artifacts(policy_id, window_id)"
            )
            conn.commit()

    def record_artifact(
        self,
        artifact: dict[str, Any],
        *,
        local_path: str | None = None,
        storage_kind: str = "local",
        storage_uri: str | None = None,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_type, lane, producer_id, producer_role,
                    policy_id, window_id, commodity_use, created_at,
                    local_path, storage_kind, storage_uri
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (artifact_id) DO UPDATE SET
                    local_path = EXCLUDED.local_path,
                    storage_kind = EXCLUDED.storage_kind,
                    storage_uri = EXCLUDED.storage_uri
                """,
                (
                    artifact["artifact_id"],
                    artifact["artifact_type"],
                    artifact.get("lane", "shared"),
                    artifact["producer_id"],
                    artifact.get("producer_role", "unknown"),
                    artifact["policy_id"],
                    artifact["window_id"],
                    artifact.get("commodity_use", "dual_use"),
                    artifact["created_at"],
                    local_path,
                    storage_kind,
                    storage_uri,
                ),
            )
            conn.commit()

    def list_records(
        self,
        artifact_type: str,
        *,
        window_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM artifacts
            WHERE artifact_type = %s
        """
        params: list[Any] = [artifact_type]
        if window_id is not None:
            query += " AND window_id = %s"
            params.append(window_id)
        query += " ORDER BY window_id ASC, created_at ASC"
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    def get_record(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM artifacts WHERE artifact_type = %s AND artifact_id = %s",
                (artifact_type, artifact_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [column.name for column in cursor.description]
        return dict(zip(columns, row, strict=False))

    def latest_policy_record(self) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM artifacts
                WHERE artifact_type = 'policy'
                ORDER BY window_id DESC, created_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [column.name for column in cursor.description]
        return dict(zip(columns, row, strict=False))
