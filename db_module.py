from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


@dataclass(frozen=True)
class UserMatch:
    id: int
    name: str
    distance: float
    first_seen: datetime
    last_seen: datetime


class MemoryDatabase:
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
    ) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn: Optional[psycopg.Connection[Any]] = None

    @classmethod
    def from_env(cls) -> "MemoryDatabase":
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "memory_assistant"),
            user=os.getenv("POSTGRES_USER", "memory_user"),
            password=os.getenv("POSTGRES_PASSWORD", "memory_password"),
        )

    def connect(self) -> "MemoryDatabase":
        if self.conn and not self.conn.closed:
            return self

        self.conn = psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            autocommit=True,
            row_factory=dict_row,
        )
        register_vector(self.conn)
        self.verify_schema()
        return self

    def close(self) -> None:
        if self.conn and not self.conn.closed:
            self.conn.close()

    def verify_schema(self) -> None:
        conn = self._conn()
        row = conn.execute(
            """
            SELECT
                EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS has_vector,
                to_regclass('public.users') AS users_table,
                to_regclass('public.conversations') AS conversations_table
            """
        ).fetchone()
        if not row or not row["has_vector"] or not row["users_table"] or not row["conversations_table"]:
            raise RuntimeError(
                "Database schema is not ready. Start Docker Compose with db/init.sql mounted first."
            )

    def find_matching_user(self, embedding: Sequence[float], threshold: float) -> Optional[UserMatch]:
        vector = self._embedding_vector(embedding)
        row = self._conn().execute(
            """
            SELECT id, name, first_seen, last_seen, face_embedding <=> %s AS distance
            FROM users
            ORDER BY face_embedding <=> %s
            LIMIT 1
            """,
            (vector, vector),
        ).fetchone()

        if not row or row["distance"] is None or float(row["distance"]) > threshold:
            return None

        return UserMatch(
            id=int(row["id"]),
            name=str(row["name"]),
            distance=float(row["distance"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    def create_user(self, name: str, embedding: Sequence[float]) -> int:
        vector = self._embedding_vector(embedding)
        row = self._conn().execute(
            """
            INSERT INTO users (name, face_embedding)
            VALUES (%s, %s)
            RETURNING id
            """,
            (name.strip() or "Unknown", vector),
        ).fetchone()
        if not row:
            raise RuntimeError("Could not create a new user.")
        return int(row["id"])

    def update_last_seen(self, user_id: int) -> None:
        self._conn().execute(
            "UPDATE users SET last_seen = NOW() WHERE id = %s",
            (user_id,),
        )

    def add_conversation(self, user_id: int, transcript: str, summary: str) -> int:
        row = self._conn().execute(
            """
            INSERT INTO conversations (user_id, transcript, summary)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (user_id, transcript.strip(), summary.strip()),
        ).fetchone()
        if not row:
            raise RuntimeError("Could not save the conversation.")
        return int(row["id"])

    def get_latest_conversation(self, user_id: int) -> Optional[dict[str, Any]]:
        return self._conn().execute(
            """
            SELECT id, transcript, summary, created_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

    def get_conversation_history(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT id, transcript, summary, created_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
        return list(rows)

    def _conn(self) -> psycopg.Connection[Any]:
        if not self.conn or self.conn.closed:
            raise RuntimeError("Database connection is not open.")
        return self.conn

    @staticmethod
    def _embedding_vector(embedding: Sequence[float]) -> Vector:
        values = [float(value) for value in embedding]
        if len(values) != 128:
            raise ValueError(f"Expected a 128-dimensional face embedding, got {len(values)}.")
        return Vector(values)
