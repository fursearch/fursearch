"""Identification tracking database for measuring real-world performance."""

import json
import os
import sqlite3
import threading
from typing import Optional

from sam3_fursearch.config import Config
from sam3_fursearch.storage.database import retry_on_locked, get_git_version


class IdentificationTracker:
    """Tracks identification requests, matches, and user feedback."""

    _TIMEOUT = 10.0
    _BUSY_TIMEOUT_MS = 15000

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(Config.BASE_DIR, "tracking.db")
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=self._TIMEOUT)
            conn.execute(f"PRAGMA busy_timeout = {self._BUSY_TIMEOUT_MS}")
            self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_database(self):
        conn = self._connect()
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER,
                telegram_username TEXT,
                telegram_chat_id INTEGER,
                telegram_message_id INTEGER,
                image_path TEXT,
                image_width INTEGER,
                image_height INTEGER,
                num_segments INTEGER,
                num_datasets INTEGER,
                dataset_names TEXT,
                git_version TEXT,
                segmentor_model TEXT,
                segmentor_concept TEXT,
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                segment_index INTEGER,
                segment_bbox TEXT,
                segment_confidence REAL,
                dataset_name TEXT,
                embedder TEXT,
                merge_strategy TEXT,
                character_name TEXT,
                match_confidence REAL,
                match_distance REAL,
                matched_post_id TEXT,
                matched_source TEXT,
                rank_in_dataset INTEGER,
                rank_after_merge INTEGER,
                FOREIGN KEY (request_id) REFERENCES identification_requests(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS identification_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                segment_index INTEGER,
                character_name TEXT,
                is_correct INTEGER,
                correct_character TEXT,
                telegram_user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES identification_requests(id)
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_matches_request ON identification_matches(request_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request ON identification_feedback(request_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_user ON identification_requests(telegram_user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_chat ON identification_requests(telegram_chat_id)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_user_id INTEGER,
                telegram_chat_id INTEGER NOT NULL,
                telegram_username TEXT,
                message TEXT NOT NULL,
                admin_message_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_admin_msg ON user_feedback_messages(admin_message_id)")

        # Tracks every subsequent bot message in a feedback conversation (after the first)
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback_thread (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                admin_message_id INTEGER,
                user_message_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES user_feedback_messages(id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_thread_admin ON feedback_thread(admin_message_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_thread_user ON feedback_thread(user_message_id)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    @retry_on_locked()
    def log_request(
        self,
        telegram_user_id: Optional[int] = None,
        telegram_username: Optional[str] = None,
        telegram_chat_id: Optional[int] = None,
        telegram_message_id: Optional[int] = None,
        image_path: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        num_segments: int = 0,
        num_datasets: int = 0,
        dataset_names: Optional[str] = None,
        git_version: Optional[str] = None,
        segmentor_model: Optional[str] = None,
        segmentor_concept: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> int:
        """Log an identification request. Returns the request_id."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """INSERT INTO identification_requests
            (telegram_user_id, telegram_username, telegram_chat_id, telegram_message_id,
             image_path, image_width, image_height, num_segments, num_datasets,
             dataset_names, git_version, segmentor_model, segmentor_concept, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                telegram_user_id, telegram_username, telegram_chat_id, telegram_message_id,
                image_path, image_width, image_height, num_segments, num_datasets,
                dataset_names, git_version or get_git_version(), segmentor_model,
                segmentor_concept, processing_time_ms,
            ),
        )
        conn.commit()
        return c.lastrowid

    @retry_on_locked()
    def log_matches(self, request_id: int, matches_data: list[dict]):
        """Batch insert match records for a request."""
        if not matches_data:
            return
        conn = self._connect()
        c = conn.cursor()
        c.executemany(
            """INSERT INTO identification_matches
            (request_id, segment_index, segment_bbox, segment_confidence,
             dataset_name, embedder, merge_strategy, character_name,
             match_confidence, match_distance, matched_post_id, matched_source,
             rank_in_dataset, rank_after_merge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    request_id,
                    m.get("segment_index"),
                    json.dumps(m["segment_bbox"]) if m.get("segment_bbox") else None,
                    m.get("segment_confidence"),
                    m.get("dataset_name"),
                    m.get("embedder"),
                    m.get("merge_strategy"),
                    m.get("character_name"),
                    m.get("match_confidence"),
                    m.get("match_distance"),
                    m.get("matched_post_id"),
                    m.get("matched_source"),
                    m.get("rank_in_dataset"),
                    m.get("rank_after_merge"),
                )
                for m in matches_data
            ],
        )
        conn.commit()

    @retry_on_locked()
    def add_feedback(
        self,
        request_id: int,
        segment_index: int,
        character_name: str,
        is_correct: Optional[bool],
        correct_character: Optional[str] = None,
        telegram_user_id: Optional[int] = None,
    ):
        """Record user feedback on a match."""
        conn = self._connect()
        c = conn.cursor()
        is_correct_int = None if is_correct is None else (1 if is_correct else 0)
        c.execute(
            """INSERT INTO identification_feedback
            (request_id, segment_index, character_name, is_correct,
             correct_character, telegram_user_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, segment_index, character_name, is_correct_int,
             correct_character, telegram_user_id),
        )
        conn.commit()
        return c.lastrowid

    @retry_on_locked()
    def set_setting(self, key: str, value: str):
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, value),
        )
        conn.commit()

    def get_setting(self, key: str) -> Optional[str]:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = c.fetchone()
        return row[0] if row else None

    @retry_on_locked()
    def log_user_feedback(
        self,
        telegram_user_id: Optional[int],
        telegram_chat_id: int,
        telegram_username: Optional[str],
        message: str,
    ) -> int:
        """Store a user feedback message. Returns the row id."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """INSERT INTO user_feedback_messages
            (telegram_user_id, telegram_chat_id, telegram_username, message)
            VALUES (?, ?, ?, ?)""",
            (telegram_user_id, telegram_chat_id, telegram_username, message),
        )
        conn.commit()
        return c.lastrowid

    @retry_on_locked()
    def set_feedback_admin_message_id(self, feedback_id: int, admin_message_id: int):
        """Record which admin-chat message_id corresponds to a feedback entry."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "UPDATE user_feedback_messages SET admin_message_id = ? WHERE id = ?",
            (admin_message_id, feedback_id),
        )
        conn.commit()

    @retry_on_locked()
    def add_feedback_thread_message(
        self,
        feedback_id: int,
        admin_message_id: Optional[int] = None,
        user_message_id: Optional[int] = None,
    ):
        """Record a bot message sent in either side of a feedback conversation."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            "INSERT INTO feedback_thread (feedback_id, admin_message_id, user_message_id) VALUES (?, ?, ?)",
            (feedback_id, admin_message_id, user_message_id),
        )
        conn.commit()

    def _feedback_row_to_dict(self, row) -> dict:
        return {
            "id": row[0],
            "telegram_user_id": row[1],
            "telegram_chat_id": row[2],
            "telegram_username": row[3],
            "message": row[4],
            "admin_message_id": row[5],
        }

    def get_feedback_by_admin_message_id(self, admin_message_id: int) -> Optional[dict]:
        """Return the feedback row for any admin-side message in the conversation, or None."""
        conn = self._connect()
        c = conn.cursor()
        # Check original notification
        c.execute(
            """SELECT id, telegram_user_id, telegram_chat_id, telegram_username, message, admin_message_id
            FROM user_feedback_messages WHERE admin_message_id = ?""",
            (admin_message_id,),
        )
        row = c.fetchone()
        if row:
            return self._feedback_row_to_dict(row)
        # Check follow-up thread messages
        c.execute(
            """SELECT ufm.id, ufm.telegram_user_id, ufm.telegram_chat_id, ufm.telegram_username,
                      ufm.message, ufm.admin_message_id
               FROM feedback_thread ft
               JOIN user_feedback_messages ufm ON ufm.id = ft.feedback_id
               WHERE ft.admin_message_id = ?""",
            (admin_message_id,),
        )
        row = c.fetchone()
        return self._feedback_row_to_dict(row) if row else None

    def get_feedback_by_user_message_id(self, user_chat_id: int, user_message_id: int) -> Optional[dict]:
        """Return the feedback row for a bot message sent to the user's chat, or None."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """SELECT ufm.id, ufm.telegram_user_id, ufm.telegram_chat_id, ufm.telegram_username,
                      ufm.message, ufm.admin_message_id
               FROM feedback_thread ft
               JOIN user_feedback_messages ufm ON ufm.id = ft.feedback_id
               WHERE ft.user_message_id = ? AND ufm.telegram_chat_id = ?""",
            (user_message_id, user_chat_id),
        )
        row = c.fetchone()
        return self._feedback_row_to_dict(row) if row else None

    def get_stats(self) -> dict:
        """Get tracking statistics: request counts, unique users, hit rates, avg time."""
        conn = self._connect()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM identification_requests")
        total_requests = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT telegram_user_id) FROM identification_requests WHERE telegram_user_id IS NOT NULL")
        unique_users = c.fetchone()[0]

        c.execute("SELECT AVG(processing_time_ms) FROM identification_requests WHERE processing_time_ms IS NOT NULL")
        avg_time = c.fetchone()[0]

        c.execute("SELECT SUM(num_segments) FROM identification_requests")
        total_segments = c.fetchone()[0] or 0

        # Hit rate: requests that found at least one match
        c.execute("""
            SELECT COUNT(DISTINCT r.id) FROM identification_requests r
            JOIN identification_matches m ON m.request_id = r.id
            WHERE m.match_confidence > 0
        """)
        requests_with_matches = c.fetchone()[0]

        # Per-dataset breakdown: matches served, unique requests hit, top-1 hit rate
        # top-1 means the match was rank_after_merge == 1 (the result the user actually sees)
        c.execute("""
            SELECT m.dataset_name,
                   COUNT(*) as total_matches,
                   COUNT(DISTINCT m.request_id) as requests_with_hits,
                   SUM(CASE WHEN m.rank_after_merge = 1 THEN 1 ELSE 0 END) as top1_hits
            FROM identification_matches m
            WHERE m.dataset_name IS NOT NULL
            GROUP BY m.dataset_name
        """)
        per_dataset = {}
        for name, total_matches, requests_hit, top1_hits in c.fetchall():
            per_dataset[name] = {
                "total_matches": total_matches,
                "requests_with_hits": requests_hit,
                "hit_rate": f"{requests_hit / total_requests:.1%}" if total_requests > 0 else "N/A",
                "top1_hits": top1_hits,
                "top1_rate": f"{top1_hits / total_requests:.1%}" if total_requests > 0 else "N/A",
            }

        # Feedback stats
        c.execute("SELECT COUNT(*) FROM identification_feedback")
        total_feedback = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM identification_feedback WHERE is_correct = 1")
        correct_feedback = c.fetchone()[0]

        return {
            "total_requests": total_requests,
            "unique_users": unique_users,
            "total_segments_found": total_segments,
            "avg_processing_time_ms": round(avg_time) if avg_time else None,
            "requests_with_matches": requests_with_matches,
            "hit_rate": f"{requests_with_matches / total_requests:.1%}" if total_requests > 0 else "N/A",
            "per_dataset": per_dataset,
            "total_feedback": total_feedback,
            "correct_feedback": correct_feedback,
        }
