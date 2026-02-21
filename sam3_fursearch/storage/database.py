import math
import os
import sys
import sqlite3
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, wraps
from typing import Optional

from sam3_fursearch.config import Config


SOURCE_NFC25 = "nfc25"
SOURCE_NFC26 = "nfc26"
SOURCE_MANUAL = "manual"
SOURCE_TGBOT = "tgbot"

SOURCES_AVAILABLE = [SOURCE_NFC25, SOURCE_NFC26, SOURCE_MANUAL, SOURCE_TGBOT]


def bucketize(values: dict | Counter, max_buckets: int = 8) -> dict[str, int]:
    """Group a {value: count} mapping into dynamic histogram buckets.

    If there are few distinct values, keeps them exact.
    Otherwise creates ~max_buckets log-scaled ranges.
    """
    if not values:
        return {}
    sorted_vals = sorted(values.items())
    if len(sorted_vals) <= max_buckets:
        return {str(v): c for v, c in sorted_vals}
    min_val, max_val = sorted_vals[0][0], sorted_vals[-1][0]
    log_min = math.log1p(min_val)
    log_max = math.log1p(max_val)
    step = (log_max - log_min) / max_buckets
    boundaries = []
    for i in range(1, max_buckets):
        b = int(math.expm1(log_min + step * i))
        if not boundaries or b > boundaries[-1]:
            boundaries.append(b)
    buckets = {}
    bi = 0
    for val, count in sorted_vals:
        while bi < len(boundaries) and val > boundaries[bi]:
            bi += 1
        lo = min_val if bi == 0 else boundaries[bi - 1] + 1
        hi = boundaries[bi] if bi < len(boundaries) else max_val
        key = str(lo) if lo == hi else f"{lo}-{hi}"
        buckets[key] = buckets.get(key, 0) + count
    return buckets


def retry_on_locked(max_retries: int = 8, base_delay: float = 0.2):
    """Retry on 'database is locked' with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        last_error = e
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


@lru_cache(maxsize=1)
def get_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Config.BASE_DIR,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


_NFC_YEAR = {
    SOURCE_NFC25: "2025",
    SOURCE_NFC26: "2026",
}


def get_source_url(source: Optional[str], post_id: str, character_name: Optional[str] = None) -> Optional[str]:
    """Generate a URL for a post based on its source."""
    if source in _NFC_YEAR:
        return f"https://blob.nordicfuzzcon.org/user-content/{_NFC_YEAR[source]}/fursuit-badge/{post_id}.png"
    # TODO: this will not work for all characters since the name will not match furtrack all the time
    if character_name:
        return f"https://furtrack.com/index/{character_name.replace(' ', '-').lower()}"
    return None


def get_source_image_url(source: Optional[str], post_id: str) -> Optional[str]:
    """Generate a direct image URL for a post based on its source (thumbnail-sized for sending)."""
    if source in _NFC_YEAR:
        return f"https://blob.nordicfuzzcon.org/user-content/{_NFC_YEAR[source]}/fursuit-badge/{post_id}-300.webp"
    return None


@dataclass
class Detection:
    id: Optional[int]
    post_id: str
    character_name: Optional[str]
    embedding_id: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    confidence: float
    segmentor_model: str = "unknown"
    created_at: Optional[datetime] = None
    source: Optional[str] = None
    uploaded_by: Optional[str] = None
    source_filename: Optional[str] = None
    preprocessing_info: Optional[str] = None
    git_version: Optional[str] = None


class Database:
    _SELECT_FIELDS = """
        id, post_id, character_name, embedding_id, bbox_x, bbox_y,
        bbox_width, bbox_height, confidence, segmentor_model, created_at,
        source, uploaded_by, source_filename, preprocessing_info, git_version
    """
    _TIMEOUT = 10.0
    _BUSY_TIMEOUT_MS = 15000

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
        print(f"DB initialized: {db_path}", file=sys.stderr)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=self._TIMEOUT)
            conn.execute(f"PRAGMA busy_timeout = {self._BUSY_TIMEOUT_MS}")
            self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_database(self):
        conn = self._connect()
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                character_name TEXT,
                embedding_id INTEGER UNIQUE NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                confidence REAL DEFAULT 0.0,
                segmentor_model TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_filename TEXT,
                preprocessing_info TEXT
            )
        """)
        # Migrate columns first (before creating indexes that depend on them)
        c.execute("PRAGMA table_info(detections)")
        existing_columns = {row[1] for row in c.fetchall()}
        new_columns = [
            ("source", "TEXT"),
            ("uploaded_by", "TEXT"),
            ("source_filename", "TEXT"),
            ("preprocessing_info", "TEXT"),
            ("git_version", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                c.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")

        # Create indexes (after columns exist)
        c.execute("CREATE INDEX IF NOT EXISTS idx_post_id ON detections(post_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_character_name ON detections(character_name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_embedding_id ON detections(embedding_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_post_preproc ON detections(post_id, preprocessing_info)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_source ON detections(source)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_source_post_preproc ON detections(source, post_id, preprocessing_info)")

        # Metadata table (key-value store for dataset-level config)
        c.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Submission metadata table (telegram-specific per-post metadata)
        c.execute("""
            CREATE TABLE IF NOT EXISTS submission_metadata (
                post_id TEXT PRIMARY KEY,
                character_url TEXT,
                submitted_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    @staticmethod
    def read_metadata_lightweight(db_path: str, key: str) -> Optional[str]:
        """Read a metadata value without full Database initialization.

        Useful when you need to check metadata (e.g. embedder) before deciding
        whether to open the full Database. Returns None if DB/table/key doesn't exist.
        """
        if not os.path.exists(db_path):
            return None
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
            if not cursor.fetchone():
                conn.close()
                return None
            cursor = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    @retry_on_locked()
    def get_metadata(self, key: str) -> Optional[str]:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = c.fetchone()
        return row[0] if row else None

    @retry_on_locked()
    def set_metadata(self, key: str, value: str):
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

    _INSERT_SQL = """
        INSERT INTO detections
        (post_id, character_name, embedding_id, bbox_x, bbox_y, bbox_width, bbox_height,
         confidence, segmentor_model, source, uploaded_by, source_filename,
         preprocessing_info, git_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def _detection_to_tuple(self, d: Detection) -> tuple:
        return (
            d.post_id, d.character_name, d.embedding_id,
            d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
            d.confidence, d.segmentor_model, d.source, d.uploaded_by,
            d.source_filename, d.preprocessing_info,
            d.git_version or get_git_version(),
        )

    @retry_on_locked()
    def add_detections_batch(self, detections: list[Detection]) -> list[int]:
        if not detections:
            return []
        conn = self._connect()
        c = conn.cursor()
        row_ids = []
        for d in detections:
            c.execute(self._INSERT_SQL, self._detection_to_tuple(d))
            row_ids.append(c.lastrowid)
        conn.commit()
        return row_ids

    @retry_on_locked()
    def add_detection(self, detection: Detection) -> int:
        return self.add_detections_batch([detection])[0]

    def _row_to_detection(self, row) -> Detection:
        return Detection(
            id=row[0],
            post_id=row[1],
            character_name=row[2],
            embedding_id=row[3],
            bbox_x=row[4],
            bbox_y=row[5],
            bbox_width=row[6],
            bbox_height=row[7],
            confidence=row[8],
            segmentor_model=row[9],
            created_at=row[10],
            source=row[11] if len(row) > 11 else None,
            uploaded_by=row[12] if len(row) > 12 else None,
            source_filename=row[13] if len(row) > 13 else None,
            preprocessing_info=row[14] if len(row) > 14 else None,
            git_version=row[15] if len(row) > 15 else None,
        )

    @retry_on_locked()
    def get_detection_by_embedding_id(self, embedding_id: int) -> Optional[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE embedding_id = ?", (embedding_id,))
        row = c.fetchone()
        return self._row_to_detection(row) if row else None

    @retry_on_locked()
    def get_detection_by_id(self, detection_id: int) -> Optional[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE id = ?", (detection_id,))
        row = c.fetchone()
        return self._row_to_detection(row) if row else None

    @retry_on_locked()
    def get_detections_by_post_id(self, post_id: str) -> list[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE post_id = ?", (post_id,))
        rows = c.fetchall()
        return [self._row_to_detection(row) for row in rows]

    @retry_on_locked()
    def get_detections_by_character(self, character_name: str) -> list[Detection]:
        conn = self._connect()
        c = conn.cursor()
        c.execute(f"SELECT {self._SELECT_FIELDS} FROM detections WHERE character_name = ?", (character_name,))
        rows = c.fetchall()
        return [self._row_to_detection(row) for row in rows]

    @retry_on_locked()
    def get_all_character_names(self) -> list[str]:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT DISTINCT character_name FROM detections WHERE character_name IS NOT NULL ORDER BY character_name")
        return [row[0] for row in c.fetchall()]

    @retry_on_locked()
    def get_all_post_ids(self) -> list[str]:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT DISTINCT post_id FROM detections")
        return [row[0] for row in c.fetchall()]

    @retry_on_locked()
    def get_stats(self):
        conn = self._connect()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM detections")
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT character_name) FROM detections WHERE character_name IS NOT NULL")
        unique_chars = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT post_id) FROM detections")
        unique_posts = c.fetchone()[0]

        c.execute("""
            SELECT character_name, COUNT(*) as count FROM detections
            WHERE character_name IS NOT NULL
            GROUP BY character_name ORDER BY count DESC LIMIT 10
        """)
        top_chars = c.fetchall()

        c.execute("""
            SELECT segmentor_model, COUNT(*) as count FROM detections
            GROUP BY segmentor_model ORDER BY count DESC
        """)
        segmentor_breakdown = dict(c.fetchall())

        c.execute("""
            SELECT preprocessing_info, COUNT(*) as count FROM detections
            WHERE preprocessing_info IS NOT NULL
            GROUP BY preprocessing_info ORDER BY count DESC LIMIT 10
        """)
        preprocessing_breakdown = dict(c.fetchall())

        c.execute("""
            SELECT git_version, COUNT(*) as count FROM detections
            GROUP BY git_version ORDER BY count DESC LIMIT 10
        """)
        git_version_breakdown = dict(c.fetchall())

        c.execute("""
            SELECT source, COUNT(*) as count FROM detections
            GROUP BY source ORDER BY count DESC
        """)
        source_breakdown = dict(c.fetchall())

        return {
            "total_detections": total,
            "unique_characters": unique_chars,
            "unique_posts": unique_posts,
            "top_characters": top_chars,
            "segmentor_breakdown": segmentor_breakdown,
            "preprocessing_breakdown": preprocessing_breakdown,
            "git_version_breakdown": git_version_breakdown,
            "source_breakdown": source_breakdown,
        }

    def get_character_post_counts(self) -> dict[str, int]:
        """Return {character_name: num_distinct_posts} for all characters."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT character_name, COUNT(DISTINCT post_id) as cnt
            FROM detections WHERE character_name IS NOT NULL
            GROUP BY character_name
        """)
        return dict(c.fetchall())

    def get_post_segment_counts(self) -> dict[str, int]:
        """Return {post_id: num_segments} for all posts."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT post_id, COUNT(*) as cnt
            FROM detections GROUP BY post_id
        """)
        return dict(c.fetchall())

    @retry_on_locked()
    def has_post(self, post_id: str) -> bool:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT 1 FROM detections WHERE post_id = ? LIMIT 1", (post_id,))
        exists = c.fetchone() is not None
        return exists

    @retry_on_locked()
    def get_posts_needing_update(
        self, post_ids: list[str], preprocessing_info: str, source: str
    ) -> set[str]:
        """Return post_ids not yet processed with given preprocessing_info and source."""
        if not post_ids or not preprocessing_info:
            return set(post_ids)
        conn = self._connect()
        c = conn.cursor()
        already_processed = set()
        # SQLite has a variable limit (default 999), batch to stay under it
        batch_size = 900
        for i in range(0, len(post_ids), batch_size):
            batch = post_ids[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            c.execute(
                f"SELECT DISTINCT post_id FROM detections WHERE post_id IN ({placeholders}) AND preprocessing_info = ? AND source = ?",
                (*batch, preprocessing_info, source)
            )
            already_processed.update(row[0] for row in c.fetchall())

        return set(post_ids) - already_processed

    @retry_on_locked()
    def get_next_embedding_id(self) -> int:
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT MAX(embedding_id) FROM detections")
        result = c.fetchone()[0]
        return 0 if result is None else result + 1

    @retry_on_locked()
    def delete_orphaned_detections(self, max_valid_embedding_id: int) -> int:
        """Delete detections with embedding_id > max_valid_embedding_id."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM detections WHERE embedding_id > ?", (max_valid_embedding_id,))
        deleted = c.rowcount
        conn.commit()
        return deleted

    # --- Submission metadata (per-post metadata like character URLs) ---

    @retry_on_locked()
    def set_submission_metadata(self, post_id: str, character_url: Optional[str] = None,
                                submitted_by: Optional[str] = None):
        """Upsert submission metadata for a post."""
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO submission_metadata (post_id, character_url, submitted_by)
               VALUES (?, ?, ?)""",
            (post_id, character_url, submitted_by),
        )
        conn.commit()

    @retry_on_locked()
    def get_submission_metadata(self, post_id: str) -> Optional[dict]:
        """Get submission metadata for a post. Returns dict or None."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT post_id, character_url, submitted_by FROM submission_metadata WHERE post_id = ?",
                  (post_id,))
        row = c.fetchone()
        if not row:
            return None
        return {"post_id": row[0], "character_url": row[1], "submitted_by": row[2]}

    @retry_on_locked()
    def get_all_submission_metadata(self) -> dict[str, dict]:
        """Load all submission metadata. Returns {post_id: {character_url, submitted_by}}."""
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT post_id, character_url, submitted_by FROM submission_metadata")
        result = {}
        for row in c.fetchall():
            result[row[0]] = {"character_url": row[1], "submitted_by": row[2]}
        return result
