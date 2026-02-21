#!/usr/bin/env bash
# Checkpoint SQLite WAL files into their .db files.
# Usage: ./checkpoint_db.sh [file.db ...] (defaults to all *.db in current dir)

set -euo pipefail

if [ $# -gt 0 ]; then
    dbs=("$@")
else
    dbs=(*.db)
fi

for db in "${dbs[@]}"; do
    if [ ! -f "$db" ]; then
        echo "skip: $db (not found)"
        continue
    fi
    wal="${db}-wal"
    shm="${db}-shm"
    if [ ! -f "$wal" ]; then
        echo "skip: $db (no WAL)"
        continue
    fi
    wal_size=$(stat --printf='%s' "$wal" 2>/dev/null || stat -f '%z' "$wal")
    sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);"
    echo "done: $db (WAL was ${wal_size} bytes)"
done
