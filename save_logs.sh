#!/bin/bash
cd /mnt/volume/fursearch/logs
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

LATEST_FILE=$(ls -1t . | grep fursearch-tgbot- | head -n1)
SINCE=$(date -d @"$(stat -c %Y "$LATEST_FILE")" '+%Y-%m-%d %H:%M:%S')
echo "$LATEST_FILE ($SINCE)"
journalctl -u fursearch-tgbot.service --since "$SINCE" | grep -v "\-\- No entries \-\-" > fursearch-tgbot-$TIMESTAMP.log
