#!/bin/bash

# 1. 설정
PROJECT_DIR="/home/now0930/telegram_music_channel"
BACKUP_DIR="$PROJECT_DIR/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="db_backup_$TIMESTAMP.tar.gz"

# 2. 백업 디렉토리 생성
mkdir -p "$BACKUP_DIR"

# 3. 데이터베이스 백업 수행
# app/ 폴더 내부의 핵심 DB 파일들을 압축합니다.
cd "$PROJECT_DIR"
tar -czf "$BACKUP_DIR/$BACKUP_NAME" \
    app/music_vector_db \
    app/chroma.sqlite3 \
    app/bot_data.db

# 4. 오래된 백업 삭제 (최근 7일분만 유지)
#find "$BACKUP_DIR" -name "db_backup_*.tar.gz" -mtime +7 -delete

echo "[$TIMESTAMP] 백업 완료: $BACKUP_NAME"
