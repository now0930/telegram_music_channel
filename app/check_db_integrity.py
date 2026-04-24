import chromadb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_db_integrity")

DB_PATH = "./music_vector_db"
COLLECTION_NAME = "music_library"

def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    total = collection.count()
    logger.info("현재 DB 레코드 수: %d", total)
    if total == 0:
        logger.info("DB 가 비어있습니다.")
        return

    # 전수 조사
    results = collection.get(include=["metadatas"])
    ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])

    # 1. 중복 ID 검사
    unique_ids = set()
    duplicate_ids = set()
    for id_ in ids:
        if id_ in unique_ids:
            duplicate_ids.add(id_)
        else:
            unique_ids.add(id_)
            
    if duplicate_ids:
        logger.warning("중복된 ID 발견: %d 개 - 삭제 처리: %s", len(duplicate_ids), list(duplicate_ids)[:10])
        collection.delete(ids=list(duplicate_ids))

    # 2. ID 누락 및 메타데이터/타입 검사
    invalid_ids = []
    for id_, meta in zip(ids, metadatas):
        if not id_:
            invalid_ids.append(id_)
            continue
            
        if not meta:
            invalid_ids.append(id_)
            continue
            
        # 연도(year) 값이 정수가 아닌 경우 무결성 위반 처리
        year = meta.get("year")
        if year is not None:
            try:
                int(year)
            except (ValueError, TypeError):
                logger.warning("잘못된 year 데이터 타입 (ID: %s, year: %s)", id_, year)
                invalid_ids.append(id_)

    if invalid_ids:
        # 빈 문자열 ID 등은 삭제할 수 없으므로 유효한 ID 만 필터링하여 삭제
        valid_invalid_ids = [id_ for id_ in invalid_ids if id_]
        logger.warning("유효하지 않거나 깨진 데이터 발견: %d 개. 삭제를 진행합니다...", len(valid_invalid_ids))
        if valid_invalid_ids:
            collection.delete(ids=valid_invalid_ids)
    else:
        logger.info("모든 데이터가 유효합니다. 무결성 검사 통과!")

    logger.info("작업 완료. 최종 DB 레코드 수: %d", collection.count())

if __name__ == "__main__":
    main()
