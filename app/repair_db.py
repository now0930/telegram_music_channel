import chromadb

# ChromaDB 연결
client = chromadb.PersistentClient(path="./music_vector_db")
collection = client.get_collection("music_library")

# 1. 모든 ID를 가져와서 검증
all_data = collection.get(include=['metadatas'])
ids = all_data['ids']
metadatas = all_data['metadatas']

print(f"총 {len(ids)}개의 항목 확인 중...")

# 2. 데이터가 비정상적인 경우 삭제 (필요시)
invalid_ids = []
for i, id_val in enumerate(ids):
    if not id_val or id_val == "None":
        invalid_ids.append(id_val)

if invalid_ids:
    print(f"{len(invalid_ids)}개의 잘못된 ID 발견: 삭제 시도")
    collection.delete(ids=invalid_ids)
    print("삭제 완료.")
else:
    print("ID 무결성 확인 완료. 문제가 없습니다.")
