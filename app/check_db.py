# check_db.py
from main import collection

# 딱 5개만 뽑아서 메타데이터 구조 확인
results = collection.get(limit=5)

print("🔍 DB 샘플 데이터 확인:")
for i in range(len(results['ids'])):
    print(f"ID: {results['ids'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    print("-" * 30)

# 전체 곡 수 확인
print(f"📦 전체 곡 수: {collection.count()}곡")
