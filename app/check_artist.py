# check_artists.py
from main import collection

# 1. 메타데이터에서 artist 필드만 500개 가져와서 어떤 값들이 있는지 확인
results = collection.get(limit=500, include=['metadatas'])

artists = {}
for meta in results['metadatas']:
    a = meta.get('artist', 'Unknown')
    artists[a] = artists.get(a, 0) + 1

print("📊 DB 내 주요 가수 분포 (상위 20개 샘플):")
print("-" * 40)
sorted_artists = sorted(artists.items(), key=lambda x: x[1], reverse=True)
for name, count in sorted_artists[:20]:
    print(f"- {name}: {count}곡")

print("-" * 40)
# 2. 'Imagine'이라는 글자가 포함된 메타데이터가 단 하나라도 있는지 확인
print("❓ 'Imagine' 키워드 포함 여부 확인 중...")
all_data = collection.get()
found_raw = [m.get('artist') for m in all_data['metadatas'] if 'Imagine' in str(m.get('artist', ''))]

if found_raw:
    print(f"✅ 발견된 유사 이름: {list(set(found_raw))}")
else:
    print("❌ 'Imagine'이 포함된 가수가 단 한 명도 없습니다.")
