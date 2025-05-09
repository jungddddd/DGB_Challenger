from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sentence_transformers.util import cos_sim

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

model = SentenceTransformer('all-MiniLM-L6-v2')

input_data = [
    {"text": "20대 여성 대학생", "features": [60, 0, 1]},
    {"text": "30대 남성 회사원", "features": [250, 1, 0]},
    {"text": "50대 여성 자영업", "features": [300, 1, 1]},
]

trend_keywords = {
    "현실적 실수요자": ["아파트구입", "실속주택", "소형아파트", "대출한도", "실수요자", "실거주", "전세탈출", "내집마련",
                 "생애최초", "신혼부부", "청년주택", "실용성", "가격대비성능", "주거안정", "소형평수", "20평대", "1인가구",
                 "자녀양육", "교육환경", "교통편의", "구축아파트", "신축선호", "전세대출", "디딤돌대출", "보금자리론",
                 "중도금대출", "LTV제한", "DSR규제", "고정금리", "금리부담", "주택시장", "매매전환", "전세불안",
                 "주거복지", "실거주요건", "입주대기", "중소형아파트", "무주택자", "청약가점", "주거비절감", "역세권",
                 "생활인프라", "단지환경", "실거주매물", "임대차보호", "전세사기방지", "거주지이동", "수도권외곽", "출퇴근거리",
                 "실용적선택"],
    "변동성 수익 추구자": ["강남재건축", "고위험지역", "변동성", "투자수익", "단기차익",
                   "재건축초과이익", "조합설립", "정비사업", "분양권전매", "갭투자",
                   "전세끼고매수", "알짜입지", "개발호재", "시세차익", "급등지역", "저평가지역",
                   "관리처분계획", "입지선점", "부동산투자", "단기매매", "매도타이밍", "분양권투자",
                   "청약경쟁률", "규제완화", "투자과열지구", "투기과열지구", "준공임박", "입주권거래",
                   "대출레버리지", "법인투자", "자산배분", "세금이슈", "양도세절세", "다주택자전략",
                   "금리민감도", "유동성장세", "고수익지향", "하이리스크", "투자심리", "거래량급증",
                   "전세가율", "실거래차익", "공급부족", "풍선효과", "분양가상승", "피주고매수",
                   "시세반등", "거래절벽", "준공예정"],
    "노후대비": ["시장하락", "노후대비", "전원주택", "지방이주", "시골생활", "은퇴설계",
             "정착지탐색", "생활비절감", "안정적주거", "장기거주", "전세탈출", "주거안정",
             "삶의질", "귀촌", "귀농", "전세자유", "자급자족", "한적한환경", "저밀도주거", "자연친화",
             "실거주중심", "은퇴이주", "시니어라이프", "고령화대응", "시골주택", "토지구입", "단독주택",
             "인프라중심", "지방중소도시", "생활권변화", "노년기생활", "주택다운사이징", "저렴한집값",
             "생활환경개선", "부동산리스크회피", "세컨하우스", "다주택전략", "실거주우선", "비도시선호",
             "생활패턴변화", "슬로우라이프", "자산보전", "부채최소화", "부동산분산", "장기안정성", "생활편의시설",
             "교통편변화", "건강중심", "커뮤니티형성"],
}

feature_array = np.array([d["features"] for d in input_data])
feature_min = feature_array.min(axis=0)
feature_max = feature_array.max(axis=0)
feature_scaled = (feature_array - feature_min) / (feature_max - feature_min + 1e-9)

all_embeddings = []
labels = []
colors = []

for i, d in enumerate(input_data):
    text_emb = model.encode(d["text"])
    hybrid = np.concatenate([text_emb, feature_scaled[i]])
    all_embeddings.append(hybrid)
    labels.append(d["text"])
    colors.append('blue')

for trend_name, keywords in trend_keywords.items():
    keyword_embs = model.encode(keywords)
    centroid = np.mean(keyword_embs, axis=0)

    dummy_features = np.zeros_like(feature_scaled[0])
    centroid_with_feature = np.concatenate([centroid, dummy_features])

    all_embeddings.append(centroid_with_feature)
    labels.append(f"[트렌드] {trend_name}")
    colors.append('red')

tsne = TSNE(n_components=3, perplexity=3, random_state=42)
reduced = tsne.fit_transform(np.array(all_embeddings))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate(labels):
    x, y, z = reduced[i]
    ax.scatter(x, y, z, c=colors[i], s=100 if colors[i] == 'blue' else 150, marker='o' if colors[i] == 'blue' else 'X')
    ax.text(x + 1, y + 1, z + 1, label, fontsize=9)

ax.set_title("입력 문장(수치 포함) vs 트렌드 중심점 (3D 시각화)")
plt.tight_layout()
plt.show()

print("[트렌드 유사도 점수]\n")

input_vectors = all_embeddings[:len(input_data)]
trend_vectors = all_embeddings[len(input_data):]
trend_names = labels[len(input_data):]

for i, input_vec in enumerate(input_vectors):
    print(f"🧍 입력: \"{labels[i]}\"")
    for j, trend_vec in enumerate(trend_vectors):
        sim = cos_sim(input_vec, trend_vec).item()  # 코사인 유사도 (0~1)
        score = round(sim * 100, 2)
        print(f"{trend_names[j]} → {score}점")
    print()

