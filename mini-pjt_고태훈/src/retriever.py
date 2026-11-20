"""RAG 파이프라인: 메뉴·가족 문서를 만들어 벡터 검색하고, 답변의 근거(contexts)를 모은다.

정확한 조건 판단(기피 여부, 가격 합계)은 tools.py 의 구조화 데이터로 하고,
이 파일은 "무엇이 있는지, 왜 이 메뉴인지"의 근거 문서를 찾아 주는 역할만 한다.
구성원 기피 항목은 개인정보 보호를 위해 문서에 넣지 않는다.
"""
import hashlib
import os
from functools import lru_cache
from typing import Optional

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from tools import (CATEGORIES, CATEGORY_KO, DATA_DIR, ROOT, expand_terms, get_member, load_family,
                   load_menus, member_ids, slots)

COLLECTION = "family_menu"
DOC_FORMAT_VERSION = "2"   # 문서 형식을 바꾸면 올려서 인덱스를 다시 만든다 (2: 구성원 기피 비공개)
DEFAULT_K = 5
SEARCH_KINDS = CATEGORIES + ["member"]


# ---------------------------------------------------------------------------
# 문서 만들기 (contexts 에도 같은 문자열을 쓴다)
# ---------------------------------------------------------------------------

def menu_line(menu_id: str) -> str:
    """메뉴 근거 문서 한 줄: "[main_side:MS12] 제육볶음 | 6,500원 | 2인분 | 1인분 350kcal | 매운맛;돼지고기"."""
    m = load_menus()[menu_id]
    tags = ";".join(sorted(m["tags"])) or "-"
    kind = f"{m['staple_type']} " if m["staple_type"] else ""
    return (f"[{m['category']}:{menu_id}] {m['name']} | {CATEGORY_KO[m['category']]} {kind}| "
            f"{m['price_per_pack']:,}원 | {m['servings_per_pack']}인분 | 1인분 {m['kcal_per_serving']}kcal | {tags}")


def _term_name(term: str) -> str:
    """menu_id 면 메뉴 이름으로, 태그면 그대로 돌려준다."""
    menus = load_menus()
    return menus[term]["name"] if term in menus else term


def _absent_meals(member_id: str) -> str:
    """구성원이 집에서 먹지 않는 끼니를 "월-점심, 화-점심" 식으로 돌려준다."""
    att = get_member(member_id)["attendance"]
    fam = load_family()
    out = [s for s in slots()
           if not att[fam["days"].index(s.split("-")[0])][fam["meals"].index(s.split("-")[1])]]
    return ", ".join(out) or "없음"


def member_line(member_id: str) -> str:
    """구성원 근거 문서 한 줄: "[member:둘째] 둘째(초등학생) | 기피: 비공개 | 선호: 돈가스(3), 계란(3) | 하루 1,500~2,200kcal | ...".

    기피 항목은 건강 정보일 수 있어 벡터 DB·contexts 에 넣지 않는다(식단 계산은 tools.py 가 원본 데이터로 한다).
    """
    m = get_member(member_id)
    avoid = "비공개" if m["avoid"] else "없음"
    prefer = ", ".join(f"{_term_name(t)}({s})" for t, s in sorted(m["prefer"].items(), key=lambda x: -x[1])) or "없음"
    return (f"[member:{member_id}] {m['role']} | 기피: {avoid} | 선호: {prefer} | "
            f"하루 {m['daily_kcal_min']:,}~{m['daily_kcal_max']:,}kcal | 집에서 안 먹는 끼니: {_absent_meals(member_id)} | "
            f"{m.get('note', '')}")


def build_documents() -> list:
    """검색할 문서 목록을 만든다. page_content 는 검색용 문장, metadata.line 은 contexts 에 넣을 한 줄."""
    docs = []
    for mid, m in sorted(load_menus().items()):
        line = menu_line(mid)
        docs.append(Document(page_content=f"{line}\n{m['description']}",
                             metadata={"doc_id": f"{m['category']}:{mid}", "kind": m["category"], "line": line}))
    for pid in member_ids():
        line = member_line(pid)
        docs.append(Document(page_content=f"{line}\n가족 구성원 {pid}의 식사 정보",
                             metadata={"doc_id": f"member:{pid}", "kind": "member", "line": line}))
    return docs


# ---------------------------------------------------------------------------
# 벡터 저장소
# ---------------------------------------------------------------------------

def _data_hash() -> str:
    """메뉴·가족 데이터와 문서 형식 버전의 해시. 바뀌면 인덱스를 다시 만든다."""
    h = hashlib.sha1(DOC_FORMAT_VERSION.encode())
    for f in [DATA_DIR / "menus.csv", DATA_DIR / "family.json"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _chroma_dir() -> str:
    """Chroma 저장 폴더. .env 의 CHROMA_DIR(기본 chroma_db)을 프로젝트 루트 기준으로 읽는다."""
    path = os.environ.get("CHROMA_DIR", "chroma_db")
    return path if os.path.isabs(path) else str(ROOT / path)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """벡터 저장소를 연다. 비어 있거나 데이터가 바뀌었으면 문서를 다시 임베딩해 채운다."""
    embeddings = BedrockEmbeddings(model_id=os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
                                   region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    data_hash = _data_hash()
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                   persist_directory=_chroma_dir(), collection_metadata={"data_hash": data_hash})
    meta = store._collection.metadata or {}
    if meta.get("data_hash") != data_hash or store._collection.count() == 0:
        store.delete_collection()
        store = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                       persist_directory=_chroma_dir(), collection_metadata={"data_hash": data_hash})
        docs = build_documents()
        store.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
    return store


def search(query: str, kind: Optional[str] = None, k: int = DEFAULT_K, exclude=None) -> list:
    """질의와 비슷한 문서를 k 개 찾아 contexts 형식의 한 줄 목록으로 돌려준다.

    kind 로 구성·구성원만 좁히고, exclude(태그 또는 menu_id 목록)에 걸리는 메뉴는 뺀다.
    벡터 검색은 "~말고" 같은 부정을 다루지 못하므로, 후보를 넉넉히 받아 태그로 거른다.
    """
    if kind and kind not in SEARCH_KINDS:
        raise ValueError(f"kind 는 {SEARCH_KINDS} 중 하나여야 합니다.")
    banned = expand_terms(exclude or [])
    fetch = k * 4 if banned else k
    if kind:
        where = {"kind": kind}
    elif any(name in query for name in member_ids() + ["가족", "구성원"]):
        where = None
    else:
        where = {"kind": {"$ne": "member"}}  # 구성원 문서는 길어서 일반 음식 질의에도 걸리므로, 사람을 묻지 않으면 뺀다
    docs = get_vectorstore().similarity_search(query, k=fetch, filter=where)
    lines = [d.metadata["line"] for d in docs if d.metadata["doc_id"].split(":", 1)[1] not in banned]
    return lines[:k]


@tool
def search_docs(query: str, kind: Optional[str] = None, k: int = DEFAULT_K, exclude: Optional[list] = None) -> str:
    """메뉴 카탈로그와 가족 구성원 문서에서 질의와 관련된 근거를 검색한다.

    "둘째가 싫어하는 게 뭐였지?", "국물 요리 뭐 있어?", "매운 거 말고 국" 같은 질문에 쓴다.
    식단을 짤 때는 이 도구가 아니라 plan_week 를 쓴다. 기피 여부·가격 합계 판단에는 쓰지 않는다.

    Args:
        query: 찾을 내용 (자연어)
        kind: 문서 종류로 좁히기. staple(주식) / soup(국) / main_side(메인반찬) / sub_side(부반찬) / member(가족 구성원). 생략하면 전체
        k: 돌려줄 문서 수 (기본 5)
        exclude: 결과에서 뺄 태그나 menu_id 목록. 예: "매운 거 말고" → ["매운맛"], "생선 빼고" → ["생선"]
    """
    lines = search(query, kind, k, exclude)
    return "\n".join(lines) if lines else "관련 문서를 찾지 못했습니다."


# ---------------------------------------------------------------------------
# 식단 응답의 contexts
# ---------------------------------------------------------------------------

def contexts_for_plan(plan: dict) -> list:
    """Plan JSON 에 나온 모든 메뉴와 출석한 구성원의 문서 한 줄을 중복 없이 모은다. 벡터 검색 없이 결정론적이다."""
    if not plan or plan.get("status") != "ok":
        return []
    menu_ids, people = [], []
    for meal in plan["meals"]:
        people += meal["attendees"]
        menu_ids += [meal[c]["menu_id"] for c in ("staple", "soup", "main_side")]
        menu_ids += [i["menu_id"] for i in meal["sub_sides"] + meal["substitutes"]]
    members = [member_line(p) for p in dict.fromkeys(people)]
    return members + [menu_line(m) for m in dict.fromkeys(menu_ids)]
