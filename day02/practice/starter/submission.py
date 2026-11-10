"""Day 2 — 런북 RAG와 평가셋.

문서를 인덱싱하고, 질문에 근거와 함께 답합니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요.
  2. 모델·임베딩은 반드시 함수 **안에서** 만듭니다.
  3. `llm` 인자가 들어오면 그것을 씁니다. 채점기가 가짜 모델을 넣습니다.
  4. `build_retriever`와 `build_rag_chain`은 **주어진 vectorstore/retriever를 그대로 씁니다.**
     함수 안에서 새로 인덱싱하지 마세요.
"""

from __future__ import annotations

import copy
import re
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import MarkdownHeaderTextSplitter

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"

PERSIST_DIR = "./chroma_db"
COLLECTION = "sds_policies"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def _default_embeddings():
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def format_docs(docs) -> str:
    """검색된 문서를 프롬프트에 넣을 문자열로 만듭니다."""
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}" for d in docs
    )


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 파일별 업무 메타데이터 (교육용 더미 데이터입니다.)
#   - 실습 문서 본문에는 담당 조직·수정일 정보가 없으므로 여기서 관리합니다.
#   - department: 답이 이상할 때 "누구에게 물어볼지"를 정하는 키
#   - updated_at: 폐기된 런북을 근거로 든 답을 걸러내는 단서 (YYYY-MM-DD)
FILE_META: dict[str, dict[str, str]] = {
    "leave_policy.md": {"department": "people", "updated_at": "2026-03-02"},
    "oncall_policy.md": {"department": "sre", "updated_at": "2026-06-30"},
    "incident_runbook.md": {"department": "sre", "updated_at": "2026-11-10"},
}

# 청킹 설정 — 마크다운 헤더 기준 분할
#   - 런북·규정 문서는 "## 3. 롤백 기준"처럼 섹션 하나가 하나의 주제입니다.
#     섹션 단위로 자르면 의미 단위가 깨지지 않고, 청크 metadata에 섹션 이름이 남습니다.
#   - 실습 문서의 섹션은 90~350자라 섹션 하나가 그대로 청크 하나가 됩니다.
#     (문서당 3~4개. 고정 길이 자르기처럼 문장 중간이 끊기지 않습니다.)
HEADERS_TO_SPLIT_ON = [("#", "title"), ("##", "section")]
# 첫 "##" 앞부분은 "# 제목 + > 교육용 더미 데이터입니다." 뿐인 짧은 조각입니다.
# 검색에 쓸 내용이 없고 무관한 질문에도 걸리는 잡음이라 이 길이 미만이면 버립니다.
MIN_CHUNK_CHARS = 50


def _file_meta(path: Path) -> dict[str, str]:
    """파일 하나의 metadata 3키(source / department / updated_at)를 만듭니다.

    FILE_META에 없는 파일이 들어와도 세 키가 빠지지 않도록
    department는 "unknown", updated_at은 파일 수정 시각으로 채웁니다.
    """
    known = FILE_META.get(path.name, {})
    try:
        fallback_date = date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        fallback_date = date.today().isoformat()
    return {
        # 경로 전체가 아니라 원본 파일명만 남깁니다. (출처 표시·평가셋 비교용)
        "source": path.name,
        "department": known.get("department", "unknown"),
        "updated_at": known.get("updated_at", fallback_date),
    }


def build_chunks(doc_paths: list[str]) -> list:
    """문서를 읽어 청크 Document 리스트를 만듭니다.

    반드시 지킬 것:
      - 모든 청크 metadata에 source / department / updated_at 세 키가 있어야 합니다.
      - 메타데이터는 **분할하기 전에** 붙여야 상속됩니다.
      - 고정 길이로 자르지 말고 RecursiveCharacterTextSplitter 또는
        MarkdownHeaderTextSplitter를 쓰세요.

    Args:
        doc_paths: 읽을 마크다운 파일 경로 목록

    Returns:
        langchain_core.documents.Document 의 리스트
    """
    # 1) 파일을 읽어 원본 Document를 만들면서 metadata 3키를 **먼저** 붙입니다.
    #    (TextLoader는 langchain_community 의존성이 필요해 직접 읽습니다.)
    originals: list[Document] = []
    for raw_path in doc_paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        originals.append(Document(page_content=text, metadata=_file_meta(path)))

    # 2) 헤더 기준으로 분할합니다.
    #    strip_headers=False: "## 2. 이월" 같은 헤더 줄을 본문에 남겨 둡니다.
    #    기본값(True)이면 헤더 단어가 본문에서 빠져 "이월", "롤백 기준" 같은 핵심어로
    #    검색되지 않고, assess_retrieval의 키워드 매칭에서도 놓치게 됩니다.
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    chunks: list[Document] = []
    for original in originals:
        # MarkdownHeaderTextSplitter에는 split_documents가 없고 split_text(문자열)만 있습니다.
        # 그래서 원본 Document의 metadata가 자동으로 넘어가지 않으므로,
        # 1)에서 원본에 붙여 둔 metadata를 각 섹션 청크에 그대로 물려줍니다(상속).
        # 섹션 청크가 가진 title/section 키는 그 위에 더해집니다.
        for section in splitter.split_text(original.page_content):
            if len(section.page_content.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Document(
                    page_content=section.page_content,
                    metadata={**original.metadata, **section.metadata},
                )
            )
    return chunks


# k=4: 섹션 하나가 청크 1~2개이므로 정답 청크와 바로 옆 문맥까지 잡을 수 있고,
#      문서가 3개뿐이라 이보다 크면 무관한 런북 청크가 컨텍스트를 채워 모델이 끌려갑니다.
RETRIEVER_K = 4


def build_retriever(vectorstore):
    """주어진 벡터저장소에서 retriever를 만듭니다.

    여기서 새로 인덱싱하지 마세요. 인자로 받은 vectorstore를 그대로 씁니다.
    k 값은 직접 정하고, 왜 그 값인지 주석으로 남기세요.
    """
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


# 운영 채널 질문은 로그를 통째로 붙여 넣는 경우가 많습니다.
# 너무 긴 질문은 검색·프롬프트 비용만 키우므로 앞부분만 사용합니다.
MAX_QUESTION_CHARS = 1000

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 사내 운영 런북과 규정을 안내하는 어시스턴트야. "
     "아래 '사내 문서'에 적힌 내용만 근거로 답해. 추측하거나 일반 상식으로 보충하지 마. "
     "문서에 답이 없으면 정확히 '제공된 문서에서 답을 찾을 수 없습니다.'라고만 답해. "
     "답은 한국어로 짧고 명확하게 써.\n\n"
     "--- 사내 문서 ---\n{context}"),
    ("human", "{question}"),
])


def _normalize_question(question: Any) -> str:
    """빈 값·dict·아주 긴 문자열 등 지저분한 입력을 문자열 질문으로 정리합니다."""
    if isinstance(question, dict):
        # {"question": "..."} 형태로 들어와도 처리합니다.
        question = question.get("question", "")
    text = "" if question is None else str(question)
    return text.strip()[:MAX_QUESTION_CHARS]


def _unique_sources(docs) -> list[str]:
    """검색된 문서의 metadata에서 source만 순서를 유지하며 중복 없이 뽑습니다.

    지어낸 파일명이 들어가지 않도록 반드시 실제 검색 결과에서만 가져옵니다.
    """
    sources: list[str] = []
    for d in docs:
        src = (d.metadata or {}).get("source")
        if src and src not in sources:
            sources.append(src)
    return sources


def _no_answer() -> dict:
    """거절 응답. 모듈 상수를 호출자가 수정하지 못하도록 사본을 돌려줍니다."""
    return copy.deepcopy(NO_ANSWER)


def build_rag_chain(retriever, llm=None):
    """질문 문자열을 받아 {"answer": str, "sources": list[str]} 를 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - 검색된 문서 본문을 프롬프트에 실제로 넣어야 합니다.
      - sources는 실제 검색 결과의 metadata에서 뽑습니다. 지어내면 안 됩니다.

    심화 파트를 구현했다면 assess_retrieval 판정을 거쳐야 합니다.
    """
    llm = llm or _default_llm()
    generate = RAG_PROMPT | llm

    def _answer(question: Any) -> dict:
        q = _normalize_question(question)
        if not q:
            # 빈 질문은 검색할 것도 없으므로 바로 거절합니다.
            return _no_answer()

        # 1) 검색: 주어진 retriever를 그대로 씁니다.
        docs = list(retriever.invoke(q) or [])

        # 2) 게이트: 검색 결과가 쓸 만하지 않으면 모델을 부르지 않고 거절합니다.
        #    근거 없이 모델에게 물으면 그럴듯한 답을 지어내기 때문입니다.
        verdict = assess_retrieval(docs, q)
        if not verdict["usable"]:
            return _no_answer()

        # 3) 생성: 검색된 문서 본문을 {context}로 프롬프트에 주입합니다.
        #    질문은 템플릿 문자열이 아니라 변수로 넘기므로 '{', '}' 같은 특수문자도 안전합니다.
        answer = get_text(generate.invoke({"context": format_docs(docs), "question": q})).strip()

        # 4) 모델 스스로 모른다고 답했다면 근거 문서도 없는 것으로 봅니다.
        if not answer or "찾을 수 없" in answer:
            return _no_answer()

        return {"answer": answer, "sources": _unique_sources(docs)}

    return RunnableLambda(_answer)


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 질문에 흔히 붙지만 "무엇에 대한 질문인지"와는 무관한 말들입니다.
# 토큰이 이 말로 시작하면 핵심어에서 뺍니다. (예: "며칠까지", "되나요")
QUESTION_STOPWORDS = (
    "어떻게", "어떤", "어떤가", "어디", "언제", "얼마", "며칠", "몇", "무엇", "뭐", "왜", "누가", "누구",
    "되나요", "되나", "되는", "하나요", "하는", "할까", "인가요", "있나요", "있는", "없나요",
    "알려", "궁금", "가능", "해도", "되요", "돼요", "이에요", "인지", "그리고", "혹시", "지금",
)
# 핵심어가 이 개수 이상 문서에 등장해야 "쓸 만하다"고 봅니다.
# 1로 둔 이유 (비율 1/3 → 완화):
#   paraphrase 질문은 문서와 단어가 거의 겹치지 않습니다.
#   예) "배포하고 나서 에러율이 튀면 되돌려도 되나요?" → 핵심어 7개 중 "배포" 1개만 런북과 일치.
#   비율 기준이면 이런 정답 질문을 모델에게 묻기도 전에 거절합니다.
# 대가: 무관한 질문도 단어 하나만 겹치면 게이트를 통과합니다.
#   그 경우는 프롬프트의 "문서에 없으면 찾을 수 없다고 답하라" 지시와
#   build_rag_chain의 '찾을 수 없' 응답 → 거절 변환이 2차 방어선이 됩니다.
#   핵심어가 하나도 겹치지 않는 경우(빈 결과, 완전히 무관한 문서)는 여전히 모델 호출 전에 끊습니다.
MIN_MATCHED = 1
# 한국어는 조사·어미가 붙으므로(연차는, 이월되나요) 앞 2글자를 어간 근사치로 씁니다.
STEM_LEN = 2


def _keywords(question: str) -> list[str]:
    """질문에서 핵심어(어간 근사치)를 중복 없이, 등장 순서대로 뽑습니다."""
    keywords: list[str] = []
    for token in re.findall(r"[가-힣A-Za-z0-9]+", question or ""):
        token = token.lower()
        if any(token.startswith(sw) for sw in QUESTION_STOPWORDS):
            continue
        if re.fullmatch(r"[가-힣]+", token):
            if len(token) < STEM_LEN:
                continue  # "쓸", "내" 같은 한 글자는 어디에나 등장해 판정을 흐립니다.
            stem = token[:STEM_LEN]
        else:
            stem = token  # 영문·숫자(P1, 5xx 등)는 그대로 비교합니다.
        if stem not in keywords:
            keywords.append(stem)
    return keywords


def assess_retrieval(docs, question: str) -> dict:
    """검색 결과가 이 질문에 답하기에 쓸만한지 판정합니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 내야 합니다.

    판정 방법
      1. 검색 결과가 없으면 거절
      2. 질문에서 핵심어(불용어 제외, 앞 2글자)를 뽑고, 하나도 없으면 거절
      3. 검색된 본문 + 청크의 문서 제목·섹션 이름(metadata)에 핵심어가 몇 개 등장하는지 셈
         (제목 "연차 휴가 규정"의 "휴가"처럼 본문에는 없고 제목에만 있는 단어도 근거로 인정)
      4. matched >= MIN_MATCHED 이면 usable

    Returns:
        {"usable": bool, "reason": str, "matched": int} 형태의 dict.
        usable이 False면 build_rag_chain은 모델을 부르지 않고 거절해야 합니다.
    """
    docs = list(docs or [])
    if not docs:
        return {"usable": False, "reason": "검색 결과가 없음", "matched": 0}

    keywords = _keywords(question)
    if not keywords:
        return {"usable": False, "reason": "질문에서 핵심어를 찾지 못함", "matched": 0}

    parts: list[str] = []
    for d in docs:
        meta = d.metadata or {}
        parts.extend(str(meta.get(key, "")) for key in ("title", "section"))
        parts.append(d.page_content)
    body = " ".join(parts).lower()

    matched = sum(1 for kw in keywords if kw in body)
    needed = MIN_MATCHED
    usable = matched >= needed

    return {
        "usable": usable,
        "reason": f"질의어 {len(keywords)}개 중 {matched}개가 본문에 등장 (기준 {needed}개)",
        "matched": matched,
    }


NO_ANSWER = {
    "answer": "제공된 문서에서 답을 찾을 수 없습니다.",
    "sources": [],
}
