"""Day 7 — 계측과 누적 평가.

6일간 쌓은 시스템을 측정합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. run_eval과 compare_eval은 LLM을 호출하지 않습니다.
     '답이 맞았는가'의 판정은 호출자가 넘겨 준 함수가 합니다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

# 판정 결과에 쓰는 상태값입니다. 이 셋만 씁니다.
PASS, FAIL, ERROR = "pass", "fail", "error"


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

class TraceCollector:
    """실행 중 일어난 일을 기록합니다.

    LangChain 콜백 핸들러로도 쓸 수 있게 만들되, 여기서는 최소 인터페이스만 요구합니다.
    trace는 결정적 구조체라서, LLM이 무슨 말을 했든 이벤트 개수와 필드는 확정 검사됩니다.
    """

    # 허용하는 이벤트 종류입니다. 오타("tool_stat" 등)가 섞이면 나중에 grep으로 셀 때
    # 숫자가 조용히 틀어지므로, 기록하는 순간 바로 막습니다.
    EVENTS = ("llm_start", "llm_end", "tool_start", "tool_end")

    def __init__(self) -> None:
        # 이벤트를 일어난 순서대로 쌓아 두는 목록입니다.
        self._events: list[dict] = []

    def record(self, event: str, name: str, **fields: Any) -> None:
        """이벤트 하나를 기록합니다.

        event: "llm_start" | "llm_end" | "tool_start" | "tool_end" 중 하나
        name : 모델 이름 또는 도구 이름
        fields: latency_s, tokens 등 부가 정보
        """
        if event not in self.EVENTS:
            raise ValueError(f"알 수 없는 이벤트입니다: {event!r} (허용: {', '.join(self.EVENTS)})")

        # 기본 필드(seq/event/name/ts)를 먼저 두고 부가 필드를 펼칩니다.
        # 부가 필드가 기본 필드 이름을 덮어쓰지 못하도록 기본 필드를 마지막에 다시 씌웁니다.
        base = {"seq": len(self._events), "event": event, "name": name, "ts": time.time()}
        self._events.append({**base, **fields, **{k: base[k] for k in ("seq", "event", "name")}})

    @property
    def events(self) -> list[dict]:
        """기록된 이벤트 목록을 반환합니다.

        호출자가 반환값을 고쳐도 내부 기록이 바뀌지 않도록 사본을 돌려줍니다.
        """
        return [dict(e) for e in self._events]

    def dump(self, path: str | Path) -> None:
        """이벤트를 JSON Lines 파일로 씁니다. 한 줄에 한 이벤트."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self._events:
                # default=str: 직렬화가 안 되는 값(예: 객체)이 섞여도 dump 자체는 죽지 않게 합니다.
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def _default_judge(item: dict, answer: Any) -> bool:
    """judge가 없을 때의 기본 판정: 답이 비어 있지 않으면 통과."""
    if answer is None:
        return False
    if isinstance(answer, str):
        return bool(answer.strip())
    # dict 응답({"answer": ...})이면 answer 필드를 기준으로 봅니다.
    if isinstance(answer, dict) and "answer" in answer:
        return _default_judge(item, answer["answer"])
    return bool(answer)


def _brief(value: Any, limit: int = 120) -> str:
    """failures의 detail에 남길 짧은 요약 문자열을 만듭니다."""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def run_eval(answer_fn: Callable[[str], Any], eval_set: list[dict],
             judge: Callable[[dict, Any], bool] | None = None) -> dict:
    """평가셋을 전부 돌려 유형별 통과율을 냅니다.

    Args:
        answer_fn: 질문 문자열을 받아 답을 돌려주는 함수. 예외가 날 수 있습니다.
        eval_set: [{"id", "question", "type", ...}] 목록
        judge: (문항, 답) -> 통과 여부. None이면 '답이 비어 있지 않으면 통과'로 봅니다.

    Returns:
        {
          "total": 전체 문항 수,
          "passed": 통과 수,
          "by_type": {유형: {"total": n, "passed": m}},
          "failures": [{"id", "type", "status", "detail"}],
          "results": [{"id", "type", "status"}],   # 추가 키: compare_eval용 문항별 상태
        }

    반드시 지킬 것:
      - answer_fn이 예외를 던져도 전체 실행이 멈추면 안 됩니다.
        그 문항은 status를 ERROR로 기록하고 계속 진행합니다.
    """
    judge = judge or _default_judge
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    results: list[dict] = []
    passed = 0

    for item in eval_set or []:
        qid = item.get("id")
        # 유형이 없는 문항도 집계에서 빠지면 '유형별 합 = 전체'가 깨지므로 unknown으로 묶습니다.
        qtype = item.get("type") or "unknown"
        bucket = by_type.setdefault(qtype, {"total": 0, "passed": 0})
        bucket["total"] += 1

        # 1) 답 생성: 터진 코드는 '틀린 답'과 구분해 ERROR로 남깁니다.
        try:
            answer = answer_fn(item.get("question", ""))
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 그 문항에서 격리합니다.
            status, detail = ERROR, f"answer_fn 예외 {type(exc).__name__}: {exc}"
        else:
            # 2) 판정: 판정 함수가 터진 경우도 같은 이유로 ERROR입니다.
            try:
                ok = bool(judge(item, answer))
            except Exception as exc:  # noqa: BLE001
                status, detail = ERROR, f"judge 예외 {type(exc).__name__}: {exc}"
            else:
                status = PASS if ok else FAIL
                detail = "" if ok else f"불통과 답변: {_brief(answer)}"

        results.append({"id": qid, "type": qtype, "status": status})
        if status == PASS:
            passed += 1
            bucket["passed"] += 1
        else:
            failures.append({"id": qid, "type": qtype, "status": status, "detail": detail})

    return {
        "total": len(results),
        "passed": passed,
        "by_type": by_type,
        "failures": failures,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def _status_map(report: dict) -> dict[Any, str] | None:
    """run_eval 결과에서 {문항 id: 상태} 표를 만듭니다. results가 없으면 None."""
    results = report.get("results")
    if not isinstance(results, list):
        return None
    return {r.get("id"): r.get("status") for r in results}


def compare_eval(before: dict, after: dict) -> dict:
    """개선 전후를 비교합니다. LLM을 호출하지 않습니다.

    통과율 총합만 보면 '무엇이 좋아지고 무엇이 깨졌는지'를 알 수 없습니다.
    한 유형을 고치면 다른 유형이 깨지는 일이 실제로 자주 일어납니다.

    Args:
        before, after: run_eval이 반환한 dict

    Returns:
        {
          "fixed":     [before에서 실패했다가 after에서 통과한 문항 id],
          "regressed": [before에서 통과했다가 after에서 실패한 문항 id],
          "delta": after.passed - before.passed,
          "safe": regressed가 비었으면 True,
        }
    """
    before_map, after_map = _status_map(before), _status_map(after)

    if before_map is not None and after_map is not None:
        # 문항별 상태표가 양쪽에 있으면 정확히 비교합니다.
        # 한쪽에만 있는 문항(새로 추가/삭제된 문항)은 전후 비교 대상이 아니므로 뺍니다.
        # 순서는 before의 문항 순서를 따라 매번 같은 결과가 나오게 합니다.
        common = [i for i in before_map if i in after_map]
        fixed = [i for i in common if before_map[i] != PASS and after_map[i] == PASS]
        regressed = [i for i in common if before_map[i] == PASS and after_map[i] != PASS]
    else:
        # results가 없으면 failures만으로 판단합니다. (같은 평가셋을 돌렸다고 가정)
        # before에서 실패였는데 after 실패 목록에 없으면 고쳐진 것, 그 반대면 깨진 것입니다.
        before_fail = [f.get("id") for f in before.get("failures", [])]
        after_fail = [f.get("id") for f in after.get("failures", [])]
        fixed = [i for i in before_fail if i not in set(after_fail)]
        regressed = [i for i in after_fail if i not in set(before_fail)]

    return {
        "fixed": fixed,
        "regressed": regressed,
        "delta": after.get("passed", 0) - before.get("passed", 0),
        # 회귀가 하나라도 있으면 통과 수가 늘었어도 안전하지 않습니다.
        "safe": not regressed,
    }
