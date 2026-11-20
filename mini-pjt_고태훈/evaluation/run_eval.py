"""test_queries.csv 의 질문을 POST /query 로 보내 결과를 자동 검사하고 evaluation/report.md 를 쓴다.

실행: python evaluation/run_eval.py [--only 1,6] [--out evaluation/report.md]

- 서버를 따로 띄우지 않고 FastAPI TestClient 로 src/agent.py 의 POST /query 를 그대로 호출한다.
- 대화(thread_id)마다 data/history 시드를 임시 폴더로 새로 복사해 HISTORY_DIR 로 지정한다.
  그래서 한 대화의 "확정"이 다른 질문의 비교 대상 3주를 바꾸지 않고, 제출 데이터도 바뀌지 않는다.
- 숫자 검사는 "answer 의 금액·칼로리 숫자가 이번 요청의 trace·contexts 어딘가에 있는지"만 본다
  (출처 검사). 숫자가 올바른 끼니·사람에 붙었는지까지는 보지 않는다.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TODAY", "2026-11-19")  # 시드 기록(W45~W47) 기준 날짜. 실행 날짜와 무관하게 같은 결과를 내려고 고정

from fastapi.testclient import TestClient  # noqa: E402

import agent  # noqa: E402
import tools  # noqa: E402

SEED_HISTORY = ROOT / "data" / "history"
CSV_PATH = Path(__file__).resolve().parent / "test_queries.csv"
FAMILY_PATH = ROOT / "data" / "family.json"
FAMILY_BYTES = FAMILY_PATH.read_bytes()  # 평가 시작 시점의 가족 정보 (변경 불가 검사용)


# ---------------------------------------------------------------------------
# 숫자 추출 (answer 와 도구 결과에 같은 규칙을 쓴다)
# ---------------------------------------------------------------------------

_NUM = r"\d[\d,]*"
_PATTERNS = [
    (re.compile(rf"({_NUM})\s*만\s*({_NUM})\s*천\s*원"), lambda a, b: a * 10000 + b * 1000),
    (re.compile(rf"({_NUM})\s*만\s*({_NUM})\s*원"), lambda a, b: a * 10000 + b),
    (re.compile(rf"({_NUM})\s*만\s*원"), lambda a: a * 10000),
    (re.compile(rf"({_NUM})\s*천\s*원"), lambda a: a * 1000),
    (re.compile(rf"({_NUM})\s*[~\-–]\s*({_NUM})\s*(?:원|kcal)", re.I), None),  # 범위: 두 수 모두 같은 단위
    (re.compile(rf"({_NUM})\s*(?:원|kcal)", re.I), lambda a: a),
]


def _to_int(s: str) -> int:
    """'18,000' → 18000."""
    return int(s.replace(",", ""))


def unit_numbers(text: str) -> list:
    """텍스트에서 '원·만원·천원·kcal' 이 붙은 숫자를 원·kcal 단위 정수로 뽑는다. 예: '1만 8천원' → 18000."""
    found, taken = [], [False] * len(text)
    for pat, fn in _PATTERNS:
        for m in pat.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            nums = [_to_int(g) for g in m.groups()]
            found += nums if fn is None else [fn(*nums)]
            for i in range(m.start(), m.end()):
                taken[i] = True
    return found


def all_numbers(obj) -> set:
    """도구 결과·contexts 안의 모든 숫자(정수 값, 문자열 속 숫자, 단위 표기 정규화 값)를 모은다."""
    out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(int(obj))
    elif isinstance(obj, str):
        out.update(_to_int(x) for x in re.findall(_NUM, obj))
        out.update(unit_numbers(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out |= all_numbers(k) | all_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= all_numbers(v)
    return out


# ---------------------------------------------------------------------------
# 검사
# ---------------------------------------------------------------------------

def _private_terms() -> dict:
    """구성원별 기피 항목 이름(태그 또는 메뉴 이름). 출력에 구성원과 함께 나오면 개인정보 누출로 본다."""
    menus = tools.load_menus()
    return {m["member_id"]: {menus[a]["name"] if a in menus else a for a in m["avoid"]}
            for m in tools.load_family()["members"]}


def privacy_leaks(text: str) -> list:
    """한 문장 안에 구성원 이름, 그 구성원의 기피 항목, '기피·싫어·못 먹' 같은 말이 함께 있으면 누출로 본다.

    메뉴 이름 속 글자(예: '생선까스'의 '생선')는 기피 항목을 말한 것이 아니므로, 카탈로그 메뉴 이름을 지운 뒤 찾는다.
    돌려주는 항목에는 근거 문장(앞 60자)을 붙인다.
    """
    menus = tools.load_menus()
    names = sorted((m["name"] for m in menus.values()), key=len, reverse=True)
    avoided_names = {p: {menus[m]["name"] for m in tools.avoid_set(p)} for p in tools.member_ids()}
    leaks = []
    for chunk in re.split(r"[.\n!?]", text):
        # 메뉴 이름으로 짐작하게 하는 누출: "엄마 가지튀김 제외", "아빠는 고구마맛탕 안 먹음"
        for piece in re.split(r"[|/,·]\s*(?=[^|/,·]*(?:아빠|엄마|첫째|둘째))", chunk):
            if re.search(r"제외|안 ?먹|못 ?먹|빼고", piece):
                for member, avoided in avoided_names.items():
                    if member in piece:
                        leaks += [f"{member}:{n} «{piece.strip()[:60]}»" for n in avoided if n in piece]
        if not re.search(r"기피|싫어|못 ?먹|알레르기|안 ?먹", chunk):
            continue
        bare = chunk
        for n in names:
            bare = bare.replace(n, " ")
        for member, terms in _private_terms().items():
            if member in bare:
                leaks += [f"{member}:{t} «{chunk.strip()[:60]}»" for t in terms if t in bare]
    return leaks


def last_plan_step(trace: list):
    """trace 에서 마지막 plan_week 호출 항목을 돌려준다."""
    steps = [s for s in trace if s["step"] == "plan_week"]
    return steps[-1] if steps else None


def check_expect(key: str, value: str, res: dict, state: dict) -> str:
    """expect 열의 조건 하나를 검사한다. 통과면 빈 문자열, 실패면 이유."""
    trace = res["trace"]
    step = last_plan_step(trace)
    plan = step["output"] if step and isinstance(step["output"], dict) else {}
    if key == "budget":
        return "" if plan.get("budget") == int(value) else f"예산 {plan.get('budget')} ≠ {value}"
    if key == "cost_weight":
        got = plan.get("constraints", {}).get("cost_weight")
        return "" if got == int(value) else f"cost_weight {got} ≠ {value}"
    if key == "extra_avoid":
        who, term = value.split(":")
        ea = plan.get("constraints", {}).get("extra_avoid", {})
        members = tools.member_ids() if who == tools.ALL_MEMBERS_KEY else [who]
        miss = [m for m in members if term not in ea.get(m, [])]
        return "" if not miss else f"extra_avoid 에 {term} 없음: {miss}"
    if key == "absent":
        who, slot = value.split("@")
        meals = {m["slot"]: m["attendees"] for m in plan.get("meals", [])}
        hits = [s for s, att in meals.items() if s.startswith(slot)
                and (who == tools.ALL_MEMBERS_KEY or who in att)]
        return "" if not hits else f"{who} 가 {hits} 에 여전히 포함됨"
    if key == "week_id":
        return "" if plan.get("week_id") == value else f"week_id {plan.get('week_id')} ≠ {value}"
    if key == "saved":
        saves = [s["output"] for s in trace if s["step"] == "save_plan" and isinstance(s["output"], dict)]
        ok = any(r.get("saved") for r in saves)
        return "" if ok == (value == "true") else f"저장 결과 {saves}"
    if key == "no_tool":
        return "" if not any(s["step"] == value for s in trace) else f"{value} 를 불렀음"
    if key == "tool":
        return "" if any(s["step"] == value for s in trace) else f"{value} 를 부르지 않음"
    if key == "relax":
        rel = plan.get("relaxations", [])
        ok = plan.get("status") == "infeasible" and any(r["type"] == value for r in rel)
        return "" if ok else f"status {plan.get('status')}, relaxations {[r['type'] for r in rel]}"
    if key == "ctx":
        return "" if any(c["doc_id"].startswith(value) for c in res["contexts"]) else f"contexts 에 [{value} 없음"
    if key == "ctx_without":
        bad = [c for c in res["contexts"] if value in c["text"].split("|")[-1]]
        return "" if not bad else f"contexts 에 {value} 메뉴 {len(bad)}개"
    if key == "pending":
        ok = any(s["step"] == value and isinstance(s["output"], dict)
                 and s["output"].get("status") == "pending_confirmation" for s in trace)
        return "" if ok else f"{value} 저장 확인 대기 상태가 아님"
    if key == "reasons_file":
        path = plan.get("reasons_file")
        ok = bool(path) and Path(path if os.path.isabs(path) else ROOT / path).exists()
        return "" if ok == (value == "true") else f"끼니별 이유 파일 {path}"
    if key == "tool_arg":  # 예: weekly_summary.slot=일-저녁
        name_arg, want = value.split("=", 1)
        name, arg = name_arg.split(".")
        got = [s["input"].get(arg) for s in trace if s["step"] == name]
        return "" if want in got else f"{name}.{arg} 값 {got} 에 {want} 없음"
    if key == "family_unchanged":
        same = FAMILY_PATH.read_bytes() == FAMILY_BYTES and tools.load_family() == json.loads(FAMILY_BYTES)
        return "" if same else "family.json 또는 메모리의 가족 정보가 바뀜"
    if key == "no_pending":
        bad = any(s["step"] == value and isinstance(s["output"], dict)
                  and s["output"].get("status") == "pending_confirmation" for s in trace)
        return "" if not bad else f"{value} 저장 확인을 물음 (묻지 않고 거절해야 함)"
    if key == "is_past":
        return "" if plan.get("is_past_week") == (value == "true") else f"is_past_week {plan.get('is_past_week')}"
    if key == "no_file":
        path = Path(os.environ["HISTORY_DIR"]) / f"{value}.json"
        return "" if not path.exists() else f"{path.name} 이 저장됨"
    if key == "file_unchanged":  # 시드에 있던 주차면 내용이 그대로인지, 없던 주차면 새로 생기지 않았는지
        path, seed = Path(os.environ["HISTORY_DIR"]) / f"{value}.json", SEED_HISTORY / f"{value}.json"
        same = (path.read_bytes() == seed.read_bytes()) if seed.exists() else not path.exists()
        return "" if same else f"{path.name} 이 바뀌거나 새로 저장됨"
    if key == "overlap_max":
        got = plan.get("repeat_from_history", {}).get("count")
        return "" if got is not None and got <= int(value) else f"최근 3주와 겹친 끼니 {got} > {value}"
    if key == "past_week":
        weeks = [s["output"].get("week_id") for s in trace
                 if s["step"] == "get_past_plan" and isinstance(s["output"], dict)]
        return "" if value in weeks else f"get_past_plan 조회 주차 {weeks} 에 {value} 없음"
    if key == "answer_not":  # '|' 로 구분한 말이 하나라도 answer 에 있으면 실패
        hit = [w for w in value.split("|") if w in res["answer"]]
        return "" if not hit else f"answer 에 들어가면 안 되는 말 {hit}"
    if key == "answer_has":
        return "" if value in res["answer"] else f"answer 에 '{value}' 없음"
    return f"알 수 없는 expect: {key}"


def evaluate(row: dict, res: dict, state: dict) -> dict:
    """질문 하나의 응답을 검사해 결과 dict 를 돌려준다."""
    fails, notes = [], []
    if sorted(res) != ["answer", "contexts", "trace"]:
        fails.append(f"응답 키 {sorted(res)}")
    if any(sorted(c) != ["doc_id", "text"] for c in res.get("contexts", [])):  # 제출 규약: contexts=[{doc_id, text}]
        fails.append("contexts 형식이 {doc_id, text} 가 아님")
    if any(sorted(s) != ["input", "output", "step"] for s in res.get("trace", [])):  # 제출 규약: trace=[{step, input, output}]
        fails.append("trace 형식이 {step, input, output} 이 아님")
    if res.get("answer", "").startswith("요청을 처리하지 못했습니다"):  # 서버 오류(예: Bedrock 한도)는 다른 검사와 무관하게 실패
        fails.append(f"서버 오류: {res['answer'][:80]}")
    trace = res.get("trace", [])
    step = last_plan_step(trace)
    plan = step["output"] if step and isinstance(step["output"], dict) else None
    etype = row["expect_type"]
    checks = [c for c in row["checks"].split(";") if c]

    if etype == "plan":
        if not plan or plan.get("status") != "ok":
            fails.append(f"식단 없음 (status {plan.get('status') if plan else 'plan_week 미호출'})")
        elif "plan_ok" in checks:
            v = tools.verify_plan(plan)
            if not v["ok"]:
                fails.append(f"check_plan 위반 {v['by_type']}")
            notes.append(f"{plan['week_id']} {plan['total_cost']:,}원/{plan['budget']:,}원, 3주 중복 "
                         f"{plan['repeat_from_history']['count']}/{plan['repeat_from_history']['limit']}")
        if plan and plan.get("status") == "ok":  # 답변에 실제 식단이 있는지: 모든 끼니의 메인반찬 이름이 answer 에 있어야 함
            missing = [m["slot"] for m in plan["meals"] if m["main_side"]["name"] not in res.get("answer", "")]
            if missing:
                fails.append(f"답변에 식단이 빠진 끼니 {len(missing)}개 {missing[:5]}")
    elif etype == "infeasible":
        if not plan:
            fails.append("plan_week 미호출")
        elif plan.get("status") == "ok":
            fails.append("해가 없어야 하는데 식단을 만듦")
        elif plan.get("status") == "infeasible" and not plan.get("relaxations"):
            fails.append("relaxations 비어 있음")
        elif plan.get("status") == "no_allowed_menu" and not plan.get("member") and not plan.get("slot"):
            fails.append("no_allowed_menu 에 member·slot 없음")

    for item in [e for e in row["expect"].split(";") if e]:
        key, value = item.split("=", 1)
        reason = check_expect(key, value, res, state)
        if reason:
            fails.append(reason)

    bad_numbers = []
    if "numbers" in checks:
        allowed = all_numbers(trace) | all_numbers(res.get("contexts", [])) | all_numbers(row["input"])
        bad_numbers = sorted({n for n in unit_numbers(res.get("answer", "")) if n not in allowed})
        if bad_numbers:
            fails.append(f"근거 없는 숫자 {bad_numbers}")
    if "privacy" in checks:
        leaks = privacy_leaks(res.get("answer", "")) + [x for c in res.get("contexts", []) for x in privacy_leaks(c["text"])]
        if leaks:
            fails.append(f"기피 정보 누출 {sorted(set(leaks))}")
    return {"pass": not fails, "fails": fails, "notes": notes, "bad_numbers": bad_numbers,
            "tools": [s["step"] for s in trace]}


# ---------------------------------------------------------------------------
# 실행과 리포트
# ---------------------------------------------------------------------------

def fresh_history(tmp_root: Path) -> Path:
    """시드 기록을 새 임시 폴더로 복사하고 HISTORY_DIR 로 지정한다."""
    dest = tmp_root / f"history_{uuid.uuid4().hex[:8]}"
    shutil.copytree(SEED_HISTORY, dest)
    os.environ["HISTORY_DIR"] = str(dest)
    return dest


def run(only=None) -> list:
    """CSV 질문을 순서대로 보내고 검사 결과 목록을 돌려준다."""
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if only:
        keep_threads = {r["thread_id"] for r in rows if r["id"] in only and r["thread_id"]}
        rows = [r for r in rows if r["id"] in only or (r["thread_id"] and r["thread_id"] in keep_threads)]
    client = TestClient(agent.app)
    run_id = uuid.uuid4().hex[:6]
    results, threads = [], {}
    with tempfile.TemporaryDirectory(prefix="eval_hist_") as tmp:
        tmp_root = Path(tmp)
        os.environ["REASONS_DIR"] = str(tmp_root / "reasons")  # 끼니별 이유 파일도 임시 폴더에 쓴다
        for row in rows:
            tid = row["thread_id"].strip()
            if not tid or tid not in threads:
                threads[tid or row["id"]] = fresh_history(tmp_root)
            else:
                os.environ["HISTORY_DIR"] = str(threads[tid])
            body = {"question": row["input"], **({"thread_id": f"{run_id}-{tid}"} if tid else {})}
            t0 = time.perf_counter()
            res = client.post("/query", json=body).json()
            sec = time.perf_counter() - t0
            r = evaluate(row, res, {})
            r.update(id=row["id"], category=row.get("category", ""), thread=tid, question=row["input"], type=row["expect_type"],
                     seconds=round(sec, 1), answer=res.get("answer", ""), contexts=res.get("contexts", []),
                     trace=res.get("trace", []), reference=row.get("reference", ""))
            results.append(r)
            print(f"{row['id']} {'PASS' if r['pass'] else 'FAIL'} {sec:5.1f}s {r['tools']} {r['fails']}", flush=True)
    return results


def write_report(results: list, out: Path) -> None:
    """검사 결과를 마크다운 리포트로 쓴다."""
    n = len(results)
    passed = sum(r["pass"] for r in results)
    plans = [r for r in results if r["type"] == "plan"]
    plan_pass = sum(not any("check_plan" in f or "식단 없음" in f for f in r["fails"]) for r in plans)
    bad_nums = sum(len(r["bad_numbers"]) for r in results)
    secs = [r["seconds"] for r in results]
    lines = [
        "# 평가 리포트",
        "",
        f"- 실행 시각: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 모델: `{os.environ.get('MODEL_ID')}`",
        f"- 질문 수: {n}",
        "",
        "## 요약",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 전체 통과 | {passed}/{n} ({passed / n:.0%}) |",
        f"| 식단 질문 조건 만족 (check_plan: 기피·예산·칼로리·중복·반복) | {plan_pass}/{len(plans)} |",
        f"| answer 의 근거 없는 금액·칼로리 숫자 | {bad_nums}개 |",
        f"| 기피 정보 누출 | {sum(any('누출' in f for f in r['fails']) for r in results)}건 |",
        f"| 응답 시간 평균 / 최대 | {sum(secs) / n:.1f}초 / {max(secs):.1f}초 |",
        *[f"| category {c} 통과 | {sum(r['pass'] for r in results if r.get('category') == c)}/"
          f"{sum(r.get('category') == c for r in results)} |"
          for c in ("positive", "negative", "edge", "guardrail") if any(r.get("category") == c for r in results)],
        "",
        "숫자 검사는 출처 검사다. answer 의 숫자가 이번 요청의 trace·contexts·질문 어딘가에 있으면 통과하며,",
        "그 숫자가 올바른 끼니·사람에 붙었는지는 보지 않는다.",
        "",
        "## 질문별 결과",
        "",
        "| id | category | thread | 유형 | 결과 | 시간 | 도구 | 메모 / 실패 이유 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        memo = "<br>".join(r["fails"] + r["notes"]).replace("|", "\\|")
        lines.append(f"| {r['id']} | {r.get('category') or '-'} | {r['thread'] or '-'} | {r['type']} | {'통과' if r['pass'] else '**실패**'} | "
                     f"{r['seconds']}초 | {', '.join(r['tools']) or '-'} | {memo} |")
    lines += ["", "## 질문과 답변 (앞 600자)", ""]
    for r in results:
        ans = r["answer"][:600].replace("\n", " ")
        lines += [f"**{r['id']}** {r['question']}", "", f"> {ans}{'…' if len(r['answer']) > 600 else ''}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """명령줄 인자를 읽어 평가를 실행하고 리포트를 쓴다."""
    ap = argparse.ArgumentParser(description="가족 식단 Agent 평가")
    ap.add_argument("--only", help="실행할 질문 id (쉼표 구분). 같은 thread 의 질문은 함께 실행")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.md"))
    args = ap.parse_args()
    results = run(set(args.only.split(",")) if args.only else None)
    write_report(results, Path(args.out))
    print(f"\n통과 {sum(r['pass'] for r in results)}/{len(results)} → {args.out}")


if __name__ == "__main__":
    main()
