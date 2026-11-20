"""가족 1주 식단 Agent 의 도메인 도구.

가격·칼로리 합계, 기피 검사, 식단 생성은 모두 이 파일의 코드가 한다.
LLM 은 이 파일의 도구를 호출하고 결과를 설명만 한다.

개인정보 보호: family.json 의 구성원별 기피 항목은 건강 정보일 수 있어 계산에만 쓰고,
도구 결과(이유 문장·위반 메시지·가족 정보)에는 "누가 무엇을 기피하는지"를 쓰지 않는다.
"""
import csv
import hashlib
import json
import math
import os
import time
from contextvars import ContextVar
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pulp
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CATEGORIES = ["staple", "soup", "main_side", "sub_side"]
SINGLE_CATEGORIES = ["staple", "soup", "main_side"]
CATEGORY_KO = {"staple": "주식", "soup": "국", "main_side": "메인반찬", "sub_side": "부반찬"}
SUB_SIDE_COUNT = 3
ALL_MEMBERS_KEY = "전체"
ALLOWED_COST_WEIGHTS = (1, 5)   # 1: 선호 우선(기본), 5: 최대한 싸게
COST_UNIT = 10000               # 목적 함수에서 비용 1만원 = cost_weight 점
TIE_EPS = 1e-7                  # menu_id 순서로 동점을 깨는 아주 작은 가중치
SOLVER_GAP = 0.002              # 최적값과 0.2% 이내면 멈춤 (최적 증명에 오래 걸리는 것을 막음, threads=1 이면 결정론적)
SOLVER_TIME_LIMIT = 60          # 안전장치(초). 정상 풀이는 10초 이내
# 솔버는 HiGHS(PuLP 인터페이스). PuLP 내장 CBC 2.10.3 은 빠듯한 예산에서 멈추고 시간 제한도 무시해서 바꿨다.
REPEAT_CAP = 2                  # 한 주 안에서 같은 공용 메뉴는 최대 2회
REPEAT_CAP_BY_STAPLE_TYPE = {"밥": 3}  # 밥은 3종류뿐이라 3회까지
HISTORY_OVERLAP_RATIO = 0.5     # 최근 3주와 (국, 메인반찬) 조합이 겹치는 끼니 ≤ 내림(상을 차리는 끼니 수 × 0.5)
PLAN_STORE_SIZE = 100           # 메모리에 보관하는 식단 수 (서버 재시작 시 사라짐)
# 해 없음일 때 완화 재풀이의 벌점 (선호 점수 한 주 합 ≈ 450 보다 훨씬 커서 필요할 때만 푼다).
# 순서: 3주 중복 1끼(500) < 반복 1회(1,000) < 예산 1만원(1,000) < 칼로리 100kcal(1,000) — 단위당 칼로리를 가장 늦게 푼다
RELAX_PENALTY = {"history": 500, "repeat": 1000, "budget": 0.1, "kcal": 10}   # budget 은 원당, kcal 은 kcal당
RELAX_GAP = 0.0005              # 완화안 금액이 부풀지 않게 더 촘촘히 푼다
HISTORY_MIN_PENALTY = 100       # min_history_overlap 일 때 최근 3주와 겹친 끼니 1개당 감점 (끼니 선호 점수보다 커서 겹침을 먼저 줄인다)
# save_plan 은 사용자가 저장 확인 질문에 "예"로 승인한 재개 요청에서만 True 가 된다 (서버가 설정, 프롬프트에만 맡기지 않는 안전장치)
SAVE_ALLOWED: ContextVar = ContextVar("save_allowed", default=False)


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------

def history_dir() -> Path:
    """식단 기록 폴더 경로를 돌려준다. .env 의 HISTORY_DIR(기본 data/history)을 프로젝트 루트 기준으로 읽는다."""
    path = Path(os.environ.get("HISTORY_DIR", "data/history"))
    return path if path.is_absolute() else ROOT / path


def reasons_dir() -> Path:
    """끼니별 이유 파일 폴더. .env 의 REASONS_DIR(기본 data/reasons)을 프로젝트 루트 기준으로 읽는다."""
    path = Path(os.environ.get("REASONS_DIR", "data/reasons"))
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def load_menus() -> dict:
    """menus.csv 를 읽어 menu_id → 메뉴 정보 dict 로 돌려준다. 숫자 열은 int 로, tags 는 집합으로 바꾼다."""
    menus = {}
    with open(DATA_DIR / "menus.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["price_per_pack"] = int(row["price_per_pack"])
            row["kcal_per_serving"] = int(row["kcal_per_serving"])
            row["servings_per_pack"] = int(row["servings_per_pack"])
            row["tags"] = set(filter(None, row["tags"].split(";")))
            menus[row["menu_id"]] = row
    return menus


@lru_cache(maxsize=1)
def load_family() -> dict:
    """family.json 을 읽어 그대로 돌려준다."""
    with open(DATA_DIR / "family.json", encoding="utf-8") as f:
        return json.load(f)


def member_ids() -> list:
    """가족 구성원 id 목록을 family.json 순서대로 돌려준다."""
    return [m["member_id"] for m in load_family()["members"]]


def get_member(member_id: str) -> dict:
    """구성원 id 로 프로필을 찾는다. 없으면 KeyError."""
    for m in load_family()["members"]:
        if m["member_id"] == member_id:
            return m
    raise KeyError(f"가족 구성원 '{member_id}' 이(가) 없습니다. 가능한 값: {member_ids()}")


def menus_by_category() -> dict:
    """구성(category) → menu_id 목록(정렬)을 돌려준다."""
    out = {c: [] for c in CATEGORIES}
    for mid in sorted(load_menus()):
        out[load_menus()[mid]["category"]].append(mid)
    return out


# ---------------------------------------------------------------------------
# 기피·선호
# ---------------------------------------------------------------------------

def menu_tags(menu_id: str) -> set:
    """메뉴의 태그 집합을 돌려준다. 주식 종류(밥·떡·면·빵)도 태그로 취급한다."""
    menu = load_menus()[menu_id]
    tags = set(menu["tags"])
    if menu["staple_type"]:
        tags.add(menu["staple_type"])
    return tags


def expand_terms(terms) -> set:
    """menu_id 또는 태그 목록을 해당하는 menu_id 집합으로 펼친다. plan_week 와 check_plan 이 같은 규칙을 쓴다."""
    terms = set(terms)
    return {mid for mid in load_menus() if mid in terms or menu_tags(mid) & terms}


def normalize_extra_avoid(extra_avoid) -> dict:
    """extra_avoid({구성원 또는 '전체': [menu_id 또는 태그]})를 구성원별 목록으로 정리한다.

    '전체' 키는 모든 구성원에게 더한다. 모르는 구성원 이름이면 KeyError.
    """
    out = {mid: [] for mid in member_ids()}
    for key, terms in (extra_avoid or {}).items():
        targets = member_ids() if key == ALL_MEMBERS_KEY else [get_member(key)["member_id"]]
        for t in targets:
            out[t].extend(terms)
    return {k: sorted(set(v)) for k, v in out.items() if v}


def avoid_set(member_id: str, extra_avoid=None) -> set:
    """구성원이 기피하는 menu_id 집합을 돌려준다. family.json 의 avoid 와 이번 요청의 extra_avoid 를 합친다."""
    extra = normalize_extra_avoid(extra_avoid).get(member_id, [])
    return expand_terms(list(get_member(member_id)["avoid"]) + extra)


def pref_score(member_id: str, menu_id: str, extra_avoid=None) -> int:
    """구성원의 메뉴 선호 점수(0~3)를 돌려준다.

    규칙: 기피 메뉴는 0. prefer 에 menu_id 가 있으면 그 점수를 쓰고,
    없으면 메뉴 태그에 걸린 점수 중 가장 큰 값을 쓴다. 해당 없으면 0.
    """
    if menu_id in avoid_set(member_id, extra_avoid):
        return 0
    prefer = get_member(member_id)["prefer"]
    if menu_id in prefer:
        return prefer[menu_id]
    return max((prefer[t] for t in menu_tags(menu_id) if t in prefer), default=0)


# ---------------------------------------------------------------------------
# 끼니와 출석
# ---------------------------------------------------------------------------

def slots() -> list:
    """한 주의 21끼 이름("월-아침" 형식)을 순서대로 돌려준다."""
    fam = load_family()
    return [f"{d}-{m}" for d in fam["days"] for m in fam["meals"]]


def slot_index(slot: str) -> tuple:
    """끼니 이름을 (요일 인덱스, 끼니 인덱스)로 바꾼다."""
    fam = load_family()
    day, meal = slot.split("-")
    return fam["days"].index(day), fam["meals"].index(meal)


def attendance(attendance_changes=None) -> dict:
    """끼니별 출석자 목록을 돌려준다. family.json 출석표에 이번 요청의 변경을 반영한다.

    attendance_changes 항목: {"member": "첫째", "day": "수", "meal": "저녁", "eats": false}.
    member 를 '전체'로 쓰면 모두에게, meal 을 생략하면 그날 세 끼 모두에 적용한다.
    """
    fam = load_family()
    table = {m["member_id"]: [list(day) for day in m["attendance"]] for m in fam["members"]}
    for ch in attendance_changes or []:
        targets = member_ids() if ch["member"] == ALL_MEMBERS_KEY else [get_member(ch["member"])["member_id"]]
        d = fam["days"].index(ch["day"])
        meals = [fam["meals"].index(ch["meal"])] if ch.get("meal") else range(len(fam["meals"]))
        for t in targets:
            for mi in meals:
                table[t][d][mi] = 1 if ch.get("eats", False) else 0
    out = {}
    for s in slots():
        d, mi = slot_index(s)
        out[s] = [p for p in member_ids() if table[p][d][mi]]
    return out


def day_of(slot: str) -> str:
    """끼니 이름에서 요일을 뽑는다."""
    return slot.split("-")[0]


def kcal_bounds(member_id: str, meals_eaten: int) -> tuple:
    """그날 먹는 끼니 수에 비례해 줄인 하루 칼로리 범위 (하한, 상한)을 돌려준다. 정수로 내림·올림한다."""
    m = get_member(member_id)
    return (math.floor(m["daily_kcal_min"] * meals_eaten / 3),
            math.ceil(m["daily_kcal_max"] * meals_eaten / 3))


# ---------------------------------------------------------------------------
# 주차와 최근 3주 기록
# ---------------------------------------------------------------------------

def _week_start(week_id: str) -> date:
    """'2026-W48' 형식의 주차를 그 주 월요일 날짜로 바꾼다."""
    year, week = week_id.split("-W")
    return date.fromisocalendar(int(year), int(week), 1)


def _week_id(day: date) -> str:
    """날짜를 ISO 주차 문자열 'YYYY-Www' 로 바꾼다."""
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def history_weeks_saved() -> list:
    """기록 폴더에 저장된 주차 목록을 오름차순으로 돌려준다."""
    d = history_dir()
    return sorted(p.stem for p in d.glob("*-W*.json")) if d.exists() else []


def today() -> date:
    """기준 날짜. .env 의 TODAY(YYYY-MM-DD)가 있으면 그 날짜, 없으면 실제 오늘 (평가를 날짜와 무관하게 재현하려고 둔다)."""
    value = os.environ.get("TODAY")
    return date.fromisoformat(value) if value else date.today()


def current_week() -> str:
    """기준 날짜가 속한 ISO 주차."""
    return _week_id(today())


def is_past_week(week_id: str) -> bool:
    """현재 주보다 앞선 주차인지. 과거 주차 식단은 다시 짜서 보여 줄 수는 있지만 확정(저장)할 수 없다."""
    return _week_start(week_id) < _week_start(current_week())


def default_target_week() -> str:
    """질문에 주차가 없을 때 계획할 주: 기록에 있는 가장 최근 주의 다음 주. 기록이 없으면 오늘이 속한 주."""
    saved = history_weeks_saved()
    if not saved:
        return _week_id(date.today())
    return _week_id(_week_start(saved[-1]) + timedelta(days=7))


def previous_weeks(target_week: str, n: int = 3) -> list:
    """target_week 바로 앞의 ISO 주 n 개를 오래된 순으로 돌려준다."""
    start = _week_start(target_week)
    return [_week_id(start - timedelta(days=7 * k)) for k in range(n, 0, -1)]


def load_history_week(week_id: str) -> dict:
    """기록 폴더에서 한 주의 식단을 읽는다. 파일이 없으면 빈 식단을 돌려준다."""
    path = history_dir() / f"{week_id}.json"
    if not path.exists():
        return {"week_id": week_id, "meals": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def history_pairs(weeks) -> set:
    """주어진 주들의 공용 (국 menu_id, 메인반찬 menu_id) 조합 집합을 돌려준다. 대체 메뉴는 넣지 않는다."""
    pairs = set()
    for w in weeks:
        for meal in load_history_week(w)["meals"]:
            if meal.get("soup") and meal.get("main_side"):
                pairs.add((meal["soup"]["menu_id"], meal["main_side"]["menu_id"]))
    return pairs


# ---------------------------------------------------------------------------
# 끼니별 먹는 사람·팩 수
# ---------------------------------------------------------------------------

def eaters(attendees, menu_id: str, avoid_by_member: dict) -> list:
    """출석자 중 그 메뉴를 기피하지 않는 사람 목록을 돌려준다. avoid_by_member 는 구성원 → 기피 menu_id 집합."""
    return [p for p in attendees if menu_id not in avoid_by_member[p]]


def packs_needed(menu_id: str, n_eaters: int) -> int:
    """n 명이 1인분씩 먹을 때 살 팩 수: 올림(n ÷ 팩당 인분 수)."""
    return math.ceil(n_eaters / load_menus()[menu_id]["servings_per_pack"]) if n_eaters else 0


def repeat_cap(menu_id: str) -> int:
    """한 주 안에서 그 메뉴를 공용으로 낼 수 있는 최대 횟수."""
    return REPEAT_CAP_BY_STAPLE_TYPE.get(load_menus()[menu_id]["staple_type"], REPEAT_CAP)


def history_overlap_limit(n_active_slots: int) -> int:
    """최근 3주와 (국, 메인반찬) 조합이 겹쳐도 되는 끼니 수 상한."""
    return math.floor(n_active_slots * HISTORY_OVERLAP_RATIO)


# ---------------------------------------------------------------------------
# 식단 최적화 (plan_week 내부)
# ---------------------------------------------------------------------------

def build_context(budget=None, attendance_changes=None, extra_avoid=None, cost_weight=1, week_id=None,
                  min_history_overlap=False) -> dict:
    """식단 최적화에 필요한 입력을 한 번에 계산해 dict 로 돌려준다.

    출석자, 구성원별 기피 집합, 끼니·메뉴별 먹는 사람과 팩 수, 선호 점수,
    예산(없으면 family.json 기본값), 계획 주차와 최근 3주 조합 H, 겹침 최소화 여부를 담는다.
    """
    if cost_weight not in ALLOWED_COST_WEIGHTS:
        raise ValueError(f"cost_weight 는 {ALLOWED_COST_WEIGHTS} 중 하나여야 합니다.")
    menus = load_menus()
    week_id = week_id or default_target_week()
    weeks = previous_weeks(week_id)
    extra = normalize_extra_avoid(extra_avoid)
    att = attendance(attendance_changes)
    avoid = {p: avoid_set(p, extra) for p in member_ids()}
    pref = {p: {m: pref_score(p, m, extra) for m in menus} for p in member_ids()}
    active = [s for s in slots() if att[s]]
    eat = {(s, m): eaters(att[s], m, avoid) for s in active for m in menus}
    return {
        "week_id": week_id,
        "history_weeks": weeks,
        "H": history_pairs(weeks),
        "budget": int(budget) if budget is not None else load_family()["default_weekly_budget"],
        "cost_weight": cost_weight,
        "min_history_overlap": bool(min_history_overlap),
        "attendance_changes": list(attendance_changes or []),
        "extra_avoid": extra,
        "attendance": att,
        "active_slots": active,
        "avoid": avoid,
        "pref": pref,
        "by_cat": menus_by_category(),
        "rank": {m: i for i, m in enumerate(sorted(menus))},
        "eaters": eat,
        "packs": {k: packs_needed(k[1], len(v)) for k, v in eat.items()},
    }


def build_model(ctx: dict, relax: bool = False) -> dict:
    """정수 계획 모델을 만든다. 돌려주는 dict 에 문제(prob)와 변수(x, z, w), 비용식(cost), 여유 변수(slack)를 담는다.

    relax=True 면 예산·칼로리(하한·상한)·3주 중복·반복(메뉴별) 제약에 벌점이 큰 여유 변수를 붙인다.
    기피·상차림 제약(1~4)에는 여유 변수를 붙이지 않는다. 사전 검사를 통과했다면 완화 모델은 항상 해가 있다.

    변수
      x[s,m] : 끼니 s 의 공용 메뉴로 m 을 차림 (먹을 사람이 1명 이상인 메뉴만 만든다 → 제약 2)
      z[s,p,m]: 끼니 s 에서 p 의 대체 메뉴가 m (주식·국·메인반찬, p 가 기피하지 않는 메뉴만)
    제약
      1. 구성 개수: 주식·국·메인반찬 각 1개, 부반찬 3개
      3. 공용 메뉴를 기피하는 출석자에게만 같은 구성의 대체 메뉴 1개
      4. 출석자마다 부반찬 3개 중 최소 1개는 먹을 수 있음
      5. 총비용(공용 팩 + 대체 1인 1팩) ≤ 예산
      6. 사람·요일마다 먹은 칼로리 합이 [하한×k/3, 상한×k/3] 안 (k = 그날 먹는 끼니 수)
      7. 최근 3주 (국, 메인반찬) 조합과 겹치는 끼니 수 ≤ 내림(상을 차리는 끼니 수 × 0.5)
         w[s,h] ≥ x[s,국] + x[s,메인] − 1 로 겹침을 센다
      8. 같은 공용 메뉴는 주 2회까지 (밥은 3회)
    목적: 선호 점수 합 − cost_weight × 비용/1만원 − ε × menu_id 순위 (동점 처리)
          min_history_overlap 이면 − HISTORY_MIN_PENALTY × (최근 3주와 겹친 끼니 수) 를 더해 겹침부터 줄인다
    """
    menus, att, avoid, pref = load_menus(), ctx["attendance"], ctx["avoid"], ctx["pref"]
    prob = pulp.LpProblem("plan_week", pulp.LpMaximize)
    slack = {}

    def sl(name, kind, integer=False, **meta):
        """relax 모드면 여유 변수를 만들어 등록하고, 아니면 0 을 돌려준다."""
        if not relax:
            return 0
        var = pulp.LpVariable(f"slack_{name}", lowBound=0, cat="Integer" if integer else "Continuous")
        slack[name] = {"kind": kind, "var": var, **meta}
        return var

    x = {}
    for s in ctx["active_slots"]:
        for m in menus:
            if ctx["eaters"][(s, m)]:
                x[(s, m)] = pulp.LpVariable(f"x_{s}_{m}", cat="Binary")
    z = {}
    for s in ctx["active_slots"]:
        for p in att[s]:
            for c in SINGLE_CATEGORIES:
                if not any(m in avoid[p] for m in ctx["by_cat"][c]):
                    continue
                for m in ctx["by_cat"][c]:
                    if m not in avoid[p]:
                        z[(s, p, m)] = pulp.LpVariable(f"z_{s}_{p}_{m}", cat="Binary")

    def xs(s, ms):
        return [x[(s, m)] for m in ms if (s, m) in x]

    for s in ctx["active_slots"]:
        # 1. 구성 개수
        for c in CATEGORIES:
            need = SUB_SIDE_COUNT if c == "sub_side" else 1
            prob += pulp.lpSum(xs(s, ctx["by_cat"][c])) == need, f"count_{s}_{c}"
        for p in att[s]:
            # 3. 단일 구성 대체
            for c in SINGLE_CATEGORIES:
                avoided = [m for m in ctx["by_cat"][c] if m in avoid[p]]
                if avoided:
                    prob += (pulp.lpSum(z[(s, p, m)] for m in ctx["by_cat"][c] if (s, p, m) in z)
                             == pulp.lpSum(xs(s, avoided))), f"sub_{s}_{p}_{c}"
            # 4. 부반찬 최소 1개
            ok = [m for m in ctx["by_cat"]["sub_side"] if m not in avoid[p]]
            prob += pulp.lpSum(xs(s, ok)) >= 1, f"side_{s}_{p}"

    # 5. 예산
    cost = (pulp.lpSum(v * menus[m]["price_per_pack"] * ctx["packs"][(s, m)] for (s, m), v in x.items())
            + pulp.lpSum(v * menus[m]["price_per_pack"] for (s, p, m), v in z.items()))
    prob += cost <= ctx["budget"] + sl("budget", "budget"), "budget"

    # 6. 칼로리 (사람 × 요일)
    for p in member_ids():
        for d in load_family()["days"]:
            day_slots = [s for s in ctx["active_slots"] if day_of(s) == d and p in att[s]]
            if not day_slots:
                continue
            kcal = (pulp.lpSum(v * menus[m]["kcal_per_serving"] for (s, m), v in x.items()
                               if s in day_slots and m not in avoid[p])
                    + pulp.lpSum(v * menus[m]["kcal_per_serving"] for (s, pp, m), v in z.items()
                                 if s in day_slots and pp == p))
            lo, hi = kcal_bounds(p, len(day_slots))
            prob += kcal >= lo - sl(f"kcal_min_{p}_{d}", "kcal_min", member=p, day=d, bound=lo), f"kcal_min_{p}_{d}"
            prob += kcal <= hi + sl(f"kcal_max_{p}_{d}", "kcal_max", member=p, day=d, bound=hi), f"kcal_max_{p}_{d}"

    # 7. 최근 3주 (국, 메인반찬) 조합 중복
    w = {}
    for s in ctx["active_slots"]:
        for a, b in sorted(ctx["H"]):
            if (s, a) in x and (s, b) in x:
                w[(s, a, b)] = pulp.LpVariable(f"w_{s}_{a}_{b}", cat="Binary")
                prob += w[(s, a, b)] >= x[(s, a)] + x[(s, b)] - 1, f"pair_{s}_{a}_{b}"
    limit = history_overlap_limit(len(ctx["active_slots"]))
    prob += (pulp.lpSum(w.values()) <= limit + sl("history", "history", integer=True, bound=limit)), "history_overlap"

    # 8. 주 안 반복
    for m in menus:
        uses = [v for (s, mm), v in x.items() if mm == m]
        if len(uses) > repeat_cap(m):
            prob += (pulp.lpSum(uses) <= repeat_cap(m)
                     + sl(f"repeat_{m}", "repeat", integer=True, menu_id=m, bound=repeat_cap(m))), f"repeat_{m}"

    # 목적 함수
    score = (pulp.lpSum(v * sum(pref[p][m] for p in ctx["eaters"][(s, m)]) for (s, m), v in x.items())
             + pulp.lpSum(v * pref[p][m] for (s, p, m), v in z.items()))
    tie = pulp.lpSum(v * ctx["rank"][m] for (s, m), v in x.items())
    penalty = pulp.lpSum(RELAX_PENALTY[s["kind"].split("_")[0]] * s["var"] for s in slack.values())
    overlap = HISTORY_MIN_PENALTY * pulp.lpSum(w.values()) if ctx.get("min_history_overlap") else 0
    prob += score - ctx["cost_weight"] * cost / COST_UNIT - TIE_EPS * tie - penalty - overlap
    return {"prob": prob, "x": x, "z": z, "w": w, "cost": cost, "slack": slack}


def solve_model(model: dict, gap: float = SOLVER_GAP) -> dict:
    """HiGHS(threads=1)로 모델을 풀고 상태와 풀이 시간(초)을 돌려준다. 기본 gap 0.2%."""
    start = time.perf_counter()
    model["prob"].solve(pulp.HiGHS(msg=False, threads=1, gapRel=gap, timeLimit=SOLVER_TIME_LIMIT))
    return {"status": pulp.LpStatus[model["prob"].status], "seconds": round(time.perf_counter() - start, 2)}


# ---------------------------------------------------------------------------
# 해 없음 처리: 사전 검사와 완화안
# ---------------------------------------------------------------------------

def precheck(ctx: dict):
    """풀기 전에 출석자마다 각 끼니·구성에 먹을 수 있는 메뉴가 있는지 확인한다.

    없으면 {"status": "no_allowed_menu", "member", "slot", "category", "message"} 를, 문제가 없으면 None 을 돌려준다.
    (부반찬은 그 끼니에 먹을 사람이 있는 서로 다른 메뉴가 3개 이상인지도 본다.)
    """
    for s in ctx["active_slots"]:
        for p in ctx["attendance"][s]:
            for c in CATEGORIES:
                if not any(m not in ctx["avoid"][p] for m in ctx["by_cat"][c]):
                    return {"status": "no_allowed_menu", "member": p, "slot": s, "category": c,
                            "message": f"{p}가 먹을 수 있는 {CATEGORY_KO[c]}이 카탈로그에 없습니다 ({s}). "
                                       f"추가한 기피 조건을 줄여야 식단을 짤 수 있습니다."}
        usable = [m for m in ctx["by_cat"]["sub_side"] if ctx["eaters"][(s, m)]]
        if len(usable) < SUB_SIDE_COUNT:
            return {"status": "no_allowed_menu", "member": None, "slot": s, "category": "sub_side",
                    "message": f"{s}에 출석자가 먹을 수 있는 부반찬이 {len(usable)}개뿐이라 3개를 차릴 수 없습니다."}
    return None


def describe_relaxations(ctx: dict, model: dict) -> list:
    """완화 재풀이에서 0 이 아닌 여유 변수를 "무엇을 얼마나 풀면 되는지" 목록으로 바꾼다. 금액은 100원 단위 올림."""
    out = []
    for name, s in sorted(model["slack"].items()):
        val = s["var"].value() or 0
        if val <= 1e-6:
            continue
        kind = s["kind"]
        if kind == "budget":
            add = math.ceil(val / 100) * 100
            out.append({"type": "budget", "amount": add, "from": ctx["budget"], "to": ctx["budget"] + add,
                        "message": f"예산을 {add:,}원 늘려야 합니다 ({ctx['budget']:,}원 → {ctx['budget'] + add:,}원)"})
        elif kind in ("kcal_min", "kcal_max"):
            amt = math.ceil(val - 1e-6)
            if amt <= 0:
                continue
            word = "하한보다" if kind == "kcal_min" else "상한보다"
            more = "부족" if kind == "kcal_min" else "초과"
            out.append({"type": kind, "member": s["member"], "day": s["day"], "amount": amt, "bound": s["bound"],
                        "message": f"{s['day']}요일 {s['member']}의 칼로리가 {word} {amt}kcal {more}합니다 "
                                   f"(그날 {'하한' if kind == 'kcal_min' else '상한'} {s['bound']:,}kcal)"})
        elif kind == "history":
            n = round(val)
            out.append({"type": "history", "amount": n, "from": s["bound"], "to": s["bound"] + n,
                        "message": f"최근 3주와 겹치는 국+메인반찬 조합 끼니 상한을 {n}끼 늘려야 합니다 "
                                   f"({s['bound']}끼 → {s['bound'] + n}끼)"})
        elif kind == "repeat":
            n = round(val)
            name_ko = load_menus()[s["menu_id"]]["name"]
            out.append({"type": "repeat", "menu_id": s["menu_id"], "amount": n, "from": s["bound"], "to": s["bound"] + n,
                        "message": f"{name_ko} 주간 반복 상한을 {n}회 늘려야 합니다 ({s['bound']}회 → {s['bound'] + n}회)"})
    return out


def read_solution(ctx: dict, model: dict) -> dict:
    """풀린 모델에서 끼니별 선택을 읽는다. 끼니 → {구성: menu_id 또는 목록, substitutes: [(구성원, 구성, menu_id)]}."""
    picked = {}
    for (s, m), v in model["x"].items():
        if v.value() and v.value() > 0.5:
            c = load_menus()[m]["category"]
            meal = picked.setdefault(s, {"sub_side": [], "substitutes": []})
            if c == "sub_side":
                meal["sub_side"].append(m)
            else:
                meal[c] = m
    for (s, p, m), v in model["z"].items():
        if v.value() and v.value() > 0.5:
            picked[s]["substitutes"].append((p, load_menus()[m]["category"], m))
    for meal in picked.values():
        meal["sub_side"].sort()
    return picked


# ---------------------------------------------------------------------------
# Plan JSON 과 식단 보관소
# ---------------------------------------------------------------------------

PLAN_STORE: dict = {}


def make_plan_id(ctx: dict) -> str:
    """입력 인자와 데이터 파일 내용으로 식단 id 를 만든다. 같은 입력·같은 데이터면 같은 id."""
    h = hashlib.sha1()
    args = {k: ctx[k] for k in ("week_id", "budget", "cost_weight", "attendance_changes", "extra_avoid")}
    if ctx.get("min_history_overlap"):  # 켰을 때만 넣어 기존 식단 id 는 그대로 둔다
        args["min_history_overlap"] = True
    h.update(json.dumps(args, ensure_ascii=False, sort_keys=True).encode())
    files = [DATA_DIR / "menus.csv", DATA_DIR / "family.json"] + [history_dir() / f"{w}.json" for w in ctx["history_weeks"]]
    for f in files:
        h.update(f.read_bytes() if f.exists() else b"")
    return "P" + h.hexdigest()[:10]


def _item(menu_id: str, eater_list: list, packs: int) -> dict:
    """Plan JSON 의 메뉴 항목 하나를 만든다."""
    menu = load_menus()[menu_id]
    return {"menu_id": menu_id, "name": menu["name"], "packs": packs,
            "cost": packs * menu["price_per_pack"], "eaters": eater_list}


def _meal_reason(ctx: dict, slot: str, meal: dict) -> str:
    """끼니별 한 줄 이유: 누가 먹는지, 누구의 선호 메뉴가 들어갔는지, 누가 무엇으로 대체했는지.

    개인정보 보호: 기피 항목(태그·메뉴)이 무엇인지는 쓰지 않고 "기피로"라고만 쓴다.
    누가 어떤 부반찬을 안 먹는지도 쓰지 않는다(메뉴 이름으로 기피 항목을 짐작할 수 있음). 그 정보는 eaters 에만 있다.
    """
    att = ctx["attendance"][slot]
    who = "가족 모두 먹음" if len(att) == len(member_ids()) else "·".join(att) + "만 먹음"
    parts = [who]
    liked = []
    for p in att:  # 그 사람이 실제로 먹는 메뉴 중 선호 점수가 가장 높은 것 하나
        mine = [i["menu_id"] for i in [meal[c] for c in SINGLE_CATEGORIES] + meal["sub_sides"] if p in i["eaters"]]
        mine += [x["menu_id"] for x in meal["substitutes"] if x["member"] == p]
        best = max(mine, key=lambda m: (ctx["pref"][p][m], -ctx["rank"][m]), default=None)
        if best and ctx["pref"][p][best] > 0:
            liked.append(f"{p}-{load_menus()[best]['name']}")
    if liked:
        parts.append("선호 반영: " + ", ".join(liked))
    for sub in meal["substitutes"]:
        parts.append(f"{sub['member']}는 기피로 {CATEGORY_KO[sub['category']]}을 {sub['name']}(으)로 대체")
    return ". ".join(parts)


def format_plan(ctx: dict, sol: dict) -> dict:
    """read_solution 결과를 Plan JSON 으로 바꾼다. 금액·칼로리 합계를 모두 여기서 계산한다."""
    menus = load_menus()
    meals, total = [], 0
    kcal = {p: {d: 0 for d in load_family()["days"]} for p in member_ids()}
    overlap = 0
    for s in ctx["active_slots"]:
        pick = sol[s]
        meal = {"slot": s, "attendees": ctx["attendance"][s]}
        for c in SINGLE_CATEGORIES:
            m = pick[c]
            meal[c] = _item(m, ctx["eaters"][(s, m)], ctx["packs"][(s, m)])
        meal["sub_sides"] = [_item(m, ctx["eaters"][(s, m)], ctx["packs"][(s, m)]) for m in pick["sub_side"]]
        meal["substitutes"] = [{"member": p, "category": c, "menu_id": m, "name": menus[m]["name"],
                                "packs": 1, "cost": menus[m]["price_per_pack"]}
                               for p, c, m in sorted(pick["substitutes"])]
        in_hist = (pick["soup"], pick["main_side"]) in ctx["H"]
        overlap += in_hist
        meal["soup_main_in_history"] = in_hist
        meal["cost"] = (sum(meal[c]["cost"] for c in SINGLE_CATEGORIES)
                        + sum(i["cost"] for i in meal["sub_sides"] + meal["substitutes"]))
        meal["reason"] = _meal_reason(ctx, s, meal)
        total += meal["cost"]
        for item in [meal[c] for c in SINGLE_CATEGORIES] + meal["sub_sides"]:
            for p in item["eaters"]:
                kcal[p][day_of(s)] += menus[item["menu_id"]]["kcal_per_serving"]
        for sub in meal["substitutes"]:
            kcal[sub["member"]][day_of(s)] += menus[sub["menu_id"]]["kcal_per_serving"]
        meals.append(meal)
    return {
        "status": "ok",
        "plan_id": make_plan_id(ctx),
        "week_id": ctx["week_id"],
        "is_past_week": is_past_week(ctx["week_id"]),
        "budget": ctx["budget"],
        "total_cost": total,
        "constraints": {
            "attendance_changes": ctx["attendance_changes"],
            "extra_avoid": ctx["extra_avoid"],
            "repeat_cap": {"기본": REPEAT_CAP, **REPEAT_CAP_BY_STAPLE_TYPE},
            "cost_weight": ctx["cost_weight"],
            "min_history_overlap": ctx["min_history_overlap"],
            "history_weeks": ctx["history_weeks"],
        },
        "meals": meals,
        "daily_kcal": kcal,
        "repeat_from_history": {"count": overlap, "limit": history_overlap_limit(len(ctx["active_slots"]))},
    }


def store_plan(plan: dict) -> None:
    """Plan JSON 을 plan_id 로 메모리에 보관한다. 오래된 것부터 지워 PLAN_STORE_SIZE 개를 넘지 않게 한다."""
    PLAN_STORE[plan["plan_id"]] = plan
    while len(PLAN_STORE) > PLAN_STORE_SIZE:
        PLAN_STORE.pop(next(iter(PLAN_STORE)))


def verify_plan(plan) -> dict:
    """Plan JSON 을 데이터 파일과 plan 에 기록된 조건만으로 처음부터 다시 계산해 위반 목록을 돌려준다.

    최적화 코드와 계산을 공유하지 않는다(태그 펼치기·출석표·칼로리 범위·주차 같은 데이터 해석만 공용).
    검사: 상차림 구성, 카탈로그에 없는 메뉴, 기피(공용·대체·부반찬 최소 1개), 팩 수·금액·총액·예산,
    사람·요일 칼로리, 최근 3주 (국, 메인반찬) 조합 중복, 주 안 반복. 평가 스크립트도 이 함수를 쓴다.
    """
    if isinstance(plan, str):
        plan = json.loads(plan)
    if plan.get("status") != "ok":
        return {"ok": False, "checked": False, "violations": [],
                "note": f"status 가 ok 가 아닌 결과({plan.get('status')})는 식단이 아니라 검사하지 않습니다."}
    menus, fam = load_menus(), load_family()
    cons = plan.get("constraints", {})
    v = []

    def bad(kind, slot, detail):
        v.append({"type": kind, "slot": slot, "detail": detail})

    extra = cons.get("extra_avoid", {})
    avoid = {p: avoid_set(p, extra) for p in member_ids()}
    att = attendance(cons.get("attendance_changes"))
    expected_slots = [s for s in slots() if att[s]]
    got_slots = [m["slot"] for m in plan["meals"]]
    if got_slots != expected_slots:
        bad("structure", None, f"상을 차린 끼니 목록이 출석표와 다름: 빠짐 {sorted(set(expected_slots) - set(got_slots))}, "
                               f"남음 {sorted(set(got_slots) - set(expected_slots))}")

    total, kcal, use, overlap = 0, {}, {}, 0
    weeks = cons.get("history_weeks") or previous_weeks(plan["week_id"])
    if weeks != previous_weeks(plan["week_id"]):
        bad("history", None, f"비교한 주차 {weeks} 가 {plan['week_id']} 의 직전 3주가 아님")
    hist = history_pairs(weeks)

    for meal in plan["meals"]:
        s = meal["slot"]
        A = att.get(s, [])
        if sorted(meal.get("attendees", [])) != sorted(A):
            bad("structure", s, f"출석자 {meal.get('attendees')} ≠ 출석표 {A}")
        items = [(c, meal.get(c)) for c in SINGLE_CATEGORIES] + [("sub_side", i) for i in meal.get("sub_sides", [])]
        if len(meal.get("sub_sides", [])) != SUB_SIDE_COUNT or len({i["menu_id"] for i in meal["sub_sides"]}) != SUB_SIDE_COUNT:
            bad("structure", s, "부반찬은 서로 다른 3개여야 함")
        meal_cost = 0
        for c, item in items:
            if not item or item.get("menu_id") not in menus:
                bad("unknown_menu", s, f"{CATEGORY_KO[c]} 메뉴가 카탈로그에 없음: {item}")
                continue
            m = item["menu_id"]
            menu = menus[m]
            if menu["category"] != c:
                bad("structure", s, f"{menu['name']} 은(는) {CATEGORY_KO[c]}이 아님")
            if item.get("name") != menu["name"]:
                bad("unknown_menu", s, f"{m} 이름 불일치: {item.get('name')} ≠ {menu['name']}")
            can_eat = [p for p in A if m not in avoid[p]]
            served = item.get("eaters", [])
            for p in served:
                if m in avoid[p]:
                    bad("avoid", s, f"{p}에게 기피 메뉴 제공 ({CATEGORY_KO[c]}, 메뉴 이름은 개인정보 보호로 생략)")
            if sorted(served) != sorted(can_eat):
                bad("structure", s, f"{menu['name']} 먹는 사람 {served} ≠ 계산 {can_eat}")
            if not can_eat:
                bad("structure", s, f"아무도 먹을 수 없는 {menu['name']} 을(를) 차림")
            packs = math.ceil(len(can_eat) / menu["servings_per_pack"])
            cost = packs * menu["price_per_pack"]
            if item.get("packs") != packs or item.get("cost") != cost:
                bad("cost", s, f"{menu['name']} 팩/금액 {item.get('packs')}/{item.get('cost')} ≠ 계산 {packs}/{cost}")
            meal_cost += cost
            use[m] = use.get(m, 0) + 1
            for p in can_eat:
                kcal.setdefault(p, {}).setdefault(day_of(s), 0)
                kcal[p][day_of(s)] += menu["kcal_per_serving"]

        # 대체 메뉴: 단일 구성을 기피하는 출석자마다 정확히 1개, 같은 구성, 기피하지 않는 메뉴
        subs = meal.get("substitutes", [])
        for c in SINGLE_CATEGORIES:
            base = (meal.get(c) or {}).get("menu_id")
            need = sorted(p for p in A if base in avoid[p])
            got = sorted(x["member"] for x in subs if x.get("category") == c)
            if need != got:
                bad("avoid", s, f"{CATEGORY_KO[c]} 대체 대상 {need} ≠ 대체 받은 사람 {got}")
        for x in subs:
            m = x.get("menu_id")
            if m not in menus:
                bad("unknown_menu", s, f"대체 메뉴가 카탈로그에 없음: {x}")
                continue
            if x.get("member") not in A:
                bad("structure", s, f"출석하지 않은 {x.get('member')}에게 대체 메뉴")
                continue
            if menus[m]["category"] != x.get("category"):
                bad("structure", s, f"대체 메뉴 {menus[m]['name']} 이(가) 같은 구성이 아님")
            if m in avoid[x["member"]]:
                bad("avoid", s, f"{x['member']}의 대체 메뉴도 기피 메뉴 ({CATEGORY_KO[x['category']]}, 메뉴 이름은 개인정보 보호로 생략)")
            if x.get("packs") != 1 or x.get("cost") != menus[m]["price_per_pack"]:
                bad("cost", s, f"대체 {menus[m]['name']} 팩/금액 {x.get('packs')}/{x.get('cost')} ≠ 1/{menus[m]['price_per_pack']}")
            meal_cost += menus[m]["price_per_pack"]
            kcal.setdefault(x["member"], {}).setdefault(day_of(s), 0)
            kcal[x["member"]][day_of(s)] += menus[m]["kcal_per_serving"]

        # 부반찬 최소 1개
        for p in A:
            if not any(i.get("menu_id") in menus and i["menu_id"] not in avoid[p] for i in meal.get("sub_sides", [])):
                bad("avoid", s, f"{p}가 먹을 수 있는 부반찬이 없음")
        if meal.get("cost") != meal_cost:
            bad("cost", s, f"끼니 금액 {meal.get('cost')} ≠ 계산 {meal_cost}")
        total += meal_cost
        pair = ((meal.get("soup") or {}).get("menu_id"), (meal.get("main_side") or {}).get("menu_id"))
        overlap += pair in hist

    # 예산·총액
    if plan.get("total_cost") != total:
        bad("cost", None, f"총액 {plan.get('total_cost')} ≠ 계산 {total}")
    if total > plan["budget"]:
        bad("budget", None, f"총액 {total:,}원이 예산 {plan['budget']:,}원 초과")

    # 칼로리 (사람 × 요일)
    for p in member_ids():
        for d in fam["days"]:
            n = sum(p in att[f"{d}-{m}"] for m in fam["meals"])
            got = kcal.get(p, {}).get(d, 0)
            if n:
                lo, hi = kcal_bounds(p, n)
                if not lo <= got <= hi:
                    bad("kcal", f"{d}", f"{p} {got}kcal 이 범위 {lo}~{hi}kcal 밖 ({n}끼)")
            reported = plan.get("daily_kcal", {}).get(p, {}).get(d, 0)
            if reported != got:
                bad("kcal", f"{d}", f"{p} 보고 {reported}kcal ≠ 계산 {got}kcal")

    # 최근 3주 조합 중복
    limit = math.floor(len(expected_slots) * HISTORY_OVERLAP_RATIO)
    if overlap > limit:
        bad("history", None, f"최근 3주와 겹치는 (국, 메인반찬) 조합 끼니 {overlap} > 상한 {limit}")
    rep = plan.get("repeat_from_history", {})
    if rep.get("count") != overlap or rep.get("limit") != limit:
        bad("history", None, f"보고 {rep} ≠ 계산 count {overlap}, limit {limit}")

    # 주 안 반복
    caps = cons.get("repeat_cap", {})
    for m, n in sorted(use.items()):
        cap = caps.get(menus[m]["staple_type"], caps.get("기본", REPEAT_CAP))
        if n > cap:
            bad("repeat", None, f"{menus[m]['name']} 주 {n}회 > 상한 {cap}회")

    counts = {}
    for x in v:
        counts[x["type"]] = counts.get(x["type"], 0) + 1
    return {"ok": not v, "checked": True, "violation_count": len(v), "by_type": counts,
            "violations": v, "total_cost": total, "budget": plan["budget"],
            "history_overlap": {"count": overlap, "limit": limit}}


def make_plan(budget=None, attendance_changes=None, extra_avoid=None, cost_weight=1, week_id=None,
              min_history_overlap=False) -> dict:
    """식단을 짜서 Plan JSON 을 돌려준다(plan_week 도구의 본체).

    결과 status
      ok              : 모든 조건을 지킨 식단 (Plan JSON)
      no_allowed_menu : 사전 검사에서 누군가 먹을 메뉴가 없음 (풀지 않음)
      infeasible      : 조건을 모두 지킬 수 없음. relaxations 에 무엇을 얼마나 풀면 되는지 담는다 (식단은 주지 않음)
      timeout         : 시간 제한 안에 답을 못 찾음
    """
    ctx = build_context(budget, attendance_changes, extra_avoid, cost_weight, week_id, min_history_overlap)
    base ={"week_id": ctx["week_id"], "budget": ctx["budget"]}
    pre = precheck(ctx)
    if pre:
        return {**pre, **base}
    model = build_model(ctx)
    res = solve_model(model)
    if res["status"] == "Infeasible":
        relaxed = build_model(ctx, relax=True)
        rres = solve_model(relaxed, gap=RELAX_GAP)
        if rres["status"] != "Optimal":
            return {"status": "timeout", **base, "message": "완화안을 시간 안에 찾지 못했습니다."}
        return {"status": "infeasible", **base, "solve_seconds": round(res["seconds"] + rres["seconds"], 2),
                "relaxations": describe_relaxations(ctx, relaxed),
                "message": "조건을 모두 지키는 식단이 없습니다. 아래 중 하나 이상을 풀면 식단을 짤 수 있습니다."}
    if res["status"] != "Optimal":
        return {"status": "timeout", **base, "message": "시간 안에 식단을 찾지 못했습니다."}
    plan = format_plan(ctx, read_solution(ctx, model))
    plan["solve_seconds"] = res["seconds"]
    plan["reasons_file"] = save_meal_reasons(plan)
    store_plan(plan)
    return plan


# ---------------------------------------------------------------------------
# 에이전트 도구 (@tool)
# ---------------------------------------------------------------------------

def _dumps(obj) -> str:
    """도구 결과를 한국어가 그대로 보이는 JSON 문자열로 바꾼다."""
    return json.dumps(obj, ensure_ascii=False)


def summarize_plan(plan: dict) -> str:
    """Plan JSON 을 LLM 이 읽기 쉬운 짧은 텍스트로 줄인다. 숫자는 Plan JSON 값을 그대로 옮긴다(새로 계산하지 않음)."""
    if plan["status"] != "ok":
        return _dumps(plan)
    lines = [f"status: ok | plan_id: {plan['plan_id']} | week_id: {plan['week_id']} | "
             f"총액 {plan['total_cost']:,}원 / 예산 {plan['budget']:,}원 | "
             f"최근 3주와 겹친 국+메인 조합 {plan['repeat_from_history']['count']}끼 (상한 {plan['repeat_from_history']['limit']}끼)",
             f"조건: {_dumps(plan['constraints'])}",
             *([f"주의: {plan['week_id']} 는 현재 주({current_week()})보다 지난 주차라 보기만 가능하고 확정(저장)할 수 없음"]
               if plan.get("is_past_week") else []),
             "끼니별 이유는 파일에 저장됨. 주간 이유 요약은 weekly_summary(plan_id) 로 받는다.",
             "끼니 | 먹는 사람 | 주식, 국, 메인반찬 / 부반찬 3개 | 대체"]
    for m in plan["meals"]:
        subs = ", ".join(f"{x['member']}→{x['name']}" for x in m["substitutes"]) or "-"
        lines.append(f"{m['slot']} | {'·'.join(m['attendees'])} | {m['staple']['name']}, {m['soup']['name']}, "
                     f"{m['main_side']['name']} / {'·'.join(s['name'] for s in m['sub_sides'])} | {subs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 주간 이유 요약과 끼니별 이유 파일
# ---------------------------------------------------------------------------

def week_summary(plan: dict) -> list:
    """Plan JSON 에서 한 주 식단의 이유를 코드로 계산해 요약 문장 목록으로 돌려준다(LLM 이 세지 않는다).

    상차림·총액, 집에서 안 먹는 끼니, 선호 반영, 기피 대체 횟수, 다양성, 칼로리 검사 결과를 담는다.
    개인정보 보호: 기피 항목이 무엇인지는 쓰지 않고 대체 횟수만 쓴다.
    """
    meals = plan["meals"]
    extra = plan["constraints"].get("extra_avoid", {})
    att = attendance(plan["constraints"].get("attendance_changes"))
    items = lambda m: [m["staple"], m["soup"], m["main_side"], *m["sub_sides"]]  # noqa: E731
    lines = [f"상차림: {len(meals)}끼, {sum(len(m['attendees']) for m in meals)}인분, "
             f"총액 {plan['total_cost']:,}원 / 예산 {plan['budget']:,}원"]
    absent = {p: sum(p not in att[s] for s in slots()) for p in member_ids()}
    if any(absent.values()):
        lines.append("집에서 안 먹는 끼니: " + ", ".join(f"{p} {n}끼" for p, n in absent.items() if n)
                     + " — 그 끼니는 먹는 사람의 선호를 기준으로 골랐습니다")
    pref = []
    for p in member_ids():
        mine = [m for m in meals if p in m["attendees"]]
        hit = sum(any(p in i["eaters"] and pref_score(p, i["menu_id"], extra) > 0 for i in items(m))
                  or any(x["member"] == p and pref_score(p, x["menu_id"], extra) > 0 for x in m["substitutes"])
                  for m in mine)
        pref.append(f"{p} {hit}/{len(mine)}끼")
    lines.append("선호 메뉴가 들어간 끼니: " + ", ".join(pref))
    subs = {p: sum(x["member"] == p for m in meals for x in m["substitutes"]) for p in member_ids()}
    lines.append(f"기피 반영: 기피하는 메뉴 대신 같은 종류의 대체 메뉴를 {sum(subs.values())}번 준비했습니다 ("
                 + ", ".join(f"{p} {n}번" for p, n in subs.items() if n) + "). 부반찬은 사람마다 먹을 수 있는 것이 1개 이상입니다")
    use = {}
    for m in meals:
        for i in items(m):
            use[i["menu_id"]] = use.get(i["menu_id"], 0) + 1
    rep = plan["repeat_from_history"]
    lines.append(f"다양성: 최근 3주({', '.join(plan['constraints']['history_weeks'])})와 겹친 국+메인반찬 조합 "
                 f"{rep['count']}끼(상한 {rep['limit']}끼), 이번 주 서로 다른 메뉴 {len(use)}가지, 같은 메뉴는 주 2회까지(밥 3회)")
    v = verify_plan(plan)
    lines.append("칼로리·조건 검사: " + ("모든 구성원의 요일별 칼로리가 각자 범위 안이고 기피·예산·중복·반복 위반이 없습니다"
                                       if v["ok"] else f"위반 {v['violation_count']}건 {v['by_type']}"))
    if plan.get("is_past_week"):
        lines.append(f"참고: {plan['week_id']} 는 지난 주차라 보기만 가능하고 확정(저장)할 수 없습니다")
    return lines


def save_meal_reasons(plan: dict) -> str:
    """끼니별 상세 이유와 주간 요약을 REASONS_DIR 에 저장한다.

    <plan_id>.json 은 weekly_summary 도구가 읽는 용도, <plan_id>.md 는 사람이 읽는 정리본이다.
    돌려주는 값은 .md 파일의 경로(프로젝트 기준 상대 경로)다.
    """
    folder = reasons_dir()
    folder.mkdir(parents=True, exist_ok=True)
    summary = week_summary(plan)
    meals = [{"slot": m["slot"], "attendees": m["attendees"],
              "menus": [m["staple"]["name"], m["soup"]["name"], m["main_side"]["name"],
                        *[s["name"] for s in m["sub_sides"]]],
              "substitutes": [{"member": x["member"], "name": x["name"]} for x in m["substitutes"]],
              "cost": m["cost"], "reason": m["reason"]} for m in plan["meals"]]
    data = {"plan_id": plan["plan_id"], "week_id": plan["week_id"], "weekly_summary": summary, "meals": meals}
    (folder / f"{plan['plan_id']}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# {plan['week_id']} 끼니별 식단 이유", "",
          f"- plan_id: {plan['plan_id']}", f"- 총액 {plan['total_cost']:,}원 / 예산 {plan['budget']:,}원", "",
          "## 주간 요약", "", *[f"- {s}" for s in summary], "", "## 끼니별 이유", ""]
    for m in meals:
        who = "전체" if len(m["attendees"]) == len(member_ids()) else "·".join(m["attendees"])
        menus = f"{'·'.join(m['menus'][:3])} / {'·'.join(m['menus'][3:])}"
        subs = ", ".join(f"{x['member']}→{x['name']}" for x in m["substitutes"]) or "없음"
        md += [f"### {m['slot']} ({who})", "", f"- 메뉴: {menus}", f"- 대체: {subs}",
               f"- 금액: {m['cost']:,}원", f"- 이유: {m['reason']}", ""]
    path = folder / f"{plan['plan_id']}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


class AttendanceChange(BaseModel):
    """출석 변경 한 건."""
    member: str = Field(description="구성원 이름(아빠, 엄마, 첫째, 둘째) 또는 '전체'")
    day: Literal["월", "화", "수", "목", "금", "토", "일"] = Field(description="요일")
    meal: Optional[Literal["아침", "점심", "저녁"]] = Field(default=None, description="끼니. 생략하면 그날 세 끼 모두")
    eats: bool = Field(default=False, description="집에서 먹으면 true, 안 먹으면 false")


class PlanWeekInput(BaseModel):
    """plan_week 입력."""
    budget: Optional[int] = Field(default=None, description="주간 예산(원). 질문에 없으면 생략 → 가족 기본 예산")
    attendance_changes: list[AttendanceChange] = Field(
        default_factory=list, description="이번 주 출석 변경. 예: 첫째가 수요일 저녁 안 먹음 → {member:첫째, day:수, meal:저녁, eats:false}")
    extra_avoid: dict[str, list[str]] = Field(
        default_factory=dict, description="이번 요청에서 추가로 뺄 메뉴. {구성원 또는 '전체': [태그 또는 menu_id]}. 예: 이번 주는 매운 거 빼고 → {'전체': ['매운맛']}")
    cost_weight: Literal[1, 5] = Field(default=1, description="1: 선호 우선(기본), 5: '최대한 싸게' 요청일 때")
    min_history_overlap: bool = Field(default=False, description="'지난 식단과 최대한 겹치지 않게' 요청이면 true → 최근 3주와 (국, 메인반찬) 조합이 겹치는 끼니를 최소로")
    week_id: Optional[str] = Field(default=None, description="계획할 주 'YYYY-Www'. 같은 대화에서 다시 짤 때는 직전 식단의 week_id 를 그대로 넣는다. 처음이면 생략")


@tool("plan_week", args_schema=PlanWeekInput, response_format="content_and_artifact")
def plan_week(budget=None, attendance_changes=None, extra_avoid=None, cost_weight=1, week_id=None,
              min_history_overlap=False):
    """가족 1주(21끼) 식단을 최적화 도구로 짠다. 식단을 짜거나 다시 짜 달라는 요청에는 반드시 이 도구를 쓴다.

    결과 status: ok(식단과 끼니별 이유·금액·칼로리), infeasible(조건을 모두 지킬 수 없음 → relaxations 에
    무엇을 얼마나 풀면 되는지), no_allowed_menu(누군가 먹을 메뉴가 없음), timeout.
    금액·칼로리 숫자는 이 결과에 있는 값만 그대로 인용한다.
    """
    changes = [c.model_dump() if isinstance(c, BaseModel) else c for c in attendance_changes or []]
    try:
        plan = make_plan(budget, changes, extra_avoid or None, cost_weight, week_id, min_history_overlap)
    except (KeyError, ValueError) as e:
        err = {"status": "error", "message": str(e).strip("'\"")}
        return _dumps(err), err
    return summarize_plan(plan), plan


@tool
def check_plan(plan_id: str) -> str:
    """plan_week 가 만든 식단(plan_id)을 독립적으로 다시 계산해 기피·예산·칼로리·중복·반복 위반이 없는지 검사한다.

    Args:
        plan_id: plan_week 결과의 plan_id
    """
    plan = PLAN_STORE.get(plan_id)
    if not plan:
        return _dumps({"ok": False, "message": f"plan_id {plan_id} 식단을 찾지 못했습니다. 서버가 재시작됐거나 다른 대화의 식단일 수 있으니 plan_week 로 다시 짜 주세요."})
    r = verify_plan(plan)
    return _dumps({k: r[k] for k in ("ok", "violation_count", "by_type", "violations", "total_cost", "budget", "history_overlap")})


def family_info(member: Optional[str] = None) -> dict:
    """구성원 프로필(선호·칼로리 범위·집에서 안 먹는 끼니·메모)과 기본 예산을 돌려준다. 기피 항목은 개수만 알린다."""
    fam = load_family()
    targets = [get_member(member)] if member else fam["members"]
    out = []
    for m in targets:
        absent = [s for s in slots() if m["member_id"] not in attendance()[s]]
        prefer = {load_menus()[k]["name"] if k in load_menus() else k: v for k, v in m["prefer"].items()}
        out.append({"member": m["member_id"], "role": m["role"],
                    "daily_kcal_min": m["daily_kcal_min"], "daily_kcal_max": m["daily_kcal_max"],
                    "prefer": prefer,
                    "avoid": f"비공개 {len(m['avoid'])}개 (건강 정보일 수 있어 공개하지 않음, 식단 계산에는 반영됨)",
                    "absent_meals": absent, "note": m.get("note", "")})
    return {"default_weekly_budget": fam["default_weekly_budget"], "members": out}


@tool
def get_family_info(member: Optional[str] = None) -> str:
    """가족 구성원의 선호 음식, 하루 칼로리 범위, 집에서 안 먹는 끼니, 기본 주간 예산을 조회한다.

    기피 음식은 개인정보 보호를 위해 무엇인지 알려 주지 않는다(개수만). 식단 계산에는 반영되어 있다.

    Args:
        member: 아빠, 엄마, 첫째, 둘째 중 하나. 생략하면 가족 전체
    """
    try:
        return _dumps(family_info(member))
    except KeyError as e:
        return _dumps({"error": str(e).strip("'\"")})


def save_block_reason(plan_id: Optional[str]) -> Optional[str]:
    """식단을 저장할 수 없는 이유를 돌려준다. 저장할 수 있으면 None.

    없는 plan_id, 현재 주보다 지난 주차, 이미 확정된 주차는 저장하지 않는다.
    에이전트는 이 값이 None 일 때만 사용자에게 저장 확인을 묻는다.
    """
    plan = PLAN_STORE.get(plan_id or "")
    if not plan:
        return f"plan_id {plan_id} 식단을 찾지 못했습니다. 서버가 재시작됐거나 다른 대화의 식단일 수 있으니 plan_week 로 다시 짠 뒤 확정해 주세요."
    if is_past_week(plan["week_id"]):
        return (f"{plan['week_id']} 는 현재 주({current_week()})보다 지난 주차라, 식단을 다시 짜서 볼 수는 있지만 "
                f"확정(저장)할 수 없습니다.")
    if (history_dir() / f"{plan['week_id']}.json").exists():
        return f"{plan['week_id']} 식단이 이미 확정되어 있어 덮어쓰지 않았습니다."
    return None


@tool
def save_plan(plan_id: str) -> str:
    """사용자가 "확정"이라고 말했을 때, 그 대화에서 만든 식단(plan_id)을 기록 폴더(history)에 저장한다.

    이 도구를 부르면 서버가 실행 전에 사용자에게 저장 여부를 한 번 더 묻고, "예"일 때만 실제로 저장된다.
    저장한 식단은 다음 주 식단을 짤 때 '최근 3주' 비교 대상이 된다.
    현재 주보다 지난 주차, 이미 확정된 주차는 저장하지 않고 이유를 돌려준다.

    Args:
        plan_id: 확정할 식단의 plan_id (같은 대화의 직전 plan_week 결과)
    """
    reason = save_block_reason(plan_id)
    if reason:
        plan = PLAN_STORE.get(plan_id, {})
        return _dumps({"saved": False, "week_id": plan.get("week_id"), "message": reason})
    if not SAVE_ALLOWED.get():
        return _dumps({"saved": False, "message": "사용자의 저장 승인('예')이 없어 저장하지 않았습니다."})
    plan = PLAN_STORE[plan_id]
    path = history_dir() / f"{plan['week_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return _dumps({"saved": True, "week_id": plan["week_id"], "plan_id": plan_id,
                   "message": f"{plan['week_id']} 식단을 확정해 저장했습니다."})


def past_plan(week_id: Optional[str] = None) -> dict:
    """기록 폴더에서 확정된 지난 식단을 읽어 끼니별 메뉴(id·이름)와 먹은 사람으로 정리한다.

    week_id 를 생략하면 기록에 있는 가장 최근 주(= 이번에 계획할 주의 바로 앞 주)를 읽는다.
    """
    week_id = week_id or previous_weeks(default_target_week(), 1)[0]
    data = load_history_week(week_id)
    if not data["meals"]:
        return {"week_id": week_id, "found": False, "saved_weeks": history_weeks_saved(),
                "message": f"{week_id} 에 확정된 식단 기록이 없습니다."}

    def ref(item):
        return {"menu_id": item["menu_id"], "name": item["name"]}

    meals = [{"slot": m["slot"], "attendees": m.get("attendees", []),
              **{c: ref(m[c]) for c in SINGLE_CATEGORIES},
              "sub_sides": [ref(s) for s in m["sub_sides"]],
              "substitutes": [{"member": s["member"], "category": s["category"], **ref(s)} for s in m.get("substitutes", [])]}
             for m in data["meals"]]
    out = {"week_id": week_id, "found": True, "meals": meals}
    if "total_cost" in data:
        out["total_cost"] = data["total_cost"]
    return out


@tool(response_format="content_and_artifact")
def get_past_plan(week_id: Optional[str] = None):
    """확정되어 기록된 지난 식단을 조회한다. "지난주 식단이 뭐였지?" 같은 질문에 쓴다.

    Args:
        week_id: 조회할 주 'YYYY-Www'. 생략하면 가장 최근에 기록된 주(이번에 계획할 주의 바로 앞 주)
    """
    try:
        r = past_plan(week_id)
    except ValueError as e:
        r = {"found": False, "message": f"주차 형식이 잘못되었습니다(예: 2026-W47): {e}"}
    if not r.get("found"):
        return _dumps(r), r
    lines = [f"week_id: {r['week_id']} (확정된 기록)" + (f" | 총액 {r['total_cost']:,}원" if "total_cost" in r else ""),
             "끼니 | 먹은 사람 | 주식, 국, 메인반찬 / 부반찬 3개 | 대체"]
    for m in r["meals"]:
        subs = ", ".join(f"{x['member']}→{x['name']}" for x in m["substitutes"]) or "-"
        lines.append(f"{m['slot']} | {'·'.join(m['attendees'])} | {m['staple']['name']}, {m['soup']['name']}, "
                     f"{m['main_side']['name']} / {'·'.join(s['name'] for s in m['sub_sides'])} | {subs}")
    return "\n".join(lines), r


@tool
def weekly_summary(plan_id: str, slot: Optional[str] = None) -> str:
    """plan_week 가 만든 식단의 '왜 이렇게 짰는지'를 돌려준다. 식단을 짠 뒤 답변의 이유 요약에 반드시 쓴다.

    slot 을 생략하면 한 주 전체의 이유 요약(상차림·총액, 집에서 안 먹는 끼니, 선호 반영, 기피 대체 횟수,
    다양성, 칼로리 검사)을, slot 을 주면 그 끼니의 상세 이유를 끼니별 이유 파일에서 읽어 돌려준다.
    숫자는 모두 코드가 계산한 값이므로 그대로 인용한다.

    Args:
        plan_id: plan_week 결과의 plan_id
        slot: 끼니 하나의 상세 이유가 필요할 때 '요일-끼니' (예: '화-저녁')
    """
    path = reasons_dir() / f"{plan_id}.json"
    plan = PLAN_STORE.get(plan_id)
    if plan is None and not path.exists():
        return _dumps({"found": False, "message": f"plan_id {plan_id} 식단을 찾지 못했습니다. plan_week 로 다시 짜 주세요."})
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if slot:
        meals = saved["meals"] if saved else []
        meal = next((m for m in meals if m["slot"] == slot), None)
        if not meal:
            return _dumps({"found": False, "message": f"{slot} 은(는) 이 식단에서 상을 차리지 않는 끼니이거나 잘못된 끼니 이름입니다."})
        return _dumps({"found": True, "plan_id": plan_id, "slot": slot, **{k: meal[k] for k in ("attendees", "menus", "substitutes", "cost", "reason")}})
    lines = week_summary(plan) if plan else saved["weekly_summary"]
    return _dumps({"found": True, "plan_id": plan_id, "week_id": (plan or saved)["week_id"], "weekly_summary": lines,
                   "meal_reasons_file": (plan or {}).get("reasons_file") or str(path.with_suffix(".md"))})


DOMAIN_TOOLS = [plan_week, check_plan, get_family_info, save_plan, get_past_plan, weekly_summary]
