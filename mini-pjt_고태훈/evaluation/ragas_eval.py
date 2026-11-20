"""ragas 지표로 test_queries.csv 평가셋의 답변(answer)과 근거(contexts)를 채점한다.

실행: python evaluation/ragas_eval.py [--only 1,14] [--out evaluation/ragas_report.md] [--cache 경로.json]
                                     [--contexts agent|agent+tools] [--lang korean|english]

- 에이전트 호출은 run_eval.run() 을 그대로 쓴다(임시 history, TODAY 고정, 규칙 검사 결과도 함께 얻음).
- 채점 모델은 Bedrock 의 MODEL_ID(또는 --judge-model), 임베딩은 EMBED_MODEL_ID 를 쓴다. OpenAI 는 쓰지 않는다.
- 채점용 contexts (--contexts)
  - agent+tools(기본): 제출 응답의 contexts 앞에 이번 요청의 도구 결과 요약(식단 배치·총액, 완화안, 지난 식단,
    끼니 이유, 저장 결과)을 붙인다. 식단 답변의 근거는 도구 결과(trace)에 있어서다. 제출 응답의 contexts 는 바꾸지 않는다.
  - agent: 제출 응답의 contexts 만 쓴다.
- --lang korean(기본)이면 ragas 채점 프롬프트의 예시를 한국어로 바꿔(adapt_prompts) 한국어 질문·답변과 같은 언어로 채점한다.
- 지표
  - faithfulness: 답변 문장이 contexts 로 뒷받침되는 비율 (contexts 가 있는 문항만)
  - answer_relevancy: 답변이 질문에 맞는 정도 (모든 문항)
  - context_precision: 앞쪽 contexts 가 정답(reference)에 쓸모 있는 정도 (contexts 가 있는 문항, 앞 --precision-k 개만)
  - context_recall: 정답 문장이 contexts 로 뒷받침되는 비율 (contexts 가 있는 문항만)
  - factual_correctness: 정답(reference) 주장 중 답변에 담긴 비율(recall). 정답은 요약이고 식단 답변은 21끼를
    자세히 쓰므로 F1 은 자세한 답을 깎는다 (모든 문항)
"""
import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import run_eval  # noqa: E402  (src 경로·TODAY·.env 설정도 여기서 된다)
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse  # noqa: E402
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,  # noqa: E402
                           LLMContextRecall, ResponseRelevancy)

CTX_METRICS = ["faithfulness", "context_precision", "context_recall"]
ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]


def judge(model_id: str):
    """ragas 채점용 LLM·임베딩 래퍼를 만든다. 리전은 .env 의 AWS_DEFAULT_REGION."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    llm = ChatBedrockConverse(model=model_id, region_name=region, temperature=0, max_tokens=16000)  # 21끼 답변은 문장이 60개가 넘어 4096 이면 채점 출력이 잘린다
    emb = BedrockEmbeddings(model_id=os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"), region_name=region)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def load_or_run(only, cache: Path | None) -> list:
    """에이전트 응답을 모은다. --cache 파일이 있으면 그것을 읽고, 없으면 run_eval.run() 으로 만들어 저장한다."""
    if cache and cache.exists():
        results = json.loads(cache.read_text(encoding="utf-8"))
        refs = {r["id"]: r.get("reference", "") for r in csv.DictReader(open(run_eval.CSV_PATH, encoding="utf-8"))}
        for r in results:  # 정답은 CSV 최신 값으로 (캐시 뒤에 정답을 고쳐도 반영)
            r["reference"] = refs.get(r["id"], r.get("reference", ""))
        return [r for r in results if not only or r["id"] in only]
    results = run_eval.run(only)
    if cache:
        cache.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def _meal_line(meal: dict) -> str:
    """Plan JSON 끼니 하나를 채점용 근거 한 줄로 만든다. 대체는 구성과 메뉴만 쓴다(기피 항목은 쓰지 않는다)."""
    subs = ", ".join(f"{s['member']} {s['category']} 대체 {s['name']}" for s in meal.get("substitutes", []))
    return (f"[meal:{meal['slot']}] 먹는 사람 {'·'.join(meal['attendees'])} | 주식 {meal['staple']['name']} | "
            f"국 {meal['soup']['name']} | 메인반찬 {meal['main_side']['name']} | "
            f"부반찬 {', '.join(x['name'] for x in meal['sub_sides'])}" + (f" | {subs}" if subs else ""))


def tool_contexts(trace: list) -> list:
    """이번 요청 trace 의 도구 결과를 채점용 근거 문장 목록으로 바꾼다(검색·가족 정보는 제출 contexts 에 이미 있어 뺀다)."""
    out = []
    for step in trace:
        res, tool = step.get("output"), step.get("step")
        if tool == "plan_week" and isinstance(res, dict):
            if res.get("status") == "ok":
                rh = res.get("repeat_from_history", {})
                c = res.get("constraints", {})
                cond = [f"'최대한 싸게'(cost_weight {c.get('cost_weight')})" if c.get("cost_weight", 1) != 1 else "",
                        f"이번 요청 추가로 뺄 음식 {c['extra_avoid']}" if c.get("extra_avoid") else "",
                        f"출석 변경 {c['attendance_changes']}" if c.get("attendance_changes") else "",
                        "최근 3주와 겹침 최소화" if c.get("min_history_overlap") else ""]
                if any(cond):  # 요청 조건 (사용자가 이번 질문에서 직접 말한 것만, 가족 기피는 쓰지 않는다)
                    out.append(f"[plan_conditions:{res['week_id']}] " + " | ".join(x for x in cond if x))
                out.append(f"[plan:{res['week_id']}] 식단 {len(res['meals'])}끼 | 총액 {res['total_cost']:,}원 | 예산 {res['budget']:,}원 | "
                           f"최근 3주({', '.join(res['constraints']['history_weeks'])})와 겹친 끼니 {rh.get('count')}개(한도 {rh.get('limit')}) | "
                           f"지난 주차 {res.get('is_past_week')} | plan_id {res['plan_id']}")
                out += [_meal_line(m) for m in res["meals"]]
            else:
                out.append(f"[plan:{res.get('week_id')}] status {res.get('status')} | 예산 {res.get('budget')} | "
                           + " / ".join(r.get("message", "") for r in res.get("relaxations", [])))
        elif tool == "get_past_plan" and isinstance(res, dict) and res.get("found"):
            out.append(f"[past:{res['week_id']}] 기록된 지난 식단 {len(res['meals'])}끼")
            out += [_meal_line(m) for m in res["meals"]]
        elif tool in ("weekly_summary", "save_plan", "check_plan") and res:
            text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
            out.append(f"[{tool}] {text[:4000]}")
    return out


def score(results: list, model_id: str, precision_k: int, workers: int, ctx_mode: str, lang: str) -> list:
    """문항별 ragas 점수를 계산해 results 각 항목에 'ragas' dict 로 붙인다. 해당 없는 지표는 None."""
    llm, emb = judge(model_id)
    for r in results:  # 채점용 contexts. 제출 응답의 contexts 는 r["contexts"] 그대로 둔다
        r["eval_contexts"] = (tool_contexts(r.get("trace", [])) if ctx_mode == "agent+tools" else []) + [
            c["text"] if isinstance(c, dict) else c for c in r["contexts"]]
    cfg = RunConfig(max_workers=workers, timeout=600, max_retries=5)
    for r in results:
        r["ragas"] = {m: None for m in ALL_METRICS}

    def run_metrics(rows, metrics, k=None):
        """rows 에 metrics 를 적용하고 점수를 rows 에 써 넣는다."""
        if not rows:
            return
        if lang != "english":  # 채점 프롬프트 예시를 질문 언어로 바꾼다
            for metric in metrics:
                metric.set_prompts(**asyncio.run(metric.adapt_prompts(language=lang, llm=llm)))
        samples = [SingleTurnSample(user_input=r["question"], response=r["answer"] or "(빈 답변)",
                                    retrieved_contexts=(r["eval_contexts"][:k] if k else r["eval_contexts"]),
                                    reference=r["reference"]) for r in rows]
        df = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=llm, embeddings=emb,
                      run_config=cfg, show_progress=True).to_pandas()
        for r, (_, s) in zip(rows, df.iterrows()):
            for col in df.columns:
                name = next((m for m in ALL_METRICS if m in col), col)  # 예: llm_context_precision_with_reference, factual_correctness(mode=f1)
                if name in ALL_METRICS:
                    v = s[col]
                    r["ragas"][name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    with_ctx = [r for r in results if r["eval_contexts"]]
    print(f"\n[ragas] 전체 {len(results)}문항: answer_relevancy, factual_correctness", flush=True)
    run_metrics(results, [ResponseRelevancy(), FactualCorrectness(mode="recall")])
    print(f"[ragas] contexts 있는 {len(with_ctx)}문항: faithfulness, context_recall", flush=True)
    run_metrics(with_ctx, [Faithfulness(), LLMContextRecall()])
    print(f"[ragas] contexts 있는 {len(with_ctx)}문항: context_precision (앞 {precision_k}개)", flush=True)
    run_metrics(with_ctx, [LLMContextPrecisionWithReference()], k=precision_k)
    return results


def _fmt(v) -> str:
    """점수 표기. 해당 없으면 '-'."""
    return "-" if v is None else f"{v:.2f}"


def _mean(vals: list):
    """None 을 뺀 평균. 값이 없으면 None."""
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def write_outputs(results: list, out: Path, model_id: str, precision_k: int, ctx_mode: str, lang: str) -> None:
    """마크다운 리포트와 문항별 점수 CSV(같은 이름 .csv)를 쓴다."""
    types = sorted({r["type"] for r in results})
    lines = [
        "# ragas 평가 리포트",
        "",
        f"- 실행 시각: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 에이전트 모델: `{os.environ.get('MODEL_ID')}` / 채점 모델: `{model_id}` / 임베딩: `{os.environ.get('EMBED_MODEL_ID')}`",
        f"- 문항 수: {len(results)} (채점용 contexts 있는 문항 {sum(bool(r['eval_contexts']) for r in results)})",
        f"- 채점용 contexts: `{ctx_mode}`"
        + (" (도구 결과 요약 + 제출 contexts. 제출 응답의 contexts 는 그대로)" if ctx_mode == "agent+tools" else " (제출 contexts 만)"),
        f"- 채점 프롬프트 언어: {lang}",
        f"- context_precision 은 채점용 contexts 앞 {precision_k}개로 계산",
        "",
        "## 지표 평균",
        "",
        "| 지표 | 전체 | " + " | ".join(types) + " |",
        "|---|---|" + "---|" * len(types),
    ]
    for m in ALL_METRICS:
        row = [_fmt(_mean([r["ragas"][m] for r in results]))]
        row += [_fmt(_mean([r["ragas"][m] for r in results if r["type"] == t])) for t in types]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines += [
        "",
        "- `-` 는 해당 문항이 없거나 지표를 적용하지 않은 경우 (contexts 가 없는 거절·확정 응답은 faithfulness·context 지표 제외)",
        "- factual_correctness(recall)는 ragas 정의상 tp 를 '답변 주장 중 정답에 있는 것'으로 센다. 그래서 답변이 정답보다 많은 항목을 "
        "나열하면(q14 국 10개, q15 21끼) 정답 내용을 모두 담아도 0에 가깝게 나온다",
        "- 제출 contexts 만 쓰면 식단 답변의 끼니 배치·총액 근거(도구 결과)가 빠져 faithfulness·context_recall 이 구조적으로 낮게 나온다",
        "",
        "## 문항별 점수",
        "",
        "| id | 유형 | 규칙 검사 | " + " | ".join(ALL_METRICS) + " | 질문 |",
        "|---|---|---|" + "---|" * len(ALL_METRICS) + "---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['type']} | {'통과' if r['pass'] else '실패'} | "
                     + " | ".join(_fmt(r["ragas"][m]) for m in ALL_METRICS) + f" | {r['question']} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "type", "rule_pass", *ALL_METRICS, "question"])
        for r in results:
            w.writerow([r["id"], r["type"], r["pass"], *[r["ragas"][m] for m in ALL_METRICS], r["question"]])


def main() -> None:
    """명령줄 인자를 읽어 에이전트를 실행하고 ragas 로 채점해 리포트를 쓴다."""
    ap = argparse.ArgumentParser(description="가족 식단 Agent ragas 평가")
    ap.add_argument("--only", help="실행할 질문 id (쉼표 구분). 같은 thread 의 질문은 함께 실행")
    ap.add_argument("--out", default=str(EVAL_DIR / "ragas_report.md"))
    ap.add_argument("--cache", help="에이전트 응답 저장/재사용 JSON 경로 (있으면 에이전트를 다시 부르지 않음)")
    ap.add_argument("--judge-model", default=os.environ.get("MODEL_ID"))
    ap.add_argument("--precision-k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rule-out", help="같은 에이전트 응답으로 run_eval 규칙 검사 리포트도 쓴다 (예: evaluation/round5_report.md)")
    ap.add_argument("--contexts", choices=["agent+tools", "agent"], default="agent+tools")
    ap.add_argument("--lang", default="korean", help="채점 프롬프트 언어 (english 면 변환하지 않음)")
    args = ap.parse_args()
    results = load_or_run(set(args.only.split(",")) if args.only else None, Path(args.cache) if args.cache else None)
    if args.rule_out:  # 에이전트를 한 번만 돌리고 규칙 검사 리포트도 남긴다
        run_eval.write_report(results, Path(args.rule_out))
    score(results, args.judge_model, args.precision_k, args.workers, args.contexts, args.lang)
    write_outputs(results, Path(args.out), args.judge_model, args.precision_k, args.contexts, args.lang)
    print(f"\n규칙 검사 통과 {sum(r['pass'] for r in results)}/{len(results)} → {args.out}")


if __name__ == "__main__":
    main()
