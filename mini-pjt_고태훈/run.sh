#!/usr/bin/env bash
# 가족 1주 식단 Agent 실행 스크립트
#   ./run.sh install          의존성 설치
#   ./run.sh serve            API 서버 (POST /query, 기본 포트 8000, PORT 로 변경)
#   ./run.sh eval [이름]      인-아웃 세트 규칙 평가 → evaluation/<이름>_report.md (기본 latest)
#   ./run.sh ragas            ragas 지표 평가 → evaluation/ragas_report.md
# 사전 준비: .env.example 을 .env 로 복사하고 AWS 키를 채운다.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-serve}" in
  install) python -m pip install -r requirements.txt ;;
  serve)   python -m uvicorn src.agent:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  eval)    python evaluation/run_eval.py --out "evaluation/${2:-latest}_report.md" ;;
  ragas)   python evaluation/ragas_eval.py ;;
  *)       echo "사용법: ./run.sh [install|serve|eval [이름]|ragas]"; exit 1 ;;
esac
