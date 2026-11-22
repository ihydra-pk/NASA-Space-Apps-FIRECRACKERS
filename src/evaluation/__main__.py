"""Run the benchmark against the same Explore service the web page uses.

    python -m src.evaluation                     # retrieval and answers
    python -m src.evaluation --retrieval-only    # router and search only, no answer call
    python -m src.evaluation --ids cool-flame-extinction-cause,abstain-bone-density

Questions are embedded and answered by the gateway models, so this needs
UPB_AI_GATEWAY_KEY. FIRELENS_API_* variables change the settings exactly as
they do for ``python -m src.api``. Each run writes a JSON report to
``data/evaluation/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.app import build_explore_service
from src.api.conditions import extract_conditions
from src.explore.models import ExploreRequest

from .metrics import DEFAULT_QUESTIONS, load_questions, score_question, summarize


def _request(question: str) -> ExploreRequest:
    # The request the web page produces: the API reads stated conditions first.
    return ExploreRequest(question, extract_conditions(question) or None)


def _retrieval_only(service: Any, request: ExploreRequest) -> dict[str, Any]:
    from src.explore.evidence import document_evidence

    routing = service.router.route(request.question) if service.router is not None else None
    context = service.retrieve_documents(request.question, routing=routing)
    return {"document_evidence": [item.to_dict() for item in document_evidence(context)]}


def _line(index: int, total: int, result: dict[str, Any]) -> str:
    rank = result["gold_rank"] if result["gold_rank"] is not None else "-"
    cited = {True: "yes", False: "no", None: "-"}[result["gold_cited"]]
    text = (
        f"[{index:>2}/{total}] {result['id']:<36} status={result['status'] or '-':<21} "
        f"gold_rank={rank!s:>3} gold_cited={cited:<3} {result['seconds']:>6.1f}s"
    )
    return text + (f"  ERROR {result['error']}" if result["error"] else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FireLens retrieval and answers.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="route and search only; skip the answer model (needs the current Explore service)",
    )
    parser.add_argument("--label", default="run", help="prefix for the report file name")
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        unknown = wanted - {question.id for question in questions}
        if unknown:
            parser.error("unknown question id(s): " + ", ".join(sorted(unknown)))
        questions = [question for question in questions if question.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    service = build_explore_service()
    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        request = _request(question.question)
        start = time.perf_counter()
        try:
            response = (
                _retrieval_only(service, request) if args.retrieval_only else service.explore(request).to_dict()
            )
            error = None
        except Exception as exc:  # one failing question must not end the run
            response, error = None, f"{type(exc).__name__}: {exc}"
        result = score_question(
            question,
            response,
            error=error,
            seconds=time.perf_counter() - start,
            retrieval_only=args.retrieval_only,
        )
        results.append(result)
        print(_line(index, len(questions), result), flush=True)

    created = datetime.now(timezone.utc)
    report = {
        "label": args.label,
        "created": created.isoformat(timespec="seconds"),
        "retrieval_only": args.retrieval_only,
        "settings": {key: value for key, value in sorted(os.environ.items()) if key.startswith("FIRELENS_API_")},
        "summary": summarize(results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.label}-{created:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
