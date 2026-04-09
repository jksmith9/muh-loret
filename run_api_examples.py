"""
run_api_examples.py
-------------------
Iterates through every conversation in data/conversations.json,
POSTs each one to the running Belief Extraction API, and saves
both the raw response JSON and a human-readable summary to:

    tests/test_api/
        conversation_<id>/
            response.json       <- full API response
            summary.txt         <- quick-read digest

A combined index file is also written:
    tests/test_api/index.json   <- one-line entry per conversation

Usage (from repo root, with API running on :8000):
    .venv/Scripts/python run_api_examples.py
    .venv/Scripts/python run_api_examples.py --url http://localhost:8000 --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import httpx  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/conversations.json")
OUTPUT_DIR = Path("tests/test_api")
API_URL = "http://localhost:8000/api/v1/analyze"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_summary(conv_id: int, user_id: int, response: dict) -> str:
    """Produce a compact human-readable digest of one API response."""
    beliefs = response.get("beliefs", [])
    scores = response.get("downstream_scores", {})
    lines = [
        f"Conversation : {conv_id}",
        f"User         : {user_id}",
        f"Analyzed at  : {response.get('analyzed_at', '-')}",
        f"Messages     : {response.get('message_count', 0)} total "
        f"/ {response.get('user_message_count', 0)} user",
        "",
        "-- Beliefs ------------------------------------------------------------------",
    ]
    if not beliefs:
        lines.append("  (none extracted)")
    for i, b in enumerate(beliefs, 1):
        lines.append(
            f"  {i}. [{b.get('topic','?').upper()}] {b.get('belief','')}"
        )
        lines.append(
            f"     sentiment={b.get('sentiment','?')}  confidence={b.get('confidence',0):.2f}"
        )
        if b.get("evidence"):
            lines.append(f"     evidence: \"{b['evidence']}\"")
    lines += [
        "",
        "-- Downstream Scores --------------------------------------------------------",
        f"  Value score    : {scores.get('conversation_value_score', '-')} / 10",
        f"  Dominant theme : {scores.get('dominant_theme', '-')}",
        f"  Follow-up      : {scores.get('suggested_follow_up', '-')}",
        f"  Content tags   : {', '.join(scores.get('content_tags', []))}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(api_url: str, debug: bool, timeout: float) -> None:
    conversations = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    total = len(conversations)
    print(f"Found {total} conversations in {DATA_PATH}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    run_ts = datetime.now(timezone.utc).isoformat()

    for idx, convo in enumerate(conversations):
        conv_id = convo["ref_conversation_id"]
        user_id = convo["ref_user_id"]
        msg_count = len(convo.get("messages_list", []))

        print(
            f"[{idx + 1:02d}/{total}] conversation {conv_id} "
            f"(user {user_id}, {msg_count} msgs) ... ",
            end="",
            flush=True,
        )

        url = f"{api_url}{'?debug=true' if debug else ''}"
        try:
            resp = httpx.post(
                url,
                json=convo,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            status = resp.status_code
            if status == 200:
                data = resp.json()
                label = "OK"
            else:
                data = {"error": resp.text, "status_code": status}
                label = f"FAIL HTTP {status}"
        except Exception as exc:  # noqa: BLE001
            data = {"error": str(exc)}
            status = 0
            label = f"FAIL {exc}"

        print(label)

        # Save per-conversation output
        out_dir = OUTPUT_DIR / f"conversation_{conv_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "response.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if status == 200:
            summary = format_summary(conv_id, user_id, data)
            (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
        else:
            (out_dir / "summary.txt").write_text(
                f"ERROR - HTTP {status}\n{data.get('error', '')}", encoding="utf-8"
            )

        # Index entry
        index.append(
            {
                "run_timestamp": run_ts,
                "index": idx,
                "ref_conversation_id": conv_id,
                "ref_user_id": user_id,
                "message_count": msg_count,
                "http_status": status,
                "conversation_value_score": (
                    data.get("downstream_scores", {}).get("conversation_value_score")
                    if status == 200 else None
                ),
                "dominant_theme": (
                    data.get("downstream_scores", {}).get("dominant_theme")
                    if status == 200 else None
                ),
                "belief_count": len(data.get("beliefs", [])) if status == 200 else 0,
                "output_dir": str(out_dir),
            }
        )

    # Write combined index
    index_path = OUTPUT_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final summary
    ok = sum(1 for e in index if e["http_status"] == 200)
    fail = total - ok
    print(f"\n{'-' * 60}")
    print(f"Done. {ok}/{total} succeeded, {fail} failed.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Index:            {index_path.resolve()}")

    if fail:
        print("\nFailed conversations:")
        for e in index:
            if e["http_status"] != 200:
                print(f"  conversation {e['ref_conversation_id']} -> HTTP {e['http_status']}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-test the Belief Extraction API")
    parser.add_argument("--url", default=API_URL, help="API base URL for /api/v1/analyze")
    parser.add_argument("--debug", action="store_true", help="Include raw LLM output in responses")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds)")
    args = parser.parse_args()
    main(args.url, args.debug, args.timeout)
