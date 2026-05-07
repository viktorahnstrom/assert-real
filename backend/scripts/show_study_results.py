"""
Show XADE user study results.

Reads the SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY values from
backend/.env (no external deps — uses stdlib only) and queries the
study_results table via the Supabase REST API. The service role key
bypasses RLS so this sees every row.

Usage (works with any Python 3.10+ — no venv needed):

    py backend/scripts/show_study_results.py              # summary table
    py backend/scripts/show_study_results.py --comments   # also dump free-text comments
    py backend/scripts/show_study_results.py --detail p-abc12345
                                                          # full JSON for one participant
    py backend/scripts/show_study_results.py --csv > results.csv
                                                          # raw CSV export
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


def load_env(env_path: Path) -> dict[str, str]:
    """Minimal .env parser. Strips comments and surrounding quotes."""
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split("#", 1)[0].strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        out[key.strip()] = value
    return out


def fetch_results() -> list[dict]:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env = load_env(env_path)

    base_url = (env.get("SUPABASE_URL") or "").rstrip("/")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_KEY") or ""
    if not base_url or not key:
        sys.exit(f"SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in {env_path}")

    query = urllib.parse.urlencode({"select": "*", "order": "completed_at.asc"})
    url = f"{base_url}/rest/v1/study_results?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"Supabase returned HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}")


def fmt_pct(num: float | None) -> str:
    return f"{round((num or 0) * 100)}%"


def summary_table(rows: list[dict]) -> None:
    if not rows:
        print("No participants yet.")
        return

    print(f"{len(rows)} participants in study_results\n")

    header = f"{'ID':<14}{'Score':<8}{'Conf':<6}{'Trust':<7}{'Willing':<10}{'Comment'}"
    print(header)
    print("-" * len(header))

    for r in rows:
        pid = (r.get("participant_id") or "?")[:12]
        correct = r.get("correct_count", 0)
        total = r.get("total_images", 0)
        score = f"{correct}/{total}"
        conf = f"{r.get('self_confidence_rating', '?')}/5"
        trust = f"{r.get('trust_rating', '?')}/5"
        willing = (r.get("willingness_to_use") or "-")[:8]
        comment = r.get("comments") or ""
        comment_summary = f"{len(comment)} chars" if comment.strip() else "-"
        print(f"{pid:<14}{score:<8}{conf:<6}{trust:<7}{willing:<10}{comment_summary}")


def aggregate_block(rows: list[dict]) -> None:
    if not rows:
        return

    print("\nAggregate")
    print("---------")

    accuracies = [r.get("baseline_accuracy") or 0 for r in rows]
    avg_acc = sum(accuracies) / len(accuracies)
    print(
        f"Accuracy:        avg {fmt_pct(avg_acc)}  "
        f"range {fmt_pct(min(accuracies))}-{fmt_pct(max(accuracies))}"
    )

    trust_vals = [r.get("trust_rating") for r in rows if r.get("trust_rating") is not None]
    if trust_vals:
        print(f"Trust:           avg {sum(trust_vals) / len(trust_vals):.1f}/5")

    conf_vals = [
        r.get("self_confidence_rating") for r in rows if r.get("self_confidence_rating") is not None
    ]
    if conf_vals:
        print(f"Self-confidence: avg {sum(conf_vals) / len(conf_vals):.1f}/5")

    will_counter = Counter(r.get("willingness_to_use") or "missing" for r in rows)
    parts = " ".join(f"{k}={v}" for k, v in sorted(will_counter.items()))
    print(f"Willingness:     {parts}")

    # Phase 2 most-useful-component aggregation
    component_counter: Counter[str] = Counter()
    understanding_vals: list[int] = []
    for r in rows:
        for ans in r.get("explanation_answers") or []:
            comp = ans.get("most_useful_component")
            if comp:
                component_counter[comp] += 1
            u = ans.get("understanding_rating")
            if u is not None:
                understanding_vals.append(u)

    if component_counter:
        print("\nMost useful evidence (Phase 2)")
        print("------------------------------")
        width = max(len(c) for c in component_counter)
        for comp, n in component_counter.most_common():
            print(f"  {comp:<{width}}  {n}")

    if understanding_vals:
        avg_u = sum(understanding_vals) / len(understanding_vals)
        print(f"\nUnderstanding rating: avg {avg_u:.1f}/5  ({len(understanding_vals)} answers)")


def dump_comments(rows: list[dict]) -> None:
    print("\nComments")
    print("--------")
    any_comment = False
    for r in rows:
        comment = (r.get("comments") or "").strip()
        if not comment:
            continue
        any_comment = True
        pid = r.get("participant_id") or "?"
        print(f"\n[{pid}]")
        print(comment)
    if not any_comment:
        print("(no participants left comments)")


def detail_one(rows: list[dict], participant_id: str) -> None:
    matches = [r for r in rows if r.get("participant_id") == participant_id]
    if not matches:
        print(f"No participant with id {participant_id!r}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(matches[0], indent=2, default=str, ensure_ascii=False))


def csv_export(rows: list[dict]) -> None:
    if not rows:
        return
    flattened = []
    for r in rows:
        copy = dict(r)
        copy["explanation_answers"] = json.dumps(copy.get("explanation_answers") or [])
        flattened.append(copy)
    fieldnames = list(flattened[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show XADE study results.")
    parser.add_argument(
        "--comments", action="store_true", help="dump free-text comments at the end"
    )
    parser.add_argument(
        "--detail", metavar="PARTICIPANT_ID", help="print one participant's full record"
    )
    parser.add_argument("--csv", action="store_true", help="export raw rows to stdout as CSV")
    args = parser.parse_args()

    rows = fetch_results()

    if args.csv:
        csv_export(rows)
        return

    if args.detail:
        detail_one(rows, args.detail)
        return

    print("XADE Study Results")
    print("==================")
    summary_table(rows)
    aggregate_block(rows)
    if args.comments:
        dump_comments(rows)


if __name__ == "__main__":
    main()
