"""
VecGrep GitHub Action entrypoint.

Reads inputs from INPUT_* environment variables (set by action.yml),
runs the requested vecgrep operation, writes outputs, and exits with
the appropriate code.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _set_output(name: str, value: str) -> None:
    """Write a GitHub Actions step output."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            # Use multiline delimiter to handle values with newlines
            delimiter = "EOF_VECGREP"
            f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
    else:
        print(f"::set-output name={name}::{value}")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fail(msg: str) -> None:
    print(f"::error::{msg}", flush=True)
    sys.exit(1)


def _resolve_path(raw: str) -> str:
    """Resolve path relative to GITHUB_WORKSPACE if not absolute."""
    workspace = os.environ.get("GITHUB_WORKSPACE", "")
    p = Path(raw)
    if not p.is_absolute() and workspace:
        p = Path(workspace) / p
    return str(p.resolve())


# ---------------------------------------------------------------------------
# GitHub API helpers (for PR comment mode)
# ---------------------------------------------------------------------------

def _get_pr_number() -> int | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not Path(event_path).exists():
        return None
    try:
        event = json.loads(Path(event_path).read_text())
        return event.get("pull_request", {}).get("number")
    except (json.JSONDecodeError, OSError):
        return None


def _post_pr_comment(token: str, repo: str, pr_number: int, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            _log(f"PR comment posted (HTTP {resp.status}).")
    except urllib.error.HTTPError as e:
        _log(f"Warning: failed to post PR comment — HTTP {e.code}: {e.reason}")


# ---------------------------------------------------------------------------
# VecGrep operations
# ---------------------------------------------------------------------------

def _do_index(path: str) -> str:
    from vecgrep.server import _do_index as do_index  # type: ignore
    return do_index(path, force=False)


def _do_search(path: str, query: str, top_k: int, min_score: float) -> list[dict]:
    """
    Run a semantic search and return results filtered by min_score.
    Each result: {rank, file, start_line, end_line, score, content}
    """
    from vecgrep.server import _do_index, _get_store  # type: ignore

    # Ensure index exists
    _do_index(path, force=False)

    from vecgrep.embedder import embed  # type: ignore
    import numpy as np

    query_vec = embed([query])[0]

    with _get_store(path) as store:
        raw = store.search(query_vec, top_k=top_k)

    results = []
    for i, row in enumerate(raw, start=1):
        score = float(row.get("score", 0.0))
        if score < min_score:
            continue
        results.append({
            "rank": i,
            "file": row.get("file_path", ""),
            "start_line": row.get("start_line", 0),
            "end_line": row.get("end_line", 0),
            "score": round(score, 4),
            "content": row.get("content", ""),
        })
    return results


def _do_duplicate_detection(path: str, min_score: float, top_k: int) -> list[dict]:
    """
    Find semantically similar code chunks within the codebase.
    For each chunk, search for its nearest neighbours (excluding itself).
    Returns pairs with score above min_score.
    """
    from vecgrep.server import _do_index, _get_store  # type: ignore

    _do_index(path, force=False)

    with _get_store(path) as store:
        all_rows = store.search_all()

    pairs = []
    seen: set[tuple[str, str]] = set()

    for row in all_rows:
        file_a = row.get("file_path", "")
        line_a = row.get("start_line", 0)
        vec = row.get("vector")
        if vec is None:
            continue

        import numpy as np
        vec = np.array(vec, dtype=np.float32)

        with _get_store(path) as store:
            neighbours = store.search(vec, top_k=top_k + 1)

        for neighbour in neighbours:
            file_b = neighbour.get("file_path", "")
            line_b = neighbour.get("start_line", 0)
            score = float(neighbour.get("score", 0.0))

            # Skip self-matches
            if file_a == file_b and line_a == line_b:
                continue
            if score < min_score:
                continue

            key = tuple(sorted([(file_a, line_a), (file_b, line_b)]))
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "file_a": file_a,
                "start_line_a": line_a,
                "file_b": file_b,
                "start_line_b": line_b,
                "score": round(score, 4),
            })

    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_results_markdown(results: list[dict], header: str) -> str:
    if not results:
        return f"### {header}\n\nNo results found above the score threshold."

    lines = [f"### {header}", ""]
    for r in results:
        lines.append(
            f"**[{r['rank']}]** `{r['file']}:{r['start_line']}-{r['end_line']}` "
            f"(score: {r['score']})"
        )
        lines.append("```")
        lines.append(r["content"].strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _format_duplicates_markdown(pairs: list[dict], header: str) -> str:
    if not pairs:
        return f"### {header}\n\nNo duplicate logic detected above the score threshold."

    lines = [f"### {header}", ""]
    for i, p in enumerate(pairs, start=1):
        lines.append(
            f"**[{i}]** `{p['file_a']}:{p['start_line_a']}` "
            f"vs `{p['file_b']}:{p['start_line_b']}` "
            f"(score: {p['score']})"
        )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mode           = _env("INPUT_MODE", "search")
    query          = _env("INPUT_QUERY")
    raw_path       = _env("INPUT_PATH", ".")
    top_k          = int(_env("INPUT_TOP_K", "8"))
    min_score      = float(_env("INPUT_MIN_SCORE", "0.7"))
    fail_on_match  = _env("INPUT_FAIL_ON_MATCH", "false").lower() == "true"
    fail_on_no_match = _env("INPUT_FAIL_ON_NO_MATCH", "false").lower() == "true"
    comment_header = _env("INPUT_COMMENT_HEADER", "VecGrep Semantic Search Results")
    github_token   = _env("INPUT_GITHUB_TOKEN")
    repo           = _env("GITHUB_REPOSITORY")

    path = _resolve_path(raw_path)
    _log(f"VecGrep action | mode={mode} | path={path}")

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    if mode == "index":
        stats = _do_index(path)
        _log(stats)
        _set_output("index_stats", stats)
        return

    # ------------------------------------------------------------------
    # search / validate / comment
    # ------------------------------------------------------------------
    if mode in ("search", "validate", "comment"):
        if not query:
            _fail("Input 'query' is required for mode: " + mode)

        results = _do_search(path, query, top_k, min_score)
        match_count = len(results)

        _set_output("results", json.dumps(results, indent=2))
        _set_output("match_count", str(match_count))

        if mode == "search":
            if results:
                for r in results:
                    _log(f"[{r['rank']}] {r['file']}:{r['start_line']}-{r['end_line']} (score: {r['score']})")
                    _log(r["content"].strip())
                    _log("")
            else:
                _log("No results found above the score threshold.")

        if mode == "comment":
            pr_number = _get_pr_number()
            if not github_token:
                _log("Warning: github_token not provided — skipping PR comment.")
            elif not pr_number:
                _log("Warning: could not determine PR number — skipping PR comment.")
            elif not repo:
                _log("Warning: GITHUB_REPOSITORY not set — skipping PR comment.")
            else:
                body = _format_results_markdown(results, comment_header)
                _post_pr_comment(github_token, repo, pr_number, body)

        if fail_on_match and match_count > 0:
            _fail(
                f"Found {match_count} match(es) for query '{query}' "
                f"(fail_on_match=true)."
            )
        if fail_on_no_match and match_count == 0:
            _fail(
                f"No matches found for query '{query}' "
                f"(fail_on_no_match=true)."
            )
        return

    # ------------------------------------------------------------------
    # duplicate
    # ------------------------------------------------------------------
    if mode == "duplicate":
        pairs = _do_duplicate_detection(path, min_score, top_k)
        match_count = len(pairs)

        _set_output("results", json.dumps(pairs, indent=2))
        _set_output("match_count", str(match_count))

        if pairs:
            _log(f"Found {match_count} potential duplicate(s):")
            for p in pairs:
                _log(
                    f"  {p['file_a']}:{p['start_line_a']} <-> "
                    f"{p['file_b']}:{p['start_line_b']}  (score: {p['score']})"
                )
        else:
            _log("No duplicate logic detected above the score threshold.")

        if fail_on_match and match_count > 0:
            _fail(f"Found {match_count} duplicate pair(s) (fail_on_match=true).")
        return

    _fail(f"Unknown mode: '{mode}'. Valid modes: index, search, validate, comment, duplicate.")


if __name__ == "__main__":
    main()
