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
# GitHub API helpers
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


def _github_get(token: str, url: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get_pr_changed_files(token: str, repo: str, pr_number: int) -> list[dict]:
    """Return the list of files changed in a PR with their patches."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=100"
    return _github_get(token, url)  # type: ignore


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


def _extract_added_lines(patch: str) -> str:
    """Extract only the added lines from a unified diff patch."""
    lines = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)


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
    from vecgrep.embedder import embed  # type: ignore

    _do_index(path, force=False)

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


def _do_pr_analyze(
    path: str,
    token: str,
    repo: str,
    pr_number: int,
    top_k: int,
    min_score: float,
) -> dict[str, list[dict]]:
    """
    Analyze a PR by:
    1. Fetching the list of changed files and their diffs from the GitHub API.
    2. Extracting added/modified lines from each file's patch.
    3. Using that content as a semantic query to find related code in the codebase.
    4. Returning a map of {changed_filename: [related_results]} for files that
       have meaningful related code elsewhere in the codebase.

    Files with trivial changes (<30 chars of new content) are skipped.
    Results that point back to the changed file itself are excluded.
    """
    _log("Fetching PR changed files...")
    try:
        changed_files = _get_pr_changed_files(token, repo, pr_number)
    except urllib.error.HTTPError as e:
        _fail(f"Failed to fetch PR files — HTTP {e.code}: {e.reason}")

    # Only analyse files that have a patch (excludes binary files, renames with no edits)
    files_with_patch = [
        f for f in changed_files
        if f.get("patch") and f.get("status") in ("added", "modified", "renamed")
    ]

    if not files_with_patch:
        _log("No changed files with content to analyse.")
        return {}

    _log(f"Indexing codebase at {path}...")
    _do_index(path)

    findings: dict[str, list[dict]] = {}

    # Limit to 10 files per run to keep CI times reasonable
    for file_info in files_with_patch[:10]:
        filename = file_info["filename"]
        patch = file_info.get("patch", "")
        added_content = _extract_added_lines(patch)

        if len(added_content.strip()) < 30:
            _log(f"Skipping {filename} — trivial change.")
            continue

        # Truncate to 600 chars to keep embedding meaningful
        query_content = added_content[:600]
        _log(f"Searching for code related to changes in {filename}...")

        results = _do_search(path, query_content, top_k=top_k, min_score=min_score)

        # Filter out results from the changed file itself
        workspace = os.environ.get("GITHUB_WORKSPACE", "")
        related = [
            r for r in results
            if not r["file"].endswith(filename)
        ]

        if related:
            findings[filename] = related

    return findings


def _do_duplicate_detection(path: str, min_score: float, top_k: int) -> list[dict]:
    """
    Find semantically similar code chunks within the codebase.
    Returns pairs with score above min_score.
    """
    from vecgrep.server import _do_index, _get_store  # type: ignore

    _do_index(path, force=False)

    with _get_store(path) as store:
        all_rows = store.search_all()

    pairs = []
    seen: set[tuple] = set()

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


def _format_pr_analysis_markdown(findings: dict[str, list[dict]]) -> str:
    if not findings:
        return (
            "## VecGrep PR Analysis\n\n"
            "No semantically related code found in the codebase for the changes in this PR."
        )

    lines = [
        "## VecGrep PR Analysis",
        "",
        "The following files changed in this PR have semantically related code elsewhere "
        "in the codebase. Review these to avoid duplication or ensure consistency.",
        "",
    ]

    for changed_file, results in findings.items():
        lines.append(f"---")
        lines.append(f"### Changes in `{changed_file}`")
        lines.append("")
        lines.append(f"Related code found ({len(results)} match(es)):")
        lines.append("")
        for r in results:
            lines.append(
                f"**[{r['rank']}]** `{r['file']}:{r['start_line']}-{r['end_line']}` "
                f"(score: {r['score']})"
            )
            lines.append("<details><summary>View snippet</summary>")
            lines.append("")
            lines.append("```python")
            lines.append(r["content"].strip())
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    lines.append("---")
    lines.append(
        "_Generated by [VecGrep Action](https://github.com/VecGrep/action)_"
    )
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
    mode             = _env("INPUT_MODE", "search")
    query            = _env("INPUT_QUERY")
    raw_path         = _env("INPUT_PATH", ".")
    top_k            = int(_env("INPUT_TOP_K", "8"))
    min_score        = float(_env("INPUT_MIN_SCORE", "0.7"))
    fail_on_match    = _env("INPUT_FAIL_ON_MATCH", "false").lower() == "true"
    fail_on_no_match = _env("INPUT_FAIL_ON_NO_MATCH", "false").lower() == "true"
    comment_header   = _env("INPUT_COMMENT_HEADER", "VecGrep Semantic Search Results")
    github_token     = _env("INPUT_GITHUB_TOKEN")
    repo             = _env("GITHUB_REPOSITORY")

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
    # analyze — PR diff analysis with automatic comment
    # ------------------------------------------------------------------
    if mode == "analyze":
        if not github_token:
            _fail("Input 'github_token' is required for mode: analyze")

        pr_number = _get_pr_number()
        if not pr_number:
            _fail("Could not determine PR number. Ensure this runs on a pull_request event.")
        if not repo:
            _fail("GITHUB_REPOSITORY is not set.")

        findings = _do_pr_analyze(path, github_token, repo, pr_number, top_k, min_score)
        total_matches = sum(len(v) for v in findings.items())

        _set_output("results", json.dumps(findings, indent=2))
        _set_output("match_count", str(len(findings)))

        body = _format_pr_analysis_markdown(findings)
        _log(body)
        _post_pr_comment(github_token, repo, pr_number, body)
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
            _fail(f"Found {match_count} match(es) for query '{query}' (fail_on_match=true).")
        if fail_on_no_match and match_count == 0:
            _fail(f"No matches found for query '{query}' (fail_on_no_match=true).")
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

    _fail(
        f"Unknown mode: '{mode}'. "
        "Valid modes: index, search, validate, comment, duplicate, analyze."
    )


if __name__ == "__main__":
    main()
