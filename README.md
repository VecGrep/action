# VecGrep Action

A GitHub Action for semantic code search, PR validation, and architectural enforcement using [VecGrep](https://github.com/VecGrep/VecGrep).

## Modes

| Mode | Description |
|---|---|
| `index` | Index the codebase and report stats |
| `search` | Run a semantic query and print results |
| `validate` | Search for a pattern — fail if matches found (`fail_on_match`) or not found (`fail_on_no_match`) |
| `comment` | Post search results as a PR comment |
| `duplicate` | Detect semantically similar code chunks across the codebase |

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `mode` | No | `search` | Operation mode (see table above) |
| `query` | Depends | — | Semantic search query (required for `search`, `validate`, `comment`, `duplicate`) |
| `path` | No | `.` | Path to the codebase to index/search |
| `top_k` | No | `8` | Number of results to return |
| `min_score` | No | `0.7` | Minimum similarity score threshold (0.0–1.0) |
| `fail_on_match` | No | `false` | Exit 1 if any results are found above `min_score` |
| `fail_on_no_match` | No | `false` | Exit 1 if no results are found above `min_score` |
| `comment_header` | No | `VecGrep Semantic Search Results` | PR comment heading (comment mode) |
| `github_token` | No | — | GitHub token for posting PR comments |

## Outputs

| Output | Description |
|---|---|
| `results` | Search results as a JSON array |
| `match_count` | Number of results above the score threshold |
| `index_stats` | Indexing summary (index mode) |

## Examples

### Index on every PR

```yaml
- uses: VecGrep/action@main
  with:
    mode: index
    path: .
```

### Enforce architectural rules

Fail if raw SQL queries are found; require authentication logic to exist:

```yaml
- uses: VecGrep/action@main
  with:
    mode: validate
    query: "raw SQL query string concatenation"
    min_score: "0.80"
    fail_on_match: "true"

- uses: VecGrep/action@main
  with:
    mode: validate
    query: "user authentication token verification"
    min_score: "0.75"
    fail_on_no_match: "true"
```

### Comment on PRs with related code

```yaml
- uses: VecGrep/action@main
  with:
    mode: comment
    query: "user authentication login session"
    top_k: "5"
    comment_header: "Related authentication code"
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

### Detect duplicate logic

```yaml
- uses: VecGrep/action@main
  with:
    mode: duplicate
    min_score: "0.92"
    fail_on_match: "true"
```

Full example workflows are in the [`examples/`](examples/) directory.

## Docker

The action runs in a Docker container and can also be used standalone:

```bash
docker build -t vecgrep-action .

# Search
docker run --rm \
  -v $(pwd):/workspace \
  -e INPUT_MODE=search \
  -e INPUT_QUERY="database connection setup" \
  -e INPUT_PATH=/workspace \
  vecgrep-action

# Validate — exit 1 if pattern found
docker run --rm \
  -v $(pwd):/workspace \
  -e INPUT_MODE=validate \
  -e INPUT_QUERY="hardcoded password secret api key" \
  -e INPUT_PATH=/workspace \
  -e INPUT_MIN_SCORE=0.82 \
  -e INPUT_FAIL_ON_MATCH=true \
  vecgrep-action
```

## License

MIT
