# Contributing to GryphFind

Pull requests are welcome, including Hacktoberfest ones.

1. Fork, then create a branch: `git checkout -b my-change`
2. Install: `pip install -r requirements.txt pytest`
3. Make your change. Keep the frontend dependency-free (one `index.html`).
4. Run the tests: `EMBEDDER=stub python -m pytest tests`
5. Open a PR describing what changed and how you tested it. Screenshots help for UI changes.

## Layout

- `gryphfind/embedder.py`: CLIP wrapper and the test stub
- `gryphfind/scoring.py`: turns raw similarities into a 0-100 score
- `gryphfind/store.py`: SQLite and Snowflake backends (same interface)
- `gryphfind/explain.py`: LLM match explanations with a no-LLM fallback
- `gryphfind/app.py`: Flask routes
- `gryphfind/static/index.html`: the whole UI

## Ground rules

Be kind. Follow the [MLH Code of Conduct](https://www.mlh.com/code-of-conduct). Don't commit photos of real people or real ID cards.
