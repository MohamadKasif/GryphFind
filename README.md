# GryphFind

Campus lost & found that matches items for you. Post a photo of something you found, or a photo **or just a description** of something you lost, and GryphFind ranks the most likely matches from the other side of the board.

Built in one day at **Hacktoberfest Hack Day Guelph x GDG on Campus Guelph** (Oct 19, 2026).

## Why

Lost items at a university end up scattered across front desks, residence offices and Instagram stories. Searching means walking around and describing your water bottle to five different people. GryphFind puts everything on one board and lets a model do the comparing.

## How it works

1. **Embed.** Every post is run through [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (open weights, MIT licence). Photos get an image embedding; titles and descriptions get a text embedding. CLIP puts both in the same 512-dimensional space.
2. **Match.** A new "lost" post is compared with every open "found" post (and vice versa) on three signals: photo vs photo, words vs words, and words vs photo. That last one is what lets "blue Hydro Flask with a Gryphons sticker" find a photo someone else took.
3. **Calibrate.** Raw CLIP scores sit on different scales per signal (photo-to-text similarities top out around 0.3), so each one goes through its own logistic calibration before they're combined into a 0-100 match score ([`scoring.py`](gryphfind/scoring.py)).
4. **Auto-tag.** CLIP zero-shot classification tags each photo as a category (phone, keys, water bottle…).
5. **Explain.** The top matches get a one-line "why this might be yours" from an open-weight LLM: Llama 3.1 8B through Snowflake Cortex, or any local model through Ollama. With no LLM configured it falls back to a rule-based explanation.
6. **Reunite.** "This is mine" marks both posts resolved and takes them off the board.

```
photo / text ──► CLIP (open weights) ──► vectors ──► SQLite + numpy
                                                 └─► Snowflake VECTOR + VECTOR_COSINE_SIMILARITY
                                                          │
                       top matches ──► Llama 3.1 (Cortex / Ollama) ──► "why it matches"
```

## Run it

Needs Python 3.10+. The first run downloads CLIP (~600 MB).

```bash
git clone <this repo> && cd gryphfind
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m gryphfind
```

Open http://localhost:8000. Phones on the same Wi-Fi can use `http://<your-laptop-ip>:8000` and post straight from the camera.

### Snowflake backend (optional)

1. Run [`snowflake/setup.sql`](snowflake/setup.sql) in a worksheet.
2. `cp .env.example .env` and fill in the `SNOWFLAKE_*` values.
3. Restart. The header will say "on Snowflake".

With Snowflake on, posts and their CLIP vectors are stored in `VECTOR(FLOAT, 512)` columns, all similarity search runs in the warehouse with `VECTOR_COSINE_SIMILARITY`, and match explanations come from `SNOWFLAKE.CORTEX.COMPLETE` with an open-weight Llama model. `setup.sql` also has hotspot analytics and a Cortex summary of the whole board for campus staff.

### Local LLM explanations (optional)

```bash
ollama pull llama3.2:3b
OLLAMA_HOST=http://localhost:11434 python -m gryphfind
```

### Tune matching

Take two photos of the same item and one of a different item, then:

```bash
python -m gryphfind.calibrate same1.jpg same2.jpg other.jpg "blue water bottle"
```

Set the midpoints in `gryphfind/scoring.py` between the "same" and "different" numbers.

## Tests

```bash
EMBEDDER=stub python -m pytest tests
```

`EMBEDDER=stub` swaps CLIP for a tiny deterministic embedder so tests run in a second without model weights.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/posts` | Multipart: `kind` (lost/found), `title`, `description`, `location`, `contact`, `category`, `image`. Returns the post and its matches. |
| `GET` | `/api/posts?kind=found` | Open posts on the board |
| `GET` | `/api/posts/<id>/matches` | Re-run matching for a post |
| `POST` | `/api/posts/<id>/resolve` | JSON `{"match_id": "..."}` marks both reunited |
| `GET` | `/api/stats` | Counts, top categories, top locations |

## Ideas for contributors

Issues tagged `hacktoberfest` and `good first issue` are a good place to start. Some open ideas:

- Verification question: finders hide one detail, claimants must answer it
- Blur faces and ID-card numbers in uploaded photos before saving
- Email the lost-item poster when a new strong match is found
- Store images in a Snowflake stage instead of local disk
- Bulk import from a residence front desk's spreadsheet

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. CLIP weights are © OpenAI under the MIT licence. Llama models are under the Llama community licence.
