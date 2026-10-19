# GryphFind

Campus lost & found that matches items for you. Post a photo of something you found, or a photo **or just a description** of something you lost, and GryphFind ranks the most likely matches from the other side of the board.

Built in one day at **Hacktoberfest Hack Day Guelph x GDG on Campus Guelph** (Oct 19, 2026).

## Why

Lost items at a university end up scattered across front desks, residence offices and Instagram stories. Searching means walking around and describing your water bottle to five different people. GryphFind puts everything on one board and lets a model do the comparing.

## How it works

1. **Embed.** Every post is run through [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (open weights, MIT licence). Photos get an image embedding; titles and descriptions get a text embedding. CLIP puts both in the same 512-dimensional space.
2. **Match.** A new "lost" post is compared with every open "found" post (and vice versa) on three signals: photo vs photo (CLIP), description vs photo (CLIP), and what the two descriptions *mean* (a second open-weight model, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0, ~90 MB). CLIP is what lets "blue Hydro Flask with a Gryphons sticker" find someone else's photo; MiniLM is what knows "blue capsule" and "blue container" are close.
3. **Calibrate and filter.** Raw CLIP scores sit on different scales per signal (photo-to-text similarities top out around 0.3), so each one goes through its own logistic calibration before they're combined into a 0-100 match score ([`scoring.py`](gryphfind/scoring.py)). Anything under `MIN_MATCH` (default 40) is hidden, so an Apple Pencil never shows up as a match for a lunch container.
4. **Auto-tag.** CLIP zero-shot classification tags each photo as a category (phone, keys, water bottle…).
5. **Explain.** The top matches get a one-line "why this might be yours" from an open-weight LLM: Llama 3.1 8B through Snowflake Cortex, or any local model through Ollama. With no LLM configured it falls back to a rule-based explanation.
6. **The finder decides.** Every post gets a secret key that stays in the browser that made it (the server only stores a hash). The person who lost something taps **This is mine** and answers a private proof question; that sends a *claim request* to the finder. Several people can request the same item. Only the finder sees the proofs, in **My posts**, and only the finder can tap **Hand over to them** (every other request is declined) or **Not theirs**. Everyone else just sees statuses, and **Previous matches** is a read-only public record.
   - **Contacts stay private.** Nobody browsing sees a phone number or pickup spot. The finder sees each claimant's contact; the owner sees the pickup details only after the finder hands it over.
   - **Secret detail.** Finders can note something private about the item ("initials under the lid"); it's shown only to them, next to each claimant's proof.
   - The demo fast-forward button only works on your own posts.
7. **30-day board.** Posts expire after 30 days (`EXPIRY_DAYS`). Anything with 3 days or less turns red, and a red "Leaving the board soon" alert lists those items at the top of the page. Visitors can turn on browser notifications to get alerted while the site is open.

```
photo / text ──► CLIP (open weights) ──► vectors ──► SQLite + numpy
                                                 └─► Snowflake VECTOR + VECTOR_COSINE_SIMILARITY
                                                          │
                       top matches ──► Llama 3.1 (Cortex / Ollama) ──► "why it matches"
```

## Run it

Needs Python 3.10+. The first run downloads CLIP (~600 MB) and MiniLM (~90 MB).

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

### Settings

- `EXPIRY_DAYS` (default 30): how long posts stay on the board.
- `MIN_MATCH` (default 40): lowest score shown as a match.
- `GRYPHFIND_DEMO` (default 1): shows a "Demo: skip ahead" button on each post so you can show the expiry alert to judges. Set to `0` for real use.

### Tune matching

Take two photos of the same item and one of a different item, then:

```bash
python -m gryphfind.calibrate same1.jpg same2.jpg other.jpg "blue water bottle"
```

Set the midpoints in `gryphfind/scoring.py` between the "same" and "different" numbers.

## Artwork

To use official artwork you have permission for, drop `logo.png` and/or `mascot.png` into [`gryphfind/static/brand/`](gryphfind/static/brand/) and refresh. Otherwise the built-in drawing is used.

The gryphon is original art (not the official Gryphons logo). It's generated by [`tools/make_gryphon.py`](tools/make_gryphon.py), which triangulates each body part and shades every facet from a single light source.

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
| `GET` | `/api/claims` | Public claim history (no proofs) and open request counts |
| `POST` | `/api/claims` | `{"lost_id", "found_id", "key", "proof"}`: the lost post's owner asks the finder |
| `POST` | `/api/claims/<id>/hand-over` | `{"key"}` finder only: hand it over, decline the rest |
| `POST` | `/api/claims/<id>/decline` | `{"key"}` finder only: not a match, never suggested again |
| `POST` | `/api/mine` | `{"keys": {post_id: key}}`: your posts and their requests (finders also get proofs) |
| `GET` | `/api/expiring?days=3` | Posts that will be deleted within N days |
| `POST` | `/api/demo/age/<id>` | Demo only (`GRYPHFIND_DEMO=1`): pretend a post is older |
| `GET` | `/api/stats` | Counts, top categories, top locations |

## Ideas for contributors

Issues tagged `hacktoberfest` and `good first issue` are a good place to start. Some open ideas:

- Verification question: finders hide one detail, claimants must answer it
- Blur faces and ID-card numbers in uploaded photos before saving
- Email or web-push (service worker) alerts so expiry warnings reach people even when the site is closed
- Accounts (e.g. uoguelph.ca sign-in) so posts can be managed from any device, not just the one that made them
- Store images in a Snowflake stage instead of local disk
- Bulk import from a residence front desk's spreadsheet

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. CLIP weights are © OpenAI under the MIT licence. Llama models are under the Llama community licence.
