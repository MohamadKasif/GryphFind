# MLH submission text (copy into the form)

**Project name:** GryphFind

**Tagline:** Snap a photo of what you found. Describe what you lost. Open-source AI matches them.

**Categories:** Best Open-Source AI Project, Best Use of Snowflake

## Inspiration
Lost stuff at Guelph ends up spread across the UC info desk, residence front desks, the library and random Instagram stories. Finding your water bottle means describing it over and over to different people. We wanted one board where a model does the comparing.

## What it does
Finders post a photo. Owners post a photo or just a description like "blue Hydro Flask with a Gryphons sticker". GryphFind ranks the most likely matches with a confidence score, auto-tags the item type, explains in one line why it thinks the match is right, and lets the owner send a claim request with a private proof. Only the finder sees the proofs and decides who gets it, so nobody can take an item by tapping a button; several people can claim the same item and the finder picks the real owner. A public Previous matches record shows every handoff. Posts live for 30 days; anything about to expire turns red, shows up in a red alert at the top, and can trigger browser notifications.

## How we built it
- **Open-weight AI at the core:** OpenAI's CLIP (ViT-B/32, MIT-licensed weights) via Hugging Face transformers embeds both photos and text into one vector space. That's what makes text-to-photo matching work. CLIP zero-shot classification also auto-tags categories.
- **Calibrated scoring:** photo-photo, text-text and text-photo similarities live on very different scales, so each gets its own logistic calibration before combining into a 0-100 score.
- **Snowflake:** posts and their 512-d CLIP vectors live in `VECTOR(FLOAT, 512)` columns. Matching runs in-warehouse with `VECTOR_COSINE_SIMILARITY`. `SNOWFLAKE.CORTEX.COMPLETE` with open-weight Llama 3.1 8B writes the match explanations and can summarise the board for campus staff. We also query lost-item hotspots by location.
- **App:** Flask backend, a single dependency-free HTML page that works on phones with the camera, SQLite fallback so it runs with zero setup.
- **Open source:** MIT-licensed, with tests, a contributing guide and starter issues for Hacktoberfest contributors.

## Challenges
CLIP's raw scores aren't comparable across modalities: a perfect text-to-photo match scores around 0.3 while two photos of the same object score around 0.9. We built per-signal calibration and a small CLI to tune it on real photos.

## What's next
Verification questions so only the real owner can claim an item, automatic blurring of faces and ID numbers, and notifications when a new strong match appears.

---

## 2-minute demo script
1. (Before demo) Post 4-5 "found" items with real photos: a water bottle, keys, AirPods case, a hoodie, an umbrella.
2. Click **I lost something**. Type only "blue water bottle with stickers" (no photo). Post. The bottle comes up top: words matched to a photo.
3. Click **I lost something** again and take a photo of a teammate's keys from a different angle. The keys match even though it's a different photo.
4. Point out the stamped score, the "why" line from Llama, and the category CLIP picked.
5. Use two browsers (or one normal + one private window) as two different people who both claim the bottle with different proofs. In the finder's browser open **My posts**: both requests and proofs are there. Tap **Hand over to them** on the right one; the other person's status flips to "Finder said not a match". Show **Previous matches** is read-only.
6. On any post tap **Demo: skip ahead**. The red "Leaving the board soon" alert appears and the tag turns red.
7. If on Snowflake: switch to the worksheet and run the hotspot query and the Cortex board summary from `snowflake/setup.sql`.
