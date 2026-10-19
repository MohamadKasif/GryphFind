"""GryphFind: match lost and found items on campus with open-weight CLIP."""
import os
from pathlib import Path

# Minimal .env loader (no extra dependency). Real env vars win.
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())
