#!/usr/bin/env python3
"""Create matching local keys once, without overwriting existing configuration."""
import os
import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
backend = root / 'backend/.env'
frontend = root / 'frontend/.env'
compose = root / '.env'


def read_key(path, name):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(name + '='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None


keys = [read_key(backend, 'API_KEY'), read_key(frontend, 'VITE_API_KEY'), read_key(compose, 'VITE_API_KEY')]
existing = [k for k in keys if k and k != 'replace-with-a-random-local-key']
if len(set(existing)) > 1:
    raise SystemExit('Existing API keys disagree. Align backend/.env, frontend/.env and .env manually; no files changed.')
if any(p.exists() and (not k or k == 'replace-with-a-random-local-key') for p, k in zip((backend, frontend, compose), keys)):
    raise SystemExit('An existing .env has an empty/placeholder key. Set matching API_KEY / VITE_API_KEY values; no files changed.')
key = existing[0] if existing else secrets.token_urlsafe(32)
for path, content in (
    (backend, (root / 'backend/.env.example').read_text().replace('replace-with-a-random-local-key', key)),
    (frontend, (root / 'frontend/.env.example').read_text().replace('replace-with-a-random-local-key', key)),
    (compose, f'BACKEND_PORT=8000\nFRONTEND_PORT=5173\nVITE_API_KEY={key}\nVITE_BACKEND_URL=http://localhost:8000\nVITE_MAX_UPLOAD_MB=10\n'),
):
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(content)
        print(f'Created {path.relative_to(root)}')
    else:
        print(f'Keeping {path.relative_to(root)}')
print('Configuration ready. Run: docker compose up --build')
