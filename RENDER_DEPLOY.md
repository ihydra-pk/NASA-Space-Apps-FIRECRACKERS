# Deploying FireLens on Render

One Render web service serves the page (`/`) and the API (`/api/explore`) from
the same address, for prototyping and presentations. `render.yaml` describes it.

## What happens on each deploy

1. `pip install -r requirements.txt`.
2. The all-experiment search index (`data/vectorstore/firelens.sqlite3`, 441 MB,
   too big for GitHub) is built from the committed embeddings. This needs no
   network or key and takes under a minute.
3. `python -m src.api` starts the server on `0.0.0.0:$PORT`.
4. The first question loads the index into memory (about 10 s, about 130 MB).
   Later questions reuse it.

## Set it up (once)

1. In Render: **New → Blueprint**, pick this repository and the branch that
   holds `render.yaml` (`SI_Dashboard_Ravish`).
2. When asked, enter `UPB_AI_GATEWAY_KEY`. It is never stored in the repository.
3. After the deploy finishes, open `https://<service>.onrender.com/health`
   (should show `{"status":"ok"}`), then the site itself.

Every push to that branch redeploys automatically.

## Before a presentation

The `free` plan sleeps after 15 minutes without visits; the first visit then
waits about a minute for it to wake, plus the 10 s index load. Open the site
and ask one question a few minutes before presenting, or switch the plan to
`starter` in Render (always on, paid).

## Settings

Change these in Render's Environment tab; see `Run.md` for the full list.

- `UPB_AI_GATEWAY_KEY`: required; answers and query embeddings use the UPB gateway.
- `FIRELENS_API_TOP_K` (10), `FIRELENS_API_NEIGHBOURS` (0),
  `FIRELENS_API_DEMOTE_REFERENCES` (on), `FIRELENS_API_ROUTER` (on).
- `FIRELENS_LLM_MODEL`, `FIRELENS_LLM_MAX_OUTPUT_TOKENS`.

There is no rate limit or login: keep the URL within the team, because every
question uses the key's quota.

## If the chat only gives keyword answers

The page falls back to its built-in passages when the AI server fails, and
shows the reason. "Could not reach embedding API" means Render cannot reach the
university AI gateway. Open `https://<service>.onrender.com/api/diagnose` (it
uses no key and never shows it):

- `"tcp": "connected"` on the IPv4 address and `"https": {"status": 401}`:
  the gateway is reachable; check `key_configured` and the Logs tab.
- IPv4 times out or is refused: the gateway does not accept connections from
  Render's region. Create the service again in the Frankfurt region (add
  `region: frankfurt` to `render.yaml`; Render cannot move an existing
  service), then check again.
- Still blocked: the gateway only accepts university or German networks.
  Present from a laptop where it works (`python -m src.api`, then
  `http://127.0.0.1:8000`).

`FIRELENS_FORCE_IPV4=1` (set in `render.yaml`) makes the server skip the
gateway's IPv6 address, which Render cannot route to.
