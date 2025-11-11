# apps/companion/router.py
import os, sqlite3, asyncio, json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Body, HTTPException, Depends, FastAPI

DB_PATH = os.getenv("LOUMINA_DB", "/data/loumina.db")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")  # si tu veux brancher un LLM

router = APIRouter()

# ---------- DB utils ----------
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            project_hint TEXT DEFAULT '',
            ttl_days INTEGER DEFAULT 30,
            created_at TEXT NOT NULL,
            vault INTEGER DEFAULT 0,
            consent TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS kintsugi(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ia_suggestion TEXT,
            human_choice TEXT,
            why TEXT,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS state(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        # état par défaut
        cur = conn.execute("SELECT value FROM state WHERE key='pause'")
        if cur.fetchone() is None:
            conn.execute("INSERT INTO state(key,value) VALUES('pause','false')")
    return conn

def now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- Mémoire éthique ----------
@router.post("/memory/ingest")
def memory_ingest(
    text: str = Body(...),
    ttl_days: int = Body(30),
    vault: bool = Body(False),
    consent: Optional[str] = Body(None),
    project_hint: str = Body("", embed=True),
    conn: sqlite3.Connection = Depends(get_db),
):
    with conn:
        conn.execute(
            "INSERT INTO memories(text, project_hint, ttl_days, created_at, vault, consent) VALUES(?,?,?,?,?,?)",
            (text, project_hint, ttl_days, now_utc_str(), int(vault), consent),
        )
    return {"stored": True}

@router.post("/memory/forget")
def memory_forget(
    project_hint: str = Body(...),
    conn: sqlite3.Connection = Depends(get_db),
):
    with conn:
        # on n’efface pas les coffres (vault=1)
        cur = conn.execute(
            "DELETE FROM memories WHERE vault=0 AND LOWER(project_hint) LIKE ?",
            (f"%{project_hint.lower()}%",),
        )
        deleted = cur.rowcount
    return {"deleted": deleted}

@router.post("/memory/purge")
def memory_purge(conn: sqlite3.Connection = Depends(get_db)):
    # purge TTL (created_at + ttl_days)
    with conn:
        rows = conn.execute("SELECT id, created_at, ttl_days, vault FROM memories").fetchall()
        to_delete = []
        for r in rows:
            if r["vault"] == 1:
                continue
            try:
                created = datetime.fromisoformat(r["created_at"])
            except Exception:
                # format inconnu => on garde
                continue
            if created + timedelta(days=int(r["ttl_days"])) < datetime.now(timezone.utc):
                to_delete.append(r["id"])
        for mid in to_delete:
            conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    return {"purged": len(to_delete)}

# ---------- Pause / Réflexion ----------
def get_pause(conn: sqlite3.Connection) -> bool:
    r = conn.execute("SELECT value FROM state WHERE key='pause'").fetchone()
    return (r and r["value"] == "true")

@router.post("/pause/toggle")
def toggle_pause(conn: sqlite3.Connection = Depends(get_db)):
    paused = get_pause(conn)
    new_val = "false" if paused else "true"
    with conn:
        conn.execute("REPLACE INTO state(key,value) VALUES('pause',?)", (new_val,))
    return {"pause": new_val == "true"}

def socratic_prompts(context: str) -> Dict[str, Any]:
    seeds = [
        "Quel est l'objectif derrière cet objectif ?",
        "Si tu n'avais que 30 min, que ferais-tu différemment ?",
        "Quelle imperfection ajouterait de la vie à cette pièce ?",
        "Que perdrais-tu si c'était 'parfait' ?",
        "Quelle contrainte peux-tu embrasser plutôt que contourner ?",
    ]
    return {"questions": seeds[:5], "echo": context[:400]}

@router.post("/reflect")
def reflect(context: str = Body(...), conn: sqlite3.Connection = Depends(get_db)):
    if get_pause(conn):
        return {"mode": "reflect", "payload": socratic_prompts(context)}
    # sinon: routage vers génération standard
    return {"mode": "answer", "hint": "route vers générateur principal"}

# ---------- Voix plurielle (3 chemins) ----------
VOICE_SYSTEM = (
    "Tu es un compagnon d'atelier. Propose 3 pistes distinctes: "
    "1) Mécanique (robuste, rapide), 2) Esthétique (geste, matériaux), 3) Frugale (minimal, réemploi). "
    "Réponds en JSON {'mecanique':..., 'esthetique':..., 'frugale':...} concis."
)

def _fallback_triplet(ctx: str) -> Dict[str, str]:
    # Fallback sans LLM (toujours utile en CPU-only)
    return {
        "mecanique": f"Solution robuste et rapide pour: {ctx[:80]}",
        "esthetique": f"Piste qui met en valeur le geste pour: {ctx[:80]}",
        "frugale": f"Option minimale en réemploi pour: {ctx[:80]}",
    }

@router.post("/suggest")
def suggest_triplet(context: str = Body(...)):
    # Option 1: fallback immédiat (fiable)
    triplet = _fallback_triplet(context)
    return {"choices": triplet}

    # Option 2 (décommenter si tu branches un LLM interne) :
    # import httpx
    # prompt = f"{VOICE_SYSTEM}\nContexte: {context}"
    # try:
    #     with httpx.Client(timeout=30) as client:
    #         r = client.post(f"{OLLAMA_BASE}/api/generate", json={
    #             "model": os.getenv("OLLAMA_MODEL","qwen2.5:7b"),
    #             "prompt": prompt,
    #             "stream": False
    #         })
    #         data = r.json()
    #         txt = data.get("response","").strip()
    #         return {"choices": json.loads(txt)}
    # except Exception:
    #     return {"choices": _fallback_triplet(context)}

# ---------- Journal Kintsugi ----------
@router.post("/kintsugi/record")
def kintsugi_record(
    ia_suggestion: str = Body(...),
    human_choice: str = Body(...),
    why: str = Body(""),
    conn: sqlite3.Connection = Depends(get_db),
):
    with conn:
        conn.execute(
            "INSERT INTO kintsugi(ia_suggestion,human_choice,why,created_at) VALUES(?,?,?,?)",
            (ia_suggestion, human_choice, why, now_utc_str()),
        )
    return {"stored": True}

@router.get("/kintsugi/insights")
def kintsugi_insights(conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute("SELECT * FROM kintsugi ORDER BY id DESC LIMIT 50").fetchall()
    return {"items": [dict(r) for r in rows]}

# ---------- Purge quotidienne (boucle) ----------
async def _daily_purge_loop():
    # première attente jusqu’à ~03:05 UTC
    while True:
        now = datetime.now(timezone.utc)
        target = (now + timedelta(days=1)).replace(hour=3, minute=5, second=0, microsecond=0)
        await asyncio.sleep((target - now).total_seconds())
        try:
            conn = get_db()
            memory_purge(conn)  # appelle la route interne
            conn.close()
        except Exception:
            pass

def mount(app: FastAPI):
    # à appeler depuis main pour lancer la boucle
    @app.on_event("startup")
    async def _start():
        asyncio.create_task(_daily_purge_loop())
