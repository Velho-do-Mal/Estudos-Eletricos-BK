# bk_estudos_eletricos/db.py
# ============================================================
# Camada de persistência — SQLite local + Neon PostgreSQL
# ============================================================
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

_DB_FILENAME = "bk_estudos_eletricos.db"

# ── Neon PostgreSQL ──────────────────────────────────────────
NEON_CONN_STR = (
    "postgresql://neondb_owner:npg_J6anjETyt0AN"
    "@ep-young-fog-aibjr3vn-pooler.c-4.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require"
)

def _get_pg_engine():
    try:
        from sqlalchemy import create_engine
        return create_engine(NEON_CONN_STR, future=True, connect_args={"connect_timeout": 8})
    except Exception:
        return None

def init_neon() -> bool:
    eng = _get_pg_engine()
    if eng is None:
        return False
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ee_projects (
                    id SERIAL PRIMARY KEY, name TEXT NOT NULL, client TEXT,
                    project_number TEXT, voltage_kv REAL, power_mva REAL,
                    frequency_hz REAL, n_circuits INTEGER, n_cables_per_phase INTEGER,
                    geometry_type TEXT, n_lines INTEGER, meta_json TEXT,
                    created_at TEXT, updated_at TEXT);"""))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ee_studies (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES ee_projects(id) ON DELETE CASCADE,
                    module_key TEXT NOT NULL, module_label TEXT,
                    input_json TEXT, result_json TEXT,
                    created_at TEXT, updated_at TEXT);"""))
            conn.commit()
        return True
    except Exception as e:
        print(f"Neon init: {e}")
        return False

def upsert_project_neon(project: Dict[str, Any], project_id: Optional[int] = None) -> Optional[int]:
    eng = _get_pg_engine()
    if eng is None:
        return None
    try:
        from sqlalchemy import text
        now = _utcnow_iso()
        meta_json = None
        mo = project.get("meta") or project.get("meta_json")
        if mo:
            try: meta_json = json.dumps(mo, ensure_ascii=False)
            except Exception: pass
        pid = project_id or project.get("id")
        params = {"name": project.get("name"), "client": project.get("client"),
                  "pn": project.get("project_number"), "vkv": project.get("voltage_kv"),
                  "pmva": project.get("power_mva"), "fhz": project.get("frequency_hz"),
                  "nc": project.get("n_circuits"), "ncpp": project.get("n_cables_per_phase"),
                  "gt": project.get("geometry_type"), "nl": project.get("n_lines"),
                  "mj": meta_json, "ua": now}
        with eng.connect() as conn:
            if pid:
                params["id"] = int(pid)
                conn.execute(text("""UPDATE ee_projects SET name=:name,client=:client,
                    project_number=:pn,voltage_kv=:vkv,power_mva=:pmva,frequency_hz=:fhz,
                    n_circuits=:nc,n_cables_per_phase=:ncpp,geometry_type=:gt,n_lines=:nl,
                    meta_json=:mj,updated_at=:ua WHERE id=:id"""), params)
            else:
                params["ca"] = now
                row = conn.execute(text("""INSERT INTO ee_projects
                    (name,client,project_number,voltage_kv,power_mva,frequency_hz,
                    n_circuits,n_cables_per_phase,geometry_type,n_lines,meta_json,created_at,updated_at)
                    VALUES (:name,:client,:pn,:vkv,:pmva,:fhz,:nc,:ncpp,:gt,:nl,:mj,:ca,:ua)
                    RETURNING id"""), params).fetchone()
                pid = row[0] if row else None
            conn.commit()
        return int(pid) if pid else None
    except Exception as e:
        print(f"Neon upsert: {e}")
        return None

def list_projects_neon() -> List[Dict[str, Any]]:
    eng = _get_pg_engine()
    if eng is None:
        return []
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            rows = conn.execute(text(
                "SELECT id,name,client,project_number,voltage_kv,updated_at,created_at "
                "FROM ee_projects ORDER BY COALESCE(updated_at,created_at) DESC")).fetchall()
        return [{"id":r[0],"name":r[1],"client":r[2],"project_number":r[3],
                 "voltage_kv":r[4],"updated_at":r[5],"created_at":r[6]} for r in rows]
    except Exception:
        return []

def get_project_neon(pid: int) -> Optional[Dict[str, Any]]:
    eng = _get_pg_engine()
    if eng is None:
        return None
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            row = conn.execute(text("SELECT * FROM ee_projects WHERE id=:id"),{"id":int(pid)}).fetchone()
        if not row:
            return None
        keys = ["id","name","client","project_number","voltage_kv","power_mva","frequency_hz",
                "n_circuits","n_cables_per_phase","geometry_type","n_lines","meta_json","created_at","updated_at"]
        d = dict(zip(keys, row))
        mj = d.pop("meta_json", None)
        d["meta"] = json.loads(mj) if mj else None
        return d
    except Exception:
        return None

def delete_project_neon(pid: int) -> bool:
    eng = _get_pg_engine()
    if eng is None:
        return False
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            conn.execute(text("DELETE FROM ee_projects WHERE id=:id"),{"id":int(pid)})
            conn.commit()
        return True
    except Exception:
        return False

# ── Estudos salvos ───────────────────────────────────────────

def save_study_neon(project_id: int, module_key: str, module_label: str,
                    input_data: Dict, result_data: Dict) -> Optional[int]:
    eng = _get_pg_engine()
    if eng is None:
        return None
    try:
        from sqlalchemy import text
        now = _utcnow_iso()
        ij = json.dumps(input_data, ensure_ascii=False, default=str)
        rj = json.dumps(result_data, ensure_ascii=False, default=str)
        with eng.connect() as conn:
            existing = conn.execute(text(
                "SELECT id FROM ee_studies WHERE project_id=:pid AND module_key=:mk "
                "ORDER BY id DESC LIMIT 1"), {"pid":project_id,"mk":module_key}).fetchone()
            if existing:
                conn.execute(text(
                    "UPDATE ee_studies SET input_json=:ij,result_json=:rj,module_label=:ml,"
                    "updated_at=:ua WHERE id=:id"),
                    {"ij":ij,"rj":rj,"ml":module_label,"ua":now,"id":existing[0]})
                sid = existing[0]
            else:
                r = conn.execute(text(
                    "INSERT INTO ee_studies (project_id,module_key,module_label,input_json,"
                    "result_json,created_at,updated_at) VALUES (:pid,:mk,:ml,:ij,:rj,:ca,:ua) "
                    "RETURNING id"),
                    {"pid":project_id,"mk":module_key,"ml":module_label,
                     "ij":ij,"rj":rj,"ca":now,"ua":now}).fetchone()
                sid = r[0] if r else None
            conn.commit()
        return sid
    except Exception as e:
        print(f"Neon save_study: {e}")
        return None

def list_studies_neon(project_id: int) -> List[Dict[str, Any]]:
    eng = _get_pg_engine()
    if eng is None:
        return []
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            rows = conn.execute(text(
                "SELECT id,module_key,module_label,updated_at FROM ee_studies "
                "WHERE project_id=:pid ORDER BY updated_at DESC"), {"pid":project_id}).fetchall()
        return [{"id":r[0],"module_key":r[1],"module_label":r[2],"updated_at":r[3]} for r in rows]
    except Exception:
        return []

def load_study_neon(study_id: int) -> Optional[Dict[str, Any]]:
    eng = _get_pg_engine()
    if eng is None:
        return None
    try:
        from sqlalchemy import text
        with eng.connect() as conn:
            row = conn.execute(text(
                "SELECT id,project_id,module_key,module_label,input_json,result_json,updated_at "
                "FROM ee_studies WHERE id=:id"), {"id":study_id}).fetchone()
        if not row:
            return None
        d = {"id":row[0],"project_id":row[1],"module_key":row[2],"module_label":row[3],"updated_at":row[6]}
        try: d["input_data"] = json.loads(row[4]) if row[4] else {}
        except Exception: d["input_data"] = {}
        try: d["result_data"] = json.loads(row[5]) if row[5] else {}
        except Exception: d["result_data"] = {}
        return d
    except Exception:
        return None

# ── SQLite local ─────────────────────────────────────────────

def _db_path() -> Path:
    import os
    env = os.getenv("BK_EE_DB_PATH")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "launcher.py").exists():
        return cwd / _DB_FILENAME
    return Path(__file__).resolve().parent / _DB_FILENAME

def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def init_db() -> None:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, client TEXT,
            project_number TEXT, voltage_kv REAL, power_mva REAL, frequency_hz REAL,
            n_circuits INTEGER, n_cables_per_phase INTEGER, geometry_type TEXT, n_lines INTEGER,
            created_at TEXT, updated_at TEXT)""")
        cur.execute("PRAGMA table_info(projects)")
        cols = {row[1] for row in cur.fetchall()}
        if "meta_json" not in cols:
            cur.execute("ALTER TABLE projects ADD COLUMN meta_json TEXT")
        con.commit()
    finally:
        con.close()

def list_projects() -> List[Dict[str, Any]]:
    init_db()
    con = sqlite3.connect(str(_db_path()))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT id,name,client,project_number,updated_at,created_at "
                    "FROM projects ORDER BY COALESCE(updated_at,created_at) DESC, id DESC")
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    con = sqlite3.connect(str(_db_path()))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM projects WHERE id = ?", (int(project_id),))
        r = cur.fetchone()
        if not r: return None
        d = dict(r)
        meta = d.get("meta_json")
        d["meta"] = json.loads(meta) if meta else None
        return d
    finally:
        con.close()

def upsert_project(project: Dict[str, Any], project_id: Optional[int] = None) -> int:
    init_db()
    pid = project_id or project.get("id")
    mo = project.get("meta") or project.get("meta_json")
    meta_json = json.dumps(mo, ensure_ascii=False) if mo else None
    now = _utcnow_iso()
    con = sqlite3.connect(str(_db_path()))
    try:
        cur = con.cursor()
        if pid:
            cur.execute("""UPDATE projects SET name=?,client=?,project_number=?,voltage_kv=?,
                power_mva=?,frequency_hz=?,n_circuits=?,n_cables_per_phase=?,geometry_type=?,
                n_lines=?,updated_at=?,meta_json=? WHERE id=?""",
                (project.get("name"),project.get("client"),project.get("project_number"),
                 project.get("voltage_kv"),project.get("power_mva"),project.get("frequency_hz"),
                 project.get("n_circuits"),project.get("n_cables_per_phase"),
                 project.get("geometry_type"),project.get("n_lines"),now,meta_json,int(pid)))
            if cur.rowcount == 0: pid = None
        if not pid:
            cur.execute("""INSERT INTO projects (name,client,project_number,voltage_kv,power_mva,
                frequency_hz,n_circuits,n_cables_per_phase,geometry_type,n_lines,created_at,
                updated_at,meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (project.get("name"),project.get("client"),project.get("project_number"),
                 project.get("voltage_kv"),project.get("power_mva"),project.get("frequency_hz"),
                 project.get("n_circuits"),project.get("n_cables_per_phase"),
                 project.get("geometry_type"),project.get("n_lines"),now,now,meta_json))
            pid = int(cur.lastrowid)
        con.commit()
        # Backup Neon
        try: upsert_project_neon(dict(project, id=pid), pid)
        except Exception: pass
        return int(pid)
    finally:
        con.close()

def delete_project(project_id: int) -> None:
    init_db()
    con = sqlite3.connect(str(_db_path()))
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM projects WHERE id = ?", (int(project_id),))
        con.commit()
    finally:
        con.close()
    try: delete_project_neon(project_id)
    except Exception: pass

# ── Compat ───────────────────────────────────────────────────
def get_all_projects(): return list_projects()
def fetch_projects(): return list_projects()
def projects(): return list_projects()
def save_project(p): return upsert_project(p)
def insert_project(p): return upsert_project(p)
def create_project(p): return upsert_project(p)
def add_project(p): return upsert_project(p)
