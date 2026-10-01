"""Authoritative Indian legal web search + SQLite evidence cache."""
from __future__ import annotations
import hashlib, re, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urlparse
import requests
from bs4 import BeautifulSoup
from config import DATA_DIR

DB_PATH = DATA_DIR / "web_legal_cache.sqlite3"
USER_AGENT = "LexAI-Legal-Research/1.0 (+local legal research assistant)"
OFFICIAL_SOURCES = {
    "indiacode.nic.in": "India Code — Government of India",
    "legislative.gov.in": "Legislative Department — Ministry of Law & Justice",
    "sci.gov.in": "Supreme Court of India",
    "scr.sci.gov.in": "Supreme Court Reports",
    "egazette.nic.in": "e-Gazette of India",
    "doj.gov.in": "Department of Justice — Government of India",
    "lawcommissionofindia.nic.in": "Law Commission of India",
}
MAX_PAGE_CHARS = 50000
CACHE_DAYS = 14

def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS web_pages(
        id INTEGER PRIMARY KEY, url TEXT UNIQUE, domain TEXT, title TEXT,
        content TEXT, source_name TEXT, fetched_at TEXT, content_hash TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS web_queries(
        id INTEGER PRIMARY KEY, query TEXT, created_at TEXT)""")
    c.commit()
    return c

def _allowed(url):
    try:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in OFFICIAL_SOURCES)
    except Exception:
        return False

def _domain(url):
    return (urlparse(url).hostname or "").lower()

def _search_engine(query, domain, limit=4):
    q = f"{query} site:{domain}"
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(q)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out=[]
        for a in soup.select("a.result__a"):
            href=a.get("href","").strip()
            title=a.get_text(" ",strip=True)
            if href and title and _allowed(href):
                out.append({"url":href,"title":title,"domain":domain})
                if len(out)>=limit: break
        return out
    except Exception:
        return []

def _extract(url, response):
    ctype=(response.headers.get("content-type") or "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader=PdfReader(io.BytesIO(response.content))
            text="\n".join((p.extract_text() or "") for p in reader.pages[:80])
        except Exception:
            return ""
    else:
        soup=BeautifulSoup(response.text,"html.parser")
        for x in soup(["script","style","nav","footer","header","noscript","svg"]): x.decompose()
        text=soup.get_text(" ",strip=True)
    text=re.sub(r"\s+"," ",text or "").strip()
    return text[:MAX_PAGE_CHARS]

def _save(url,title,text,domain):
    c=_conn()
    now=datetime.now(timezone.utc).isoformat()
    digest=hashlib.sha256(text.encode("utf-8","ignore")).hexdigest()
    c.execute("""INSERT INTO web_pages(url,domain,title,content,source_name,fetched_at,content_hash)
                 VALUES(?,?,?,?,?,?,?)
                 ON CONFLICT(url) DO UPDATE SET title=excluded.title,content=excluded.content,
                 source_name=excluded.source_name,fetched_at=excluded.fetched_at,
                 content_hash=excluded.content_hash""",
              (url,domain,title,text,OFFICIAL_SOURCES.get(domain,domain),now,digest))
    c.commit(); c.close()

def _cached(query, limit):
    terms=set(re.findall(r"[a-z0-9]{3,}",query.lower()))
    if not terms: return []
    c=_conn()
    rows=c.execute("SELECT url,domain,title,content,source_name,fetched_at FROM web_pages ORDER BY fetched_at DESC").fetchall()
    c.close()
    scored=[]
    for url,domain,title,content,source_name,fetched_at in rows:
        hay=(title+" "+content[:20000]).lower()
        score=sum(1 for t in terms if t in hay)
        if score: scored.append((score,url,domain,title,content,source_name,fetched_at))
    scored.sort(reverse=True)
    return [{"url":u,"domain":d,"title":t,"content":c[:12000],"source_name":s,
             "fetched_at":f,"cached":True,"score":score}
            for score,u,d,t,c,s,f in scored[:limit]]

def search_legal_web(query, max_results=5):
    q=(query or "").strip()
    if not q: return [], {"enabled":True,"results":0,"error":"empty query"}
    t0=time.perf_counter()
    c=_conn()
    c.execute("INSERT INTO web_queries(query,created_at) VALUES(?,?)",
              (q,datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

    # Reuse cached pages first, but still refresh the official web index for current law.
    cached=_cached(q,max_results)
    seen={x["url"] for x in cached}
    candidates=[]
    domains=list(OFFICIAL_SOURCES)
    for domain in domains:
        candidates.extend(_search_engine(q,domain,limit=2))
        if len(candidates)>=max_results*3: break

    results=list(cached)
    session=requests.Session()
    for item in candidates:
        if item["url"] in seen: continue
        try:
            r=session.get(item["url"],headers={"User-Agent":USER_AGENT},timeout=12)
            r.raise_for_status()
            text=_extract(item["url"],r)
            if len(text)<120: continue
            _save(item["url"],item["title"],text,item["domain"])
            results.append({**item,"content":text[:12000],"source_name":OFFICIAL_SOURCES[item["domain"]],
                            "fetched_at":datetime.now(timezone.utc).isoformat(),"cached":False})
            seen.add(item["url"])
        except Exception:
            continue
        if len(results)>=max_results: break

    # De-duplicate and cap.
    unique=[]
    seen=set()
    for x in results:
        if x["url"] in seen: continue
        seen.add(x["url"]); unique.append(x)
    unique=unique[:max_results]
    return unique, {"enabled":True,"results":len(unique),
                    "cached_results":sum(1 for x in unique if x.get("cached")),
                    "domains":sorted({x["domain"] for x in unique}),
                    "latency_ms":round((time.perf_counter()-t0)*1000,2),
                    "database":str(DB_PATH)}

def web_cache_status():
    try:
        c=_conn()
        n=c.execute("SELECT COUNT(*) FROM web_pages").fetchone()[0]
        q=c.execute("SELECT COUNT(*) FROM web_queries").fetchone()[0]
        c.close()
        return {"ready":True,"pages":n,"queries":q,"database":str(DB_PATH)}
    except Exception as e:
        return {"ready":False,"error":str(e),"pages":0,"queries":0}
