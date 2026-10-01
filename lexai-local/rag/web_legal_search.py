"""Resilient web research for LexAI.

Searches multiple public web indexes, prefers authoritative legal sources,
fetches pages/PDFs when possible, and caches evidence locally in SQLite.
"""
from __future__ import annotations
import hashlib
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
import requests
from bs4 import BeautifulSoup
from config import DATA_DIR

DB_PATH = DATA_DIR / "web_legal_cache.sqlite3"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LexAI/2.0"
MAX_PAGE_CHARS = 50000

PREFERRED_DOMAINS = {
    "indiacode.nic.in": "India Code — Government of India",
    "legislative.gov.in": "Legislative Department — Ministry of Law & Justice",
    "sci.gov.in": "Supreme Court of India",
    "scr.sci.gov.in": "Supreme Court Reports",
    "egazette.nic.in": "e-Gazette of India",
    "doj.gov.in": "Department of Justice — Government of India",
    "lawcommissionofindia.nic.in": "Law Commission of India",
    "supremecourtofindia.nic.in": "Supreme Court of India",
    "indiankanoon.org": "Indian Kanoon",
    "barandbench.com": "Bar & Bench",
    "livelaw.in": "LiveLaw",
    "wikipedia.org": "Wikipedia",
    "github.com": "GitHub",
}
TRUSTED_PREFIXES = tuple(PREFERRED_DOMAINS)

def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c=sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS web_pages(
      id INTEGER PRIMARY KEY,url TEXT UNIQUE,domain TEXT,title TEXT,content TEXT,
      source_name TEXT,fetched_at TEXT,content_hash TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS web_queries(
      id INTEGER PRIMARY KEY,query TEXT,created_at TEXT)""")
    c.commit()
    return c

def _domain(url):
    return (urlparse(url).hostname or "").lower().removeprefix("www.")

def _allowed(url):
    return bool(urlparse(url).scheme in {"http","https"} and _domain(url))

def _source_name(url):
    d=_domain(url)
    for k,v in PREFERRED_DOMAINS.items():
        if d==k or d.endswith("."+k): return v
    return d or "Web source"

def _unwrap(href):
    href=(href or "").strip()
    try:
        q=parse_qs(urlparse(href).query)
        if q.get("uddg"): return unquote(q["uddg"][0])
        if q.get("url") and q["url"][0].startswith("http"): return q["url"][0]
    except Exception:
        pass
    return href

def _parse_results(html, selector):
    soup=BeautifulSoup(html,"html.parser")
    out=[]
    for node in soup.select(selector):
        a=node if node.name=="a" else node.select_one("a")
        if not a: continue
        href=_unwrap(a.get("href",""))
        title=a.get_text(" ",strip=True)
        if not href.startswith(("http://","https://")) or not title: continue
        if any(x["url"]==href for x in out): continue
        out.append({"url":href,"title":title})
    return out

def _ddg(q):
    try:
        u="https://html.duckduckgo.com/html/?q="+quote_plus(q)
        r=requests.get(u,headers={"User-Agent":USER_AGENT},timeout=8)
        r.raise_for_status()
        return _parse_results(r.text,"a.result__a")
    except Exception: return []

def _bing(q):
    try:
        u="https://www.bing.com/search?q="+quote_plus(q)
        r=requests.get(u,headers={"User-Agent":USER_AGENT},timeout=8)
        r.raise_for_status()
        return _parse_results(r.text,"li.b_algo h2 a")
    except Exception: return []

def _google(q):
    try:
        u="https://www.google.com/search?q="+quote_plus(q)+"&num=10"
        r=requests.get(u,headers={"User-Agent":USER_AGENT},timeout=8)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,"html.parser")
        out=[]
        for a in soup.select("a"):
            href=_unwrap(a.get("href",""))
            title=a.get_text(" ",strip=True)
            if href.startswith("http") and title and "google." not in _domain(href):
                out.append({"url":href,"title":title})
        return out
    except Exception: return []

def _search_domain(q,domain):
    return _ddg(f"{q} site:{domain}") + _bing(f"{q} site:{domain}")

def _extract(url,response):
    ctype=(response.headers.get("content-type") or "").lower()
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader=PdfReader(io.BytesIO(response.content))
            text="\n".join((p.extract_text() or "") for p in reader.pages[:100])
        except Exception: return ""
    else:
        soup=BeautifulSoup(response.text,"html.parser")
        for x in soup(["script","style","nav","footer","header","noscript","svg","form"]): x.decompose()
        text=soup.get_text(" ",strip=True)
    return re.sub(r"\s+"," ",text or "").strip()[:MAX_PAGE_CHARS]

def _snippet_from_result(item):
    return item.get("snippet","")

def _save(item,text):
    c=_conn()
    now=datetime.now(timezone.utc).isoformat()
    digest=hashlib.sha256(text.encode("utf-8","ignore")).hexdigest()
    c.execute("""INSERT INTO web_pages(url,domain,title,content,source_name,fetched_at,content_hash)
      VALUES(?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET title=excluded.title,
      content=excluded.content,source_name=excluded.source_name,fetched_at=excluded.fetched_at,
      content_hash=excluded.content_hash""",
      (item["url"],_domain(item["url"]),item.get("title",""),text,_source_name(item["url"]),now,digest))
    c.commit(); c.close()

def _cached(q,limit):
    terms=set(re.findall(r"[a-z0-9]{3,}",q.lower()))
    c=_conn()
    rows=c.execute("SELECT url,domain,title,content,source_name,fetched_at FROM web_pages").fetchall()
    c.close()
    scored=[]
    for url,d,title,content,source,fetched in rows:
        hay=(title+" "+content[:30000]).lower()
        score=sum(1 for t in terms if t in hay)
        if score: scored.append((score,url,d,title,content,source,fetched))
    scored.sort(reverse=True)
    return [{"url":u,"domain":d,"title":t,"content":c[:12000],"source_name":s,
             "fetched_at":f,"cached":True,"score":score} for score,u,d,t,c,s,f in scored[:limit]]

def _rank(items,q):
    terms=set(re.findall(r"[a-z0-9]{3,}",q.lower()))
    unique={}
    for x in items:
        u=x["url"]
        if u in unique: continue
        d=_domain(u)
        score=0
        hay=(x.get("title","")+" "+x.get("snippet","")).lower()
        score += sum(2 for t in terms if t in hay)
        if any(d==p or d.endswith("."+p) for p in PREFERRED_DOMAINS): score+=8
        if d.endswith(".gov.in") or d.endswith(".nic.in"): score+=10
        if d.endswith(".ac.in"): score+=5
        unique[u]={**x,"domain":d,"source_name":_source_name(u),"web_score":score}
    return sorted(unique.values(),key=lambda x:x["web_score"],reverse=True)

def search_legal_web(query,max_results=5):
    q=(query or "").strip()
    if not q: return [],{"enabled":True,"results":0,"error":"empty query"}
    t0=time.perf_counter()
    c=_conn()
    c.execute("INSERT INTO web_queries(query,created_at) VALUES(?,?)",(q,datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

    cached=_cached(q,max_results)
    items=list(cached)
    preferred=list(PREFERRED_DOMAINS.keys())
    searches=[q, q+" India law legal section punishment"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(_ddg,s) for s in searches]+[pool.submit(_bing,s) for s in searches]
        for f in as_completed(futures):
            try: items.extend(f.result())
            except Exception: pass
    # Explicitly query high-value legal domains as a fallback.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(_search_domain,q,d) for d in preferred[:10]]
        for f in as_completed(futures):
            try: items.extend(f.result())
            except Exception: pass

    candidates=_rank(items,q)
    fresh=[]
    cached_urls={x["url"] for x in cached}
    for x in candidates:
        if x["url"] not in cached_urls and len(fresh)<max_results*3: fresh.append(x)

    def fetch_one(item):
        try:
            r=requests.get(item["url"],headers={"User-Agent":USER_AGENT},timeout=10,allow_redirects=True)
            r.raise_for_status()
            text=_extract(item["url"],r)
            if len(text)<100: return None
            return {**item,"content":text[:12000],"fetched_at":datetime.now(timezone.utc).isoformat(),"cached":False}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(fetch_one,x) for x in fresh]
        for f in as_completed(futures):
            try:
                x=f.result()
                if x:
                    _save(x,x["content"])
                    items.append(x)
            except Exception: pass

    results=_rank(items,q)[:max_results]
    return results,{"enabled":True,"results":len(results),
      "cached_results":sum(1 for x in results if x.get("cached")),
      "domains":sorted({_domain(x["url"]) for x in results}),
      "latency_ms":round((time.perf_counter()-t0)*1000,2),
      "database":str(DB_PATH),
      "search_engines":["DuckDuckGo","Bing","Google"],
      "searched_domains":len(preferred)}

def web_cache_status():
    try:
        c=_conn()
        n=c.execute("SELECT COUNT(*) FROM web_pages").fetchone()[0]
        q=c.execute("SELECT COUNT(*) FROM web_queries").fetchone()[0]
        c.close()
        return {"ready":True,"pages":n,"queries":q,"database":str(DB_PATH)}
    except Exception as e:
        return {"ready":False,"error":str(e),"pages":0,"queries":0}
