"""Resilient web research for LexAI.

Searches multiple public web indexes, prefers authoritative legal sources,
fetches pages/PDFs when possible, and caches evidence locally in SQLite.
"""
from __future__ import annotations
import base64
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
SEARCH_TIMEOUT = 10
FETCH_TIMEOUT = 12
SEARCH_ENGINES = {"bing.com", "google.com", "google.co.in", "duckduckgo.com", "html.duckduckgo.com"}
STOPWORDS = {"what","is","the","for","with","this","that","does","are","was","were","how","can","could","should","would","under","about","from","into","and","or","of","to","in","on","a","an","i","me","my","tell","give","please","india","law","legal","section","act","punishment","penalty"}

DIRECT_LEGAL_SEEDS = [
 ("BNS — India Code (official PDF)", "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a2023-45.pdf"),
]
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

def _decode_bing(value):
    if not value: return ""
    value=unquote(value)
    if value.startswith("a1"):
        raw=value[2:]+("="*(-len(value[2:])%4))
        try:
            decoded=base64.urlsafe_b64decode(raw).decode("utf-8","ignore")
            if decoded.startswith(("http://","https://")): return decoded
        except Exception: pass
    return ""

def _unwrap(href):
    href=(href or "").strip()
    try:
        q=parse_qs(urlparse(href).query)
        if q.get("uddg"): return unquote(q["uddg"][0])
        if q.get("url") and q["url"][0].startswith("http"): return unquote(q["url"][0])
        if q.get("u"):
            decoded=_decode_bing(q["u"][0])
            if decoded: return decoded
    except Exception:
        pass
    return href

def _is_search_engine(url):
    d=_domain(url)
    return d in SEARCH_ENGINES or any(d.endswith("."+x) for x in SEARCH_ENGINES)

def _valid_destination(url):
    return bool(urlparse(url).scheme in {"http","https"} and _domain(url) and not _is_search_engine(url))

def _parse_results(html, selector):
    soup=BeautifulSoup(html,"html.parser")
    out=[]
    for node in soup.select(selector):
        a=node if node.name=="a" else node.select_one("a")
        if not a: continue
        href=_unwrap(a.get("href",""))
        title=a.get_text(" ",strip=True)
        if not href.startswith(("http://","https://")) or not title: continue
        parent=node.parent if node.name=="a" else node
        snippet=parent.get_text(" ",strip=True) if parent else title
        snippet=re.sub(r"\\s+"," ",snippet)
        if len(snippet)>1800: snippet=snippet[:1800]
        if any(x["url"]==href for x in out): continue
        out.append({"url":href,"title":title,"snippet":snippet})
    return out

def _ddg(q):
    try:
        u="https://html.duckduckgo.com/html/?q="+quote_plus(q)
        r=requests.get(u,headers={"User-Agent":USER_AGENT},timeout=SEARCH_TIMEOUT)
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
    c=_conn()
    rows=c.execute("SELECT url,domain,title,content,source_name,fetched_at FROM web_pages").fetchall()
    c.close()
    out=[]
    for url,d,title,content,source,fetched in rows:
        item={"url":url,"domain":d,"title":title,"content":content[:12000],"source_name":source,"fetched_at":fetched,"cached":True}
        score=_relevance(item,q)
        if score>=0.28:
            item["web_score"]=score; out.append(item)
    out.sort(key=lambda x:x["web_score"],reverse=True)
    return out[:limit]

def _query_terms(q):
    return {x for x in re.findall(r"[a-z0-9]{3,}",q.lower()) if x not in STOPWORDS}

def _query_variants(q):
    low=q.lower()
    variants=[q]
    if any(x in low for x in ("grievous hurt","stabbing","stabbed","knife","assault")):
        variants += [
            '"voluntarily causing grievous hurt" India BNS',
            '"grievous hurt" "dangerous weapon" India BNS',
            '"grievous hurt" knife India law section punishment',
            '"voluntarily causing grievous hurt" IPC India'
        ]
    if any(x in low for x in ("punishment","penalty","fine","imprisonment")):
        variants.append(q+" India law section punishment")
    return list(dict.fromkeys(variants))

def _relevance(item,q):
    terms=_query_terms(q)
    if not terms: return 0.0
    hay=(item.get("title","")+" "+item.get("snippet","")+" "+str(item.get("content",""))[:30000]).lower()
    matched=sum(1 for t in terms if t in hay)
    score=matched/max(1,len(terms))
    qlow=q.lower()
    for phrase in ("grievous hurt","dangerous weapon","voluntarily causing","section 326","section 117"):
        if phrase in qlow and phrase in hay: score+=0.15
    d=_domain(item.get("url",""))
    if d in PREFERRED_DOMAINS or any(d.endswith("."+p) for p in PREFERRED_DOMAINS): score+=0.10
    if d.endswith(".gov.in") or d.endswith(".nic.in"): score+=0.10
    return min(1.0,score)

def _rank(items,q,limit=None):
    unique={}
    for x in items:
        u=_unwrap(x.get("url",""))
        if not _valid_destination(u): continue
        x=dict(x); x["url"]=u; x["domain"]=_domain(u); x["source_name"]=_source_name(u)
        score=_relevance(x,q)
        if score<0.28 and not x.get("direct_seed"): continue
        x["web_score"]=(0.85 if x.get("direct_seed") else score+(0.03 if x.get("content") else 0))
        if u not in unique or x["web_score"]>unique[u].get("web_score",0): unique[u]=x
    rows=sorted(unique.values(),key=lambda x:x["web_score"],reverse=True)
    return rows[:limit] if limit else rows

def _direct_seed_items(q):
    low=q.lower()
    if ("grievous hurt" in low or "stabbing" in low or "stabbed" in low or "knife" in low) and "ipc" not in low:
        return [{"url":u,"title":t,"snippet":"Official India Code Bharatiya Nyaya Sanhita 2023 text; sections 117 and 118 cover voluntarily causing grievous hurt and dangerous weapons." ,"direct_seed":True}
                for t,u in DIRECT_LEGAL_SEEDS]
    return []

def search_legal_web(query,max_results=5):
    q=(query or "").strip()
    if not q: return [],{"enabled":True,"results":0,"error":"empty query"}
    t0=time.perf_counter()
    c=_conn()
    c.execute("INSERT INTO web_queries(query,created_at) VALUES(?,?)",(q,datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

    cached=_cached(q,max_results)
    items=list(cached)+_direct_seed_items(q)
    preferred=list(PREFERRED_DOMAINS.keys())
    searches=_query_variants(q)
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures=[]
        for s in searches:
            futures.extend([pool.submit(_ddg,s),pool.submit(_bing,s),pool.submit(_google,s)])
        for f in as_completed(futures):
            try: items.extend(f.result())
            except Exception: pass
    # Explicitly query high-value legal domains as a fallback.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(_search_domain,q,d) for d in preferred[:10]]
        for f in as_completed(futures):
            try: items.extend(f.result())
            except Exception: pass

    candidates=_rank(items,q,max_results*4)
    # Always attempt direct primary-law seeds before ordinary search results.
    direct=[x for x in candidates if x.get("direct_seed")]
    fresh=[]
    cached_urls={x["url"] for x in cached}
    for x in candidates:
        if x["url"] not in cached_urls and len(fresh)<max_results*3: fresh.append(x)

    def fetch_one(item):
        try:
            r=requests.get(item["url"],headers={"User-Agent":USER_AGENT},timeout=FETCH_TIMEOUT,allow_redirects=True)
            r.raise_for_status()
            text=_extract(item["url"],r)
            if len(text)>=100:
                return {**item,"content":text[:12000],"evidence_type":"page","fetched_at":datetime.now(timezone.utc).isoformat(),"cached":False}
        except Exception:
            pass
        snippet=item.get("snippet","").strip()
        if snippet:
            return {**item,"content":snippet,"evidence_type":"search_snippet","fetched_at":datetime.now(timezone.utc).isoformat(),"cached":False}
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

    results=_rank(items,q,max_results)
    for x in results:
        x.setdefault("evidence_type","cache")
    return results,{"enabled":True,"results":len(results),
      "cached_results":sum(1 for x in results if x.get("cached")),
      "domains":sorted({_domain(x["url"]) for x in results}),
      "latency_ms":round((time.perf_counter()-t0)*1000,2),
      "database":str(DB_PATH),
      "search_engines":["DuckDuckGo","Bing","Google"],
      "searched_domains":len(preferred),"query_variants":len(searches),"relevance_threshold":0.28}

def web_cache_status():
    try:
        c=_conn()
        n=c.execute("SELECT COUNT(*) FROM web_pages").fetchone()[0]
        q=c.execute("SELECT COUNT(*) FROM web_queries").fetchone()[0]
        c.close()
        return {"ready":True,"pages":n,"queries":q,"database":str(DB_PATH)}
    except Exception as e:
        return {"ready":False,"error":str(e),"pages":0,"queries":0}
