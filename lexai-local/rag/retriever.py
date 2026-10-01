"""Production hybrid legal retriever: dense + BM25-style lexical + metadata + optional reranking."""
from __future__ import annotations
import math,pickle,re,time
from collections import Counter
from pathlib import Path
import faiss,numpy as np
from sentence_transformers import SentenceTransformer
from config import VECTOR_DIR,EMBED_MODEL,RERANK_MODEL,USE_RERANKER,RERANK_TOP_N,RETRIEVAL_TOP_K
try: from sentence_transformers import CrossEncoder
except Exception: CrossEncoder=None
INDEX_PATH=VECTOR_DIR/"faiss_index.bin"; META_PATH=VECTOR_DIR/"metadata.pkl"; TOKEN_RE=re.compile(r"[a-z0-9]{2,}")
class Retriever:
 def __init__(self):
  if not INDEX_PATH.exists() or not META_PATH.exists(): raise FileNotFoundError("Vector store missing. Run vector_store/build_vector_db.py --if-needed")
  self.index=faiss.read_index(str(INDEX_PATH))
  with META_PATH.open("rb") as f: self.rows=pickle.load(f)
  self.embedder=SentenceTransformer(EMBED_MODEL)
  embed_dim=int(self.embedder.get_sentence_embedding_dimension())
  index_dim=int(self.index.d)
  if index_dim != embed_dim:
   raise ValueError(f"Vector store dimension mismatch: FAISS index has {index_dim} dimensions, but {EMBED_MODEL} produces {embed_dim}. Rebuild the vector store with: python vector_store/build_vector_db.py")
  if int(self.index.ntotal) != len(self.rows):
   raise ValueError(f"Vector store mismatch: FAISS index has {self.index.ntotal} vectors but metadata has {len(self.rows)} rows. Rebuild the vector store.")
  self.embedding_dim=embed_dim
  self.reranker=None
  if USE_RERANKER and CrossEncoder is not None:
   try:self.reranker=CrossEncoder(RERANK_MODEL)
   except Exception:self.reranker=None
  self.doc_tokens=[self._tokens(self._text(r)) for r in self.rows]; self.df=Counter(t for toks in self.doc_tokens for t in set(toks))
  self.n=len(self.rows); self.avgdl=sum(map(len,self.doc_tokens))/max(1,self.n)
 @staticmethod
 def _tokens(s): return TOKEN_RE.findall((s or "").lower())
 @staticmethod
 def _text(r): return " ".join([str(r.get("act_name","")),str(r.get("section_number","")),str(r.get("chunk_text",""))])
 def _bm25(self,q,i):
  qt=set(self._tokens(q)); toks=self.doc_tokens[i]; tf=Counter(toks); dl=len(toks); score=0.0
  for t in qt:
   f=tf.get(t,0)
   if not f: continue
   df=self.df.get(t,0); idf=math.log(1+(self.n-df+0.5)/(df+0.5))
   score+=idf*((f*2.2)/(f+1.2*(0.65+0.35*dl/max(1,self.avgdl))))
  return score
 def _meta_boost(self,r,p):
  text=self._text(r).lower(); acts=" ".join(p.get("acts") or []).lower(); domain=str(p.get("domain","")).lower(); boost=0.0
  if acts and any(a.strip() in text for a in re.split(r"[,;]",acts) if a.strip()): boost+=0.35
  if domain=="criminal_law" and any(x in text for x in ["penal code","nyaya sanhita","criminal","evidence","crpc","bnss"]): boost+=0.30
  if domain=="traffic_law" and any(x in text for x in ["motor vehicle","traffic","driving","road"]): boost+=0.30
  if domain=="environmental_law" and any(x in text for x in ["environment","forest","wildlife","biodiversity","tree"]): boost+=0.30
  if p.get("intent")=="penalty" and any(x in text for x in ["penalty","fine","punishable","imprisonment","liable","sentence"]): boost+=0.20
  sections={str(x).upper() for x in p.get("sections") or []}; rowsec={str(r.get("section_number","")).upper()}|{str(x).upper() for x in r.get("section_tokens",[]) or []}
  if sections & rowsec: boost+=0.80
  return boost
 def search(self,query,plan,top_k=RETRIEVAL_TOP_K):
  t0=time.perf_counter(); expanded=str(plan.get("expanded_query") or query)
  qv=self.embedder.encode([expanded],convert_to_numpy=True,normalize_embeddings=False); qv=np.asarray(qv,dtype=np.float32); faiss.normalize_L2(qv)
  dense_scores,dense_ids=self.index.search(qv,min(self.n,max(60,top_k*10)))
  dense_rank={int(i):rank for rank,i in enumerate(dense_ids[0].tolist(),1) if i>=0}
  dense_score={int(i):float(s) for i,s in zip(dense_ids[0].tolist(),dense_scores[0].tolist()) if i>=0}
  lex=[] 
  for i in range(self.n):
   bm=self._bm25(expanded,i)
   if bm>0: lex.append((bm,i))
  lex.sort(reverse=True); lexical_rank={i:rank for rank,(bm,i) in enumerate(lex[:max(150,top_k*15)],1)}
  ids=set(dense_rank)|set(lexical_rank); scored=[]
  for i in ids:
   r=dict(self.rows[i]); dr=dense_rank.get(i,9999); lr=lexical_rank.get(i,9999); rrf=1/(60+dr)+1/(60+lr)
   bm=self._bm25(expanded,i); score=rrf*100+min(2.0,bm)*0.22+self._meta_boost(r,plan)
   r.update({"_doc_id":i,"dense_score":dense_score.get(i,0.0),"bm25_score":bm,"rrf_score":rrf,"score":score}); scored.append(r)
  scored.sort(key=lambda x:x["score"],reverse=True); candidates=scored[:max(RERANK_TOP_N,top_k)]
  rerank_used=False; rerank_error=""
  if self.reranker and candidates:
   try:
    ce=self.reranker.predict([(query,str(r.get("chunk_text",""))[:1400]) for r in candidates])
    for r,s in zip(candidates,ce): r["rerank_score"]=float(s); r["score"]+=float(s)*0.35
    candidates.sort(key=lambda x:x["score"],reverse=True); rerank_used=True
   except Exception as e: rerank_error=str(e)
  selected=candidates[:top_k]
  if selected:
   top=selected[0]; lex=min(1.0,top["bm25_score"]/8); dense=max(0.0,min(1.0,(top["dense_score"]+1)/2)); rank_signal=min(1.0,top["score"]/110)
   confidence=0.45*rank_signal+0.30*lex+0.25*dense
  else: confidence=0.0
  return selected,{"confidence":round(confidence,4),"candidate_count":len(scored),"latency_ms":round((time.perf_counter()-t0)*1000,2),"rerank_used":rerank_used,"reranker_model":RERANK_MODEL if self.reranker else "disabled","reranker_error":rerank_error,"reason":"strong" if confidence>=.55 else "medium" if confidence>=.34 else "weak"}
_RETRIEVER=None
def get_retriever():
 global _RETRIEVER
 if _RETRIEVER is None:_RETRIEVER=Retriever()
 return _RETRIEVER
def retrieve(query,plan,top_k=RETRIEVAL_TOP_K): return get_retriever().search(query,plan,top_k)
def retrieve_chunks(question,top_k=5,**kwargs):
 plan=kwargs.get("analysis") or {"expanded_query":question,"domain":"general","intent":"general","acts":[],"sections":[]}
 return retrieve(question,plan,top_k)
def retriever_status():
 try:r=get_retriever(); return {"ready":True,"corpus_size":r.n,"embed_model":EMBED_MODEL,"embedding_dim":r.embedding_dim,"index_dim":int(r.index.d)}
 except Exception as e:return {"ready":False,"error":str(e),"corpus_size":0,"embed_model":EMBED_MODEL}
