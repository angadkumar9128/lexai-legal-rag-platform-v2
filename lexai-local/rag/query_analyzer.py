"""Deterministic + Qwen structured query planner."""
from __future__ import annotations
import re
from rag.llm_service import json_chat

ACT_ALIASES={
 "Bharatiya Nyaya Sanhita":["bns","bharatiya nyaya sanhita","nyaya sanhita"],
 "Indian Penal Code":["ipc","indian penal code","penal code"],
 "Bharatiya Nagarik Suraksha Sanhita":["bnss","bharatiya nagarik suraksha sanhita"],
 "Code of Criminal Procedure":["crpc","code of criminal procedure"],
 "Motor Vehicles Act":["motor vehicles act","mv act","traffic","helmet","challan"],
 "Environment (Protection) Act":["environment protection act","environment","pollution"],
 "Indian Forest Act":["indian forest act","forest act","forest","tree","trees","felling"],
 "Wild Life (Protection) Act":["wildlife","wild life protection act","protected species"],
 "Forest (Conservation) Act":["forest conservation","forest clearance"],
 "Constitution of India":["constitution","fundamental right","article","writ"],
 "Indian Contract Act":["contract act","contract","agreement"],
}
EXPANSIONS={
 "stabbing":"stab stabbed stabbing knife chaku assault injury hurt grievous hurt",
 "stabbed":"stab stabbed stabbing knife chaku assault injury hurt",
 "penalties":"penalty punishment fine punishable imprisonment sentence",
 "punishments":"punishment penalty fine punishable imprisonment sentence",
 "trees":"tree trees cutting felling forest permission clearance environment",
 "tree":"tree trees cutting felling forest permission clearance environment",
 "jurmana":"penalty fine punishment","saza":"punishment penalty sentence imprisonment fine",
 "maar":"assault hurt injury stabbing murder attempt",
}
def _clean(q): return re.sub(r"\s+"," ",(q or "").strip())
def _tokens(q): return re.findall(r"[a-z0-9]{2,}",q.lower())
def _detect(q):
 low=q.lower(); act=""
 for name,aliases in ACT_ALIASES.items():
  if any(re.search(rf"\b{re.escape(a)}\b",low) for a in aliases): act=name; break
 criminal=any(x in low for x in ["stab","knife","murder","assault","crime","offence","offense","punishment","penalty","fine","imprisonment","bns","ipc"])
 env=any(x in low for x in ["tree","forest","wildlife","environment","pollution","biodiversity"])
 traffic=any(x in low for x in ["helmet","traffic","challan","motor vehicle","driving"])
 domain="environmental_law" if env else "traffic_law" if traffic else "criminal_law" if criminal else "general"
 if any(x in low for x in ["penalty","penalties","fine","fines","punishment","punishments","punishable","imprisonment","sentence"]): intent="penalty"
 elif any(x in low for x in ["what should i do","what can i do","how do i","next step","permission","clearance"]): intent="remedy"
 elif any(x in low for x in ["section","sec.","ipc","bns","act","law"]): intent="provision"
 elif any(x in low for x in ["what is","define","meaning"]): intent="definition"
 else: intent="general"
 return {"act":act,"domain":domain,"intent":intent}
def expand_query(q):
 base=_clean(q); extra=[]
 for tok in _tokens(base):
  if tok in EXPANSIONS: extra.append(EXPANSIONS[tok])
 return _clean(base+" "+" ".join(extra))
SCHEMA={"type":"object","properties":{
 "standalone_query":{"type":"string"},"acts":{"type":"array","items":{"type":"string"}},
 "sections":{"type":"array","items":{"type":"string"}},"domain":{"type":"string"},"intent":{"type":"string"},
 "jurisdiction":{"type":"string"},"language":{"type":"string"},"time_sensitive":{"type":"boolean"},
 "needs_clarification":{"type":"boolean"}},"required":["standalone_query","acts","sections","domain","intent","jurisdiction","language","time_sensitive","needs_clarification"],"additionalProperties":False}
def analyze(question,history=None):
 q=_clean(question); d=_detect(q); history=history or []
 messages=[
  {"role":"system","content":"You are LexAI's Indian legal research planner. Do not answer the question. Produce retrieval metadata only. Never invent section numbers. Preserve uncertainty. Handle English and Hindi/Hinglish."},
  {"role":"user","content":f"Question: {q}\nRecent conversation: {history[-4:]}\nHints: {d}\nExpanded terms: {expand_query(q)}"}]
 obj,meta=json_chat(messages,SCHEMA,max_tokens=350)
 if obj:
  obj.update({"original_query":q,"expanded_query":expand_query(str(obj.get("standalone_query") or q)),"planner_source":"qwen"})
  return obj
 return {"original_query":q,"standalone_query":q,"expanded_query":expand_query(q),
  "acts":[d["act"]] if d["act"] else [],
  "sections":re.findall(r"\b(?:section|sec\.?)\s*([0-9]{1,4}[A-Za-z]?)",q,re.I),
  "domain":d["domain"],"intent":d["intent"],"jurisdiction":"India",
  "language":"hi" if re.search(r"[\u0900-\u097F]",q) else "en","time_sensitive":False,
  "needs_clarification":False,"planner_source":"deterministic"}
def analyze_query(question,history=None): return analyze(question,history)
def fallback_analyze_query(question): return analyze(question,[])
def query_analyzer_status(): return {"ready":True,"mode":"qwen_structured_or_deterministic"}
