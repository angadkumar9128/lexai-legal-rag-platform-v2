"""Fast smoke test for LexAI retrieval without starting Streamlit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag.query_analyzer import analyze
from rag.retriever import retrieve
queries=[
 "what is penalties for stabbing?",
 "what is the punishment for murder?",
 "helmet nahi pehna to kitna fine hai?",
 "what should I do if I cut many trees without permission?",
 "what is section 103 of BNS?",
]
for q in queries:
 print("\n"+"="*80); print("QUERY:",q)
 p=analyze(q,[])
 print("PLAN:",{k:p.get(k) for k in ["standalone_query","expanded_query","acts","sections","domain","intent","planner_source"]})
 rows,m=retrieve(q,p,6)
 print("RETRIEVAL:",m)
 for i,r in enumerate(rows[:3],1):
  print(f"[{i}] {r.get('act_name')} | {r.get('section_number')} | score={r.get('score',0):.3f}")
  print(str(r.get('chunk_text',''))[:300].replace("\n"," "))
