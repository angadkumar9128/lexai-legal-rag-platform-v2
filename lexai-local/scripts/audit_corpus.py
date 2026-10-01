"""Audit the local legal corpus for expected Indian-law coverage."""
from pathlib import Path
import pickle
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/"vector_store"/"metadata.pkl"
if not p.exists(): raise SystemExit("metadata.pkl missing; build vector store first")
rows=pickle.loads(p.read_bytes())
print("Corpus rows:",len(rows))
acts={}
for r in rows:
 a=str(r.get("act_name","")).strip()
 acts[a]=acts.get(a,0)+1
print("\nActs / source groups:")
for a,n in sorted(acts.items(),key=lambda x:(-x[1],x[0].lower())): print(f"{n:5d}  {a}")
terms=["Bharatiya Nyaya Sanhita","BNS","Indian Penal Code","IPC","Bharatiya Nagarik Suraksha Sanhita","BNSS","Motor Vehicles Act","Environment","Forest","Wild Life"]
print("\nCoverage checks:")
for term in terms:
 hits=sum(1 for r in rows if term.lower() in (str(r.get("act_name",""))+" "+str(r.get("chunk_text",""))).lower())
 print(f"{term:40s} {hits}")
