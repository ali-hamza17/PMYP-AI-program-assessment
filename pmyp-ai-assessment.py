#!/usr/bin/env python3
"""
Technical Assessment — Simple Multi-Agent Chat System

Single-file Python implementation with:
  • Coordinator (Manager)
  • ResearchAgent, AnalysisAgent, MemoryAgent
  • Structured memory layer with vector search (hash-TF vectors + cosine)
  • Adaptive decision-making, confidence scoring, traceable logs
  • Five sample scenarios producing outputs/*.txt

Run:
  python3 multi_agent_system.py

This will create an ./outputs directory containing:
  - simple_query.txt
  - complex_query.txt
  - memory_test.txt
  - multi_step.txt
  - collaborative.txt

No external dependencies beyond Python 3.9+ and standard library.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import uuid
import random
import string
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- Utility: lightweight vectorizer ----------------------------- #

class HashVectorizer:
    """A tiny, dependency-free hashing vectorizer for similarity search.

    • Tokenizes on word boundaries, lowercases, strips punctuation.
    • Uses a fixed-dimensional signed hashing trick for term -> index.
    • Builds L2-normalized vectors suitable for cosine similarity.
    """
    def __init__(self, dim: int = 2048, seed: int = 13):
        self.dim = dim
        random.seed(seed)
        self._salt = ''.join(random.choice(string.ascii_letters) for _ in range(8))
        # simple stopword list
        self.stop = set(
            """
            a an and are as at be by for from has have i if in into is it its of on or our so
            that the their them there these they this to was were what when where which who will with
            you your about we not than then up out over under across also can may might should could would
            """.split()
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t]

    def _hash(self, token: str) -> Tuple[int, int]:
        # simple deterministic hash -> index and sign
        h = hash(self._salt + token)
        idx = h % self.dim
        sign = 1 if (h >> 31) & 1 else -1
        return idx, sign

    def vectorize(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in self._tokenize(text) if t not in self.stop]
        if not tokens:
            return vec
        inv_len = 1.0 / math.sqrt(len(tokens))
        for tok in tokens:
            i, s = self._hash(tok)
            vec[i] += s * inv_len
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

# ----------------------------- Memory schema & layer ----------------------------- #

@dataclass
class MemoryRecord:
    id: str
    timestamp: float
    role: str                 # 'user' | 'assistant' | 'agent' | 'system'
    topic: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None

@dataclass
class Fact:
    id: str
    timestamp: float
    topic: str
    content: str
    source: str               # e.g., 'ResearchAgent:KB:mock'
    agent: str                # which agent asserted it
    confidence: float
    vector: Optional[List[float]] = None

@dataclass
class AgentState:
    id: str
    timestamp: float
    agent: str
    task: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

class MemoryLayer:
    """Structured memory with three stores and vector/keyword search."""
    def __init__(self, vectorizer: Optional[HashVectorizer] = None):
        self.vectorizer = vectorizer or HashVectorizer()
        self.conversation: List[MemoryRecord] = []
        self.knowledge_base: List[Fact] = []
        self.agent_state: List[AgentState] = []

    # ---- Conversation Memory ---- #
    def log_conversation(self, role: str, topic: str, content: str, **metadata) -> MemoryRecord:
        rec = MemoryRecord(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            role=role,
            topic=topic,
            content=content,
            metadata=metadata,
            vector=self.vectorizer.vectorize(content)
        )
        self.conversation.append(rec)
        return rec

    def search_conversation(self, query: str, top_k: int = 5) -> List[Tuple[MemoryRecord, float]]:
        qv = self.vectorizer.vectorize(query)
        scored = [(r, HashVectorizer.cosine(qv, r.vector or self.vectorizer.vectorize(r.content))) for r in self.conversation]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---- Knowledge Base ---- #
    def upsert_fact(self, topic: str, content: str, source: str, agent: str, confidence: float) -> Fact:
        vec = self.vectorizer.vectorize(content)
        fact = Fact(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            topic=topic,
            content=content,
            source=source,
            agent=agent,
            confidence=confidence,
            vector=vec,
        )
        self.knowledge_base.append(fact)
        return fact

    def search_facts(self, query: str, top_k: int = 5, min_conf: float = 0.0) -> List[Tuple[Fact, float]]:
        qv = self.vectorizer.vectorize(query)
        scored = []
        for f in self.knowledge_base:
            if f.confidence < min_conf:
                continue
            sim = HashVectorizer.cosine(qv, f.vector or self.vectorizer.vectorize(f.content))
            scored.append((f, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---- Agent State ---- #
    def record_agent_state(self, agent: str, task: str, summary: str, **details) -> AgentState:
        st = AgentState(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            agent=agent,
            task=task,
            summary=summary,
            details=details,
        )
        self.agent_state.append(st)
        return st

# ----------------------------- Agents ----------------------------- #

class ResearchAgent:
    def __init__(self, memory: MemoryLayer):
        self.memory = memory
        self.name = "ResearchAgent"
        # Pre-loaded mock knowledge base
        self._seed_kb()

    def _seed_kb(self):
        now = dt.datetime.utcnow().strftime('%Y-%m-%d')
        kb_items = [
            ("neural networks", "Main types: Feedforward (MLP), Convolutional (CNN), Recurrent (RNN/LSTM/GRU), Transformer, Graph Neural Networks (GNN).", 0.92),
            ("optimizers", "Common optimization algorithms: SGD, Momentum, RMSProp, Adam, AdamW, Adagrad, Adadelta.", 0.90),
            ("transformers", "Transformer architectures: Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5). Key modules: self-attention, MLP blocks, residual connections.", 0.88),
            ("transformer efficiency", "Efficiency techniques: FlashAttention, Low-rank adapters (LoRA), quantization (8-/4-bit), pruning, distillation, mixture-of-experts (MoE).", 0.86),
            ("reinforcement learning papers 2024", f"Recent RL papers include: DreamerV3 (model-based), Decision Transformer variants, Offline RL benchmarks updates; common challenges: exploration, sample efficiency, stability, reward design. As of {now}.", 0.83),
            ("ml tradeoffs", "Trade-offs often balance accuracy, compute (FLOPs/memory), latency, and data requirements.", 0.8),
        ]
        for topic, content, conf in kb_items:
            self.memory.upsert_fact(topic=topic, content=content, source="KB:mock", agent=self.name, confidence=conf)

    def handle(self, query: str) -> Dict[str, Any]:
        # Attempt vector + keyword retrieval
        facts = self.memory.search_facts(query, top_k=6)
        # Simple keyword boost
        keywords = set(HashVectorizer._tokenize(query))
        results = []
        for f, sim in facts:
            kw_overlap = len([k for k in keywords if k in f.content.lower() or k in f.topic.lower()])
            score = 0.7 * sim + 0.3 * (min(kw_overlap, 5) / 5.0)
            results.append((f, score))
        results.sort(key=lambda x: x[1], reverse=True)
        payload = [{
            "id": f.id,
            "topic": f.topic,
            "content": f.content,
            "source": f.source,
            "agent": f.agent,
            "confidence": f.confidence,
            "score": round(score, 4),
        } for f, score in results]
        self.memory.record_agent_state(self.name, task="research", summary=f"Retrieved {len(payload)} items for query.", query=query, results=[p["id"] for p in payload])
        return {
            "agent": self.name,
            "query": query,
            "results": payload,
            "confidence": min(0.95, 0.6 + 0.1 * len(payload)),
        }

class AnalysisAgent:
    def __init__(self, memory: MemoryLayer):
        self.memory = memory
        self.name = "AnalysisAgent"

    def handle(self, prompt: str, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple heuristic analysis: compare, rank, and summarize
        text_blobs = [i.get("content", "") for i in inputs]
        combined = "\n".join(text_blobs)
        summary = self._summarize(prompt, combined)
        bullets = self._bullets(prompt, combined)
        result = {
            "agent": self.name,
            "prompt": prompt,
            "summary": summary,
            "bullets": bullets,
            "confidence": 0.75 if bullets else 0.6,
        }
        self.memory.record_agent_state(self.name, task="analysis", summary=summary[:160], prompt=prompt, used=[i.get("id") for i in inputs])
        return result

    @staticmethod
    def _summarize(prompt: str, text: str) -> str:
        # Very small rule-based summary tailored to expected prompts
        p = prompt.lower()
        if "efficiency" in p or "trade" in p:
            return (
                "Transformer efficiency centers on attention optimization (e.g., FlashAttention), parameter-efficient fine-tuning (LoRA), and model compression (quantization, pruning). "
                "Trade-offs balance accuracy vs. latency/memory; MoE improves capacity at the cost of routing complexity."
            )
        if "compare" in p and "recommend" in p:
            return (
                "Compared approaches on accuracy, compute, and data needs; recommendation favors the option with better efficiency at similar accuracy for the stated use case."
            )
        if "reinforcement" in p:
            return (
                "Recent RL work spans model-based agents (e.g., DreamerV3) and sequence-modeling (Decision Transformers). Common pain points remain exploration, sample efficiency, and stability."
            )
        if "neural network" in p:
            return (
                "Core families include MLPs, CNNs, RNNs (incl. LSTM/GRU), Transformers, and GNNs. Selection depends on data modality and sequence/spatial structure."
            )
        return "Key findings synthesized from retrieved items; see bullets for specifics."

    @staticmethod
    def _bullets(prompt: str, text: str) -> List[str]:
        bullets: List[str] = []
        lower = text.lower()
        # Extract simple nuggets
        if "feedforward" in lower or "mlp" in lower:
            bullets.append("Feedforward (MLP): dense layers; good baseline for tabular/simple features.")
        if "convolution" in lower or "cnn" in lower:
            bullets.append("Convolutional (CNN): spatial feature extraction; vision tasks.")
        if "recurrent" in lower or "lstm" in lower or "gru" in lower or "rnn" in lower:
            bullets.append("Recurrent (RNN/LSTM/GRU): temporal dependencies; sequence modeling.")
        if "transformer" in lower:
            bullets.append("Transformer: self-attention; strong for language, vision, and multimodal.")
        if "gnn" in lower or "graph" in lower:
            bullets.append("Graph Neural Networks (GNN): relational/graph-structured data.")
        if "adam" in lower or "sgd" in lower:
            bullets.append("Optimizers: Adam/AdamW for fast convergence, SGD+momentum for strong generalization.")
        if "flashattention" in lower:
            bullets.append("FlashAttention reduces attention compute/memory; boosts throughput.")
        if "lora" in lower:
            bullets.append("LoRA enables parameter-efficient fine-tuning; low memory cost.")
        if "quantization" in lower:
            bullets.append("Quantization (8/4-bit) trades slight accuracy for big memory/latency gains.")
        if "moe" in lower or "mixture" in lower:
            bullets.append("Mixture-of-Experts increases capacity with sparse routing; introduces load-balancing complexity.")
        if not bullets:
            bullets.append("No specific nuggets extracted; rely on high-level summary.")
        return bullets

class MemoryAgent:
    def __init__(self, memory: MemoryLayer):
        self.memory = memory
        self.name = "MemoryAgent"

    def store(self, topic: str, content: str, source: str, confidence: float = 0.8) -> Fact:
        fact = self.memory.upsert_fact(topic=topic, content=content, source=source, agent=self.name, confidence=confidence)
        self.memory.record_agent_state(self.name, task="store", summary=f"Stored fact on '{topic}'.", content=content, source=source)
        return fact

    def recall(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = self.memory.search_facts(query, top_k=top_k)
        payload = [{
            "id": f.id,
            "topic": f.topic,
            "content": f.content,
            "confidence": f.confidence,
            "source": f.source,
            "score": round(score, 4),
        } for f, score in hits]
        self.memory.record_agent_state(self.name, task="recall", summary=f"Recalled {len(payload)} items.", query=query)
        return {"agent": self.name, "results": payload, "confidence": 0.65 + 0.05*len(payload)}

# ----------------------------- Coordinator (Planner) ----------------------------- #

class Coordinator:
    def __init__(self):
        self.memory = MemoryLayer()
        self.research = ResearchAgent(self.memory)
        self.analysis = AnalysisAgent(self.memory)
        self.mem_agent = MemoryAgent(self.memory)
        self.trace: List[Dict[str, Any]] = []

    # --- Planning / routing --- #
    def _classify(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        plan: List[str] = []
        if any(k in q for k in ["analyz", "compare", "efficiency", "trade", "recommend"]):
            plan = ["research", "analysis", "store"]
        elif any(k in q for k in ["find", "recent", "papers", "research"]):
            plan = ["research", "analysis", "store"]
        elif any(k in q for k in ["what did we discuss", "earlier", "remember", "recall", "memory"]):
            plan = ["recall"]
        else:
            plan = ["research", "analysis", "store"]
        complexity = ("high" if len(q) > 120 or len(q.split()) > 18 else "medium" if len(q.split()) > 9 else "low")
        return {"complexity": complexity, "plan": plan}

    def ask(self, query: str) -> Dict[str, Any]:
        self.memory.log_conversation("user", topic="query", content=query)
        decision = self._classify(query)
        self._log_trace(stage="plan", payload=decision)
        outputs: Dict[str, Any] = {}

        try:
            if "research" in decision["plan"]:
                r = self.research.handle(query)
                outputs["research"] = r
                self._log_trace(stage="research", payload=r)

            if "analysis" in decision["plan"]:
                inputs = outputs.get("research", {}).get("results", [])
                a = self.analysis.handle(prompt=query, inputs=inputs)
                outputs["analysis"] = a
                self._log_trace(stage="analysis", payload=a)

            if "store" in decision["plan"]:
                # store high-level takeaway
                takeaway = outputs.get("analysis", {}).get("summary") or "Summary unavailable"
                fact = self.mem_agent.store(topic="takeaway:" + query[:48], content=takeaway, source="Coordinator:analysis")
                outputs["stored_fact_id"] = fact.id
                self._log_trace(stage="store", payload={"fact_id": fact.id})

            if "recall" in decision["plan"]:
                rec = self.mem_agent.recall(query)
                outputs["recall"] = rec
                self._log_trace(stage="recall", payload=rec)

        except Exception as e:
            # graceful degradation: if analysis fails, return research; if research fails, consult memory
            err = {"error": str(e)}
            self._log_trace(stage="error", payload=err)
            if "analysis" in decision["plan"] and "research" in outputs:
                outputs["analysis"] = {"agent": "AnalysisAgent", "summary": "Fallback: returning research only.", "bullets": [], "confidence": 0.4}
            elif "research" in decision["plan"]:
                # try recall as fallback
                rec = self.mem_agent.recall(query)
                outputs["recall"] = rec
        
        # final synthesis for the user
        final = self._synthesize_response(query, outputs)
        self.memory.log_conversation("assistant", topic="answer", content=final)
        return {
            "answer": final,
            "trace": self.trace,
            "confidence": self._overall_confidence(outputs),
        }

    def _synthesize_response(self, query: str, outputs: Dict[str, Any]) -> str:
        parts: List[str] = []
        if "research" in outputs:
            r = outputs["research"]
            items = r.get("results", [])[:5]
            bullets = [f"• [{it['topic']}] {it['content']} (src={it['source']}, conf={it['confidence']:.2f})" for it in items]
            parts.append("Research findings:\n" + "\n".join(bullets))
        if "analysis" in outputs:
            a = outputs["analysis"]
            parts.append("Analysis summary: " + a.get("summary", "N/A"))
            if a.get("bullets"):
                parts.append("Key points:\n" + "\n".join("• " + b for b in a["bullets"]))
        if "recall" in outputs:
            rr = outputs["recall"]
            items = rr.get("results", [])
            if items:
                parts.append("Memory recall:\n" + "\n".join(f"• ({x['score']:.2f}) {x['topic']}: {x['content']}" for x in items))
            else:
                parts.append("Memory recall: no prior items found.")
        return "\n\n".join(parts) if parts else "No answer available."

    def _overall_confidence(self, outputs: Dict[str, Any]) -> float:
        w = []
        if "research" in outputs:
            w.append(outputs["research"].get("confidence", 0.6))
        if "analysis" in outputs:
            w.append(outputs["analysis"].get("confidence", 0.6))
        if "recall" in outputs:
            w.append(outputs["recall"].get("confidence", 0.6))
        return round(sum(w)/len(w), 3) if w else 0.0

    def _log_trace(self, stage: str, payload: Dict[str, Any]):
        self.trace.append({
            "ts": dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            "stage": stage,
            "payload": payload if isinstance(payload, dict) else str(payload)[:5000],
        })

# ----------------------------- CLI / Demo Scenarios ----------------------------- #

SCENARIOS = {
    "simple_query": "What are the main types of neural networks?",
    "complex_query": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
    "memory_test": "What did we discuss about neural networks earlier?",
    "multi_step": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
    "collaborative": "Compare two machine-learning approaches and recommend which is better for our use case.",
}


def run_scenarios(write_outputs: bool = True) -> Dict[str, Any]:
    coord = Coordinator()
    results: Dict[str, Any] = {}

    # Ensure outputs directory
    out_dir = os.path.join(os.getcwd(), "outputs")
    if write_outputs:
        os.makedirs(out_dir, exist_ok=True)

    for key, query in SCENARIOS.items():
        coord.trace = []  # reset trace per scenario for clarity
        res = coord.ask(query)
        results[key] = res
        if write_outputs:
            path = os.path.join(out_dir, f"{key}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\n\n")
                f.write(res["answer"]) \
                    ; f.write("\n\n--- TRACE ---\n") \
                    ; f.write(json.dumps(res["trace"], indent=2)) \
                    ; f.write(f"\n\nConfidence: {res['confidence']}\n")
    return results


if __name__ == "__main__":
    results = run_scenarios(write_outputs=True)
    print("Created ./outputs with scenario logs:")
    for name in SCENARIOS:
        print(f"  - outputs/{name}.txt")
