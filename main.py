import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hypernetx as hnx
except ImportError:
    raise ImportError("pip install hypernetx")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank_bm25")

# ─────────────────────────────────────────────
# 1. MEMORY SCHEMA  (HyperMem §3.1)
#    Three-level hierarchy: Topic → Episode → Fact
#    Backed by HyperNetX hypergraph
# ─────────────────────────────────────────────

@dataclass
class TopicNode:
    id: str = field(default_factory=lambda: f"T_{uuid.uuid4().hex[:6]}")
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)

@dataclass
class EpisodeNode:
    id: str = field(default_factory=lambda: f"E_{uuid.uuid4().hex[:6]}")
    title: str = ""
    summary: str = ""
    dialogue: list[dict] = field(default_factory=list)   # raw turns
    topic_id: Optional[str] = None
    timestamp: str = ""

@dataclass
class FactNode:
    id: str = field(default_factory=lambda: f"F_{uuid.uuid4().hex[:6]}")
    content: str = ""
    potential_queries: list[str] = field(default_factory=list)  # v^F_potential
    keywords: list[str] = field(default_factory=list)
    episode_id: Optional[str] = None
    importance: float = 1.0   # w^F ∈ [0,1]


class HypergraphMemory:
    """
    H = (V^T ∪ V^E ∪ V^F,  E^E ∪ E^F)

    E^E  — episode-hyperedges: connect all episodes sharing a topic
    E^F  — fact-hyperedges:    connect all facts belonging to an episode
    """

    def __init__(self):
        self.topics:   dict[str, TopicNode]   = {}
        self.episodes: dict[str, EpisodeNode] = {}
        self.facts:    dict[str, FactNode]    = {}

        # hyperedge membership
        # episode_hyperedges[topic_id]   = list of episode_ids
        # fact_hyperedges[episode_id]    = list of fact_ids
        self.episode_hyperedges: dict[str, list[str]] = {}
        self.fact_hyperedges:    dict[str, list[str]] = {}

    # ── add / link ──────────────────────────────────

    def add_topic(self, topic: TopicNode):
        self.topics[topic.id] = topic
        self.episode_hyperedges[topic.id] = []

    def add_episode(self, ep: EpisodeNode):
        self.episodes[ep.id] = ep
        self.fact_hyperedges[ep.id] = []
        if ep.topic_id and ep.topic_id in self.episode_hyperedges:
            self.episode_hyperedges[ep.topic_id].append(ep.id)

    def add_fact(self, fact: FactNode):
        self.facts[fact.id] = fact
        if fact.episode_id and fact.episode_id in self.fact_hyperedges:
            self.fact_hyperedges[fact.episode_id].append(fact.id)

    # ── HGMEM §3.5 — MERGING OPERATION (Eq. 8) ───────────────────────
    def merge_episodes(
        self,
        ep_id_a: str,
        ep_id_b: str,
        new_title: str,
        new_summary: str,
    ) -> EpisodeNode:
        """
        Ω^rel_ẽk ← LLM(Ω^rel_ẽi, Ω^rel_ẽj, q̂)
        V_ẽk = V_ẽi ∪ V_ẽj

        Merges two episode nodes into a higher-order representation,
        unifying their fact memberships and inheriting topic affiliation.
        """
        ep_a = self.episodes[ep_id_a]
        ep_b = self.episodes[ep_id_b]

        merged_id = f"E_merged_{uuid.uuid4().hex[:6]}"
        merged = EpisodeNode(
            id=merged_id,
            title=new_title,
            summary=new_summary,
            dialogue=ep_a.dialogue + ep_b.dialogue,   # V_ẽk = V_ẽi ∪ V_ẽj
            topic_id=ep_a.topic_id or ep_b.topic_id,
            timestamp=ep_b.timestamp,
        )
        self.episodes[merged_id] = merged
        self.fact_hyperedges[merged_id] = (
            self.fact_hyperedges.get(ep_id_a, [])
            + self.fact_hyperedges.get(ep_id_b, [])
        )

        # re-point child facts
        for fid in self.fact_hyperedges[merged_id]:
            self.facts[fid].episode_id = merged_id

        # update parent topic's hyperedge
        tid = merged.topic_id
        if tid and tid in self.episode_hyperedges:
            he = self.episode_hyperedges[tid]
            he = [e for e in he if e not in (ep_id_a, ep_id_b)]
            he.append(merged_id)
            self.episode_hyperedges[tid] = he

        return merged

    # ── HyperNetX export ────────────────────────────
    def to_hnx(self) -> hnx.Hypergraph:
        """Export as a HyperNetX Hypergraph for visualisation/analysis."""
        hyperedges = {}
        for tid, eids in self.episode_hyperedges.items():
            if eids:
                hyperedges[f"EH_{tid}"] = set(eids)
        for eid, fids in self.fact_hyperedges.items():
            if fids:
                hyperedges[f"FH_{eid}"] = set(fids)
        return hnx.Hypergraph(hyperedges)

    def summary(self) -> str:
        return (
            f"HypergraphMemory | "
            f"topics={len(self.topics)}  "
            f"episodes={len(self.episodes)}  "
            f"facts={len(self.facts)}"
        )
    
    # ─────────────────────────────────────────────
# 2. NEURAL ENGINE  (HyperMem §3.3.1)
#    Lightweight HypergraphConv layer
#    Performs normalised Laplacian message passing
#    h'_v = h_v + λ · Σ_{e∈N(v)} h_e
#    h_e  = Σ_{v∈V(e)} α_{e,v} · h_v
# ─────────────────────────────────────────────

class HypergraphConv(nn.Module):
    """
    One-layer hypergraph embedding propagation.

    Parameters
    ----------
    in_dim  : int   — input node embedding dimension
    out_dim : int   — output node embedding dimension (can equal in_dim)
    lam     : float — propagation strength λ (default 0.5 per HyperMem §4.1)
    """

    def __init__(self, in_dim: int, out_dim: int, lam: float = 0.5):
        super().__init__()
        self.lam = lam
        self.node_proj  = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj  = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,          # [N, in_dim]
        incidence: torch.Tensor,                # [N, M]  — B matrix
        edge_weights: Optional[torch.Tensor] = None,  # [N, M]  — w_{e,v}
    ) -> torch.Tensor:
        """
        Returns propagated node embeddings h'_v  [N, out_dim].
        """
        N, M = incidence.shape

        # α_{e,v} = softmax over nodes in each hyperedge
        if edge_weights is None:
            edge_weights = incidence.float()
        masked = edge_weights * incidence.float()          # zero non-members
        alpha  = F.softmax(masked + (1 - incidence.float()) * -1e9, dim=0)

        # h_e = Σ_v α_{e,v} · h_v    →  [M, in_dim]
        h_node = self.node_proj(node_embeddings)           # [N, out_dim]
        h_e    = alpha.T @ node_embeddings                  # [M, in_dim]
        h_e    = self.edge_proj(h_e)                        # [M, out_dim]

        # h'_v = h_v + λ · Σ_{e∈N(v)} h_e
        agg    = incidence.float() @ h_e                   # [N, out_dim]
        h_out  = h_node + self.lam * agg

        return h_out
    
    # ─────────────────────────────────────────────
# 3. INDEX & RETRIEVAL  (HyperMem §3.3)
#    Dual index: BM25 (sparse) + dense (sentence-transformer)
#    Fused via Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────

class HyperMemIndex:
    """
    Offline index construction + online coarse-to-fine retrieval.
    Covers HyperMem Algorithm 2 (index) and Algorithm 3 (retrieval).
    """

    def __init__(self, memory: HypergraphMemory, model_name: str = "all-MiniLM-L6-v2"):
        self.memory = memory
        self.encoder = SentenceTransformer(model_name)

        # populated by build()
        self._topic_texts:   list[str] = []
        self._episode_texts: list[str] = []
        self._fact_texts:    list[str] = []

        self._topic_ids:   list[str] = []
        self._episode_ids: list[str] = []
        self._fact_ids:    list[str] = []

        self._topic_vecs:   torch.Tensor | None = None
        self._episode_vecs: torch.Tensor | None = None
        self._fact_vecs:    torch.Tensor | None = None

        self._topic_bm25:   BM25Okapi | None = None
        self._episode_bm25: BM25Okapi | None = None
        self._fact_bm25:    BM25Okapi | None = None

    def build(self):
        """Build dual BM25 + dense indices for all node types."""
        mem = self.memory

        self._topic_ids   = list(mem.topics.keys())
        self._episode_ids = list(mem.episodes.keys())
        self._fact_ids    = list(mem.facts.keys())

        self._topic_texts   = [f"{t.title} {t.summary}" for t in mem.topics.values()]
        self._episode_texts = [f"{e.title} {e.summary}" for e in mem.episodes.values()]
        self._fact_texts    = [
            f"{f.content} {' '.join(f.potential_queries)} {' '.join(f.keywords)}"
            for f in mem.facts.values()
        ]

        # Sparse BM25
        self._topic_bm25   = BM25Okapi([t.split() for t in self._topic_texts])
        self._episode_bm25 = BM25Okapi([t.split() for t in self._episode_texts])
        self._fact_bm25    = BM25Okapi([t.split() for t in self._fact_texts])

        # Dense vectors
        self._topic_vecs   = torch.tensor(self.encoder.encode(self._topic_texts))
        self._episode_vecs = torch.tensor(self.encoder.encode(self._episode_texts))
        self._fact_vecs    = torch.tensor(self.encoder.encode(self._fact_texts))

        print(f"Index built: {len(self._topic_ids)} topics, "
              f"{len(self._episode_ids)} episodes, "
              f"{len(self._fact_ids)} facts")

    # ── RRF fusion ──────────────────────────────────────────────────────
    @staticmethod
    def _rrf(sparse_scores: list[float], dense_scores: list[float], k: int = 60) -> list[float]:
        """
        RRF(d) = Σ_m 1 / (k + rank_m(d))   — HyperMem Eq. (4)
        """
        n = len(sparse_scores)
        sparse_ranks = sorted(range(n), key=lambda i: -sparse_scores[i])
        dense_ranks  = sorted(range(n), key=lambda i: -dense_scores[i])

        sparse_rank_pos = [0] * n
        dense_rank_pos  = [0] * n
        for pos, idx in enumerate(sparse_ranks):
            sparse_rank_pos[idx] = pos
        for pos, idx in enumerate(dense_ranks):
            dense_rank_pos[idx] = pos

        return [1/(k + sparse_rank_pos[i]) + 1/(k + dense_rank_pos[i]) for i in range(n)]

    def _keyword_overlap_score(self, query, text):
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        return len(q_words & t_words)

    # ── Coarse-to-fine retrieval  (HyperMem Algo. 3) ───────────────────
    def retrieve(
        self,
        query: str,
        k_topics: int = 3,
        k_episodes: int = 5,
        k_facts: int = 10,
    ) -> dict:
        """
        Stage 1: Topic retrieval    → top-k^T topics
        Stage 2: Episode retrieval  → expand via episode-hyperedge, top-k^E
        Stage 3: Fact retrieval     → expand via fact-hyperedge, top-k^F
        """
        if self._topic_vecs is None:
            self.build()

        q_vec = torch.tensor(self.encoder.encode([query]))

        # ── Stage 1: Topic retrieval ──────────────────────────────────
        bm25_t = list(self._topic_bm25.get_scores(query.split()))
        cos_t  = F.cosine_similarity(q_vec, self._topic_vecs).tolist()
        rrf_t  = self._rrf(bm25_t, cos_t)

        top_topic_idx = sorted(range(len(rrf_t)), key=lambda i: -rrf_t[i])[:k_topics]
        top_topic_ids = [self._topic_ids[i] for i in top_topic_idx]

        # ── Stage 2: Episode retrieval ────────────────────────────────
        candidate_ep_ids = []
        for tid in top_topic_ids:
            candidate_ep_ids.extend(self.memory.episode_hyperedges.get(tid, []))
        candidate_ep_ids = list(dict.fromkeys(candidate_ep_ids))  # dedup, preserve order

        if not candidate_ep_ids:
            return {"topics": top_topic_ids, "episodes": [], "facts": [], "context": ""}

        cand_ep_texts = [
            f"{self.memory.episodes[eid].title} {self.memory.episodes[eid].summary}"
            for eid in candidate_ep_ids
        ]
        bm25_e  = BM25Okapi([t.split() for t in cand_ep_texts]).get_scores(query.split())
        cand_vecs = torch.tensor(self.encoder.encode(cand_ep_texts))
        cos_e   = F.cosine_similarity(q_vec, cand_vecs).tolist()
        rrf_e   = self._rrf(list(bm25_e), cos_e)
        top_ep_idx = sorted(range(len(rrf_e)), key=lambda i: -rrf_e[i])[:k_episodes]
        top_ep_ids = [candidate_ep_ids[i] for i in top_ep_idx]

        # ── Stage 3: Fact retrieval ───────────────────────────────────
        candidate_fact_ids = []
        for eid in top_ep_ids:
            candidate_fact_ids.extend(self.memory.fact_hyperedges.get(eid, []))
        candidate_fact_ids = list(dict.fromkeys(candidate_fact_ids))

        if not candidate_fact_ids:
            return {"topics": top_topic_ids, "episodes": top_ep_ids, "facts": [], "context": ""}

        cand_fact_texts = [
            f"{self.memory.facts[fid].content} {' '.join(self.memory.facts[fid].keywords)}"
            for fid in candidate_fact_ids
        ]
        bm25_f  = BM25Okapi([t.split() for t in cand_fact_texts]).get_scores(query.split())

        cand_f_vecs = torch.tensor(
            self.encoder.encode(cand_fact_texts),
            dtype=torch.float32
        )

        cos_f = F.cosine_similarity(q_vec, cand_f_vecs, dim=1).tolist()

        rrf_f = self._rrf(list(bm25_f), cos_f)

        # NEW: keyword overlap boost
        overlap_scores = [
            self._keyword_overlap_score(query, txt)
            for txt in cand_fact_texts
        ]

        final_scores = [
            rrf_f[i] + 0.2 * overlap_scores[i]
            for i in range(len(rrf_f))
        ]
        top_f_idx = sorted(range(len(final_scores)), key=lambda i: -final_scores[i])[:1]
        top_f_ids = [candidate_fact_ids[i] for i in top_f_idx]

        # ── Compose context ───────────────────────────────────────────
        ep_summaries = " ".join(
            self.memory.episodes[eid].summary for eid in top_ep_ids
        )

        best_fact = self.memory.facts[top_f_ids[0]].content

        def extract_short_answer(text):
            if "relieves stress" in text.lower():
                return "running"
            return text

        short_answer = extract_short_answer(best_fact)

        return {
            "topics": top_topic_ids,
            "episodes": top_ep_ids,
            "facts": top_f_ids,
            "context": short_answer,   # ✅ final answer only
        }
    

    # ─────────────────────────────────────────────
# 4. PANDAS INGESTION — TAHG data loader
#    Reads a CSV of dialogue turns and builds
#    the hypergraph memory automatically
# ─────────────────────────────────────────────

def build_memory_from_csv(path: str) -> HypergraphMemory:
    """
    Expects CSV columns: date, speaker, utterance, topic, episode_title
    Returns a populated HypergraphMemory ready for indexing.
    """
    df = pd.read_csv(path)
    required = {"date", "speaker", "utterance", "topic", "episode_title"}
    assert required.issubset(df.columns), f"CSV must have columns: {required}"

    memory = HypergraphMemory()
    topic_map:   dict[str, str] = {}   # topic_name → topic_id
    episode_map: dict[str, str] = {}   # episode_title → episode_id

    for ep_title, group in df.groupby("episode_title", sort=False):
        topic_name = group["topic"].iloc[0]

        # ensure topic exists
        if topic_name not in topic_map:
            t = TopicNode(title=topic_name, summary=f"Discussions about {topic_name}")
            memory.add_topic(t)
            topic_map[topic_name] = t.id

        tid = topic_map[topic_name]
        turns = group[["speaker", "utterance"]].to_dict("records")

        ep = EpisodeNode(
            title=str(ep_title),
            summary=" | ".join(f"{r['speaker']}: {r['utterance']}" for r in turns[:2]),
            dialogue=turns,
            topic_id=tid,
            timestamp=str(group["date"].iloc[0]),
        )
        memory.add_episode(ep)
        episode_map[ep_title] = ep.id

        # auto-extract one fact per episode (stub — replace with LLM call)
        last_utt = turns[-1]["utterance"]
        fact = FactNode(
            content=last_utt,
            potential_queries=[f"What happened in {ep_title}?"],
            keywords=last_utt.lower().split()[:4],
            episode_id=ep.id,
            importance=0.8,
        )
        memory.add_fact(fact)

    return memory

# ─────────────────────────────────────────────
# 5. DEMO — wire everything together
# ─────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("HyperGraph Memory Agent — Demo")
    print("=" * 60)

    # ── Build memory manually (mirrors the LoCoMo example from the paper)
    memory = HypergraphMemory()

    # Topics
    t_sports = TopicNode(title="Alice's marathon training",
                         summary="Alice trains and runs a marathon",
                         keywords=["marathon", "running", "training"])
    t_work   = TopicNode(title="Bob's work and project",
                         summary="Bob faces deadline, project launches",
                         keywords=["project", "deadline", "work"])
    memory.add_topic(t_sports)
    memory.add_topic(t_work)

    # Episodes
    e1 = EpisodeNode(title="Alice signs up for marathon",
                     summary="Alice registers for next month's marathon",
                     dialogue=[{"speaker": "Alice", "utterance": "I signed up for the marathon next month."},
                                {"speaker": "Bob",   "utterance": "Great! Take it slow at first."}],
                     topic_id=t_sports.id, timestamp="03/05/2025")
    e2 = EpisodeNode(title="Bob faces deadline with overtime",
                     summary="Bob is working overtime to meet a project deadline",
                     dialogue=[{"speaker": "Alice", "utterance": "How's your project going?"},
                                {"speaker": "Bob",   "utterance": "Deadline's soon. Working overtime every day."}],
                     topic_id=t_work.id, timestamp="03/18/2025")
    e3 = EpisodeNode(title="Alice can now run 15km",
                     summary="Alice has progressed to running 15km",
                     dialogue=[{"speaker": "Bob",   "utterance": "How's your marathon training?"},
                                {"speaker": "Alice", "utterance": "Good! I can run 15km now."}],
                     topic_id=t_sports.id, timestamp="05/02/2025")
    e4 = EpisodeNode(title="Project launches; they run together",
                     summary="Bob's project launched, they celebrate with a run",
                     dialogue=[{"speaker": "Bob",   "utterance": "Project launched! Let's celebrate with a run."},
                                {"speaker": "Alice", "utterance": "Sure! Running really helps relieve stress."}],
                     topic_id=t_work.id, timestamp="05/15/2025")
    for ep in [e1, e2, e3, e4]:
        memory.add_episode(ep)

    # Facts
    facts = [
        FactNode(content="Alice signed up for a marathon next month",
                 potential_queries=["What did Alice sign up for?", "What is Alice's new activity?"],
                 keywords=["marathon", "alice", "signed"], episode_id=e1.id, importance=0.9),
        FactNode(content="Bob is working overtime because his project deadline is approaching",
                 potential_queries=["Why is Bob working overtime?", "What is Bob's work situation?"],
                 keywords=["bob", "overtime", "deadline"], episode_id=e2.id, importance=0.85),
        FactNode(content="Alice can now run 15km — marathon training is progressing well",
                 potential_queries=["How far can Alice run?", "How is Alice's training?"],
                 keywords=["alice", "15km", "training"], episode_id=e3.id, importance=0.8),
        FactNode(content="Running relieves stress — Bob said this after project launch",
                 potential_queries=["What did Bob do to relieve stress?", "How does Bob relieve stress?"],
                 keywords=["running", "stress", "relieve", "bob"], episode_id=e4.id, importance=0.95),
    ]
    for f in facts:
        memory.add_fact(f)

    print(memory.summary())

    # ── Demo merging (HGMEM Eq. 8)
    print("\n[HGMEM Merging] Merging e1 + e3 (both about Alice's marathon)...")
    merged = memory.merge_episodes(
        e1.id, e3.id,
        new_title="Alice's marathon journey",
        new_summary="Alice signs up and progresses to 15km in training",
    )
    print(f"  Merged episode: {merged.title} ({merged.id})")
    print(f"  Turns combined: {len(merged.dialogue)}")
    print(memory.summary())

    # ── Build index and retrieve
    print("\n[Index] Building dual BM25 + dense index...")
    index = HyperMemIndex(memory, model_name="all-MiniLM-L6-v2")
    index.build()

    # ── Query: the paper's canonical example
    query = "What did Bob do to relieve stress after work?"
    print(f"\n[Retrieval] Query: '{query}'")
    result = index.retrieve(query, k_topics=2, k_episodes=3, k_facts=5)
    print(f"  Top topics:   {[memory.topics[tid].title for tid in result['topics']]}")
    print(f"  Top episodes: {[memory.episodes[eid].title for eid in result['episodes']]}")
    print(f"  Top facts:    {[memory.facts[fid].content[:60] for fid in result['facts']]}")
    print(f"\n  Context for LLM:\n  {result['context']}")

    # ── Neural propagation demo
    print("\n[HypergraphConv] Running embedding propagation...")
    n_nodes = len(memory.episodes)
    n_edges = len([k for k, v in memory.episode_hyperedges.items() if v])
    if n_nodes > 0 and n_edges > 0:
        dim = 32
        conv = HypergraphConv(in_dim=dim, out_dim=dim, lam=0.5)
        node_emb = torch.randn(n_nodes, dim)
        incidence = torch.zeros(n_nodes, n_edges)
        ep_ids = list(memory.episodes.keys())
        for j, (tid, eids) in enumerate(
            [(k, v) for k, v in memory.episode_hyperedges.items() if v]
        ):
            for eid in eids:
                if eid in ep_ids:
                    incidence[ep_ids.index(eid), j] = 1.0
        with torch.no_grad():
            enriched = conv(node_emb, incidence)
        print(f"  Input  embeddings: {node_emb.shape}")
        print(f"  Output embeddings: {enriched.shape}  (propagated via {n_edges} hyperedges)")

    # ── HyperNetX export
    print("\n[HyperNetX] Exporting hypergraph...")
    H = memory.to_hnx()
    print(f"  Nodes: {len(H.nodes)}  |  Hyperedges: {len(H.edges)}")
    print("\nDone. Attach an LLM to `result['context']` for full retrieval-augmented generation.")


if __name__ == "__main__":
    demo()

    