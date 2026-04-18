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