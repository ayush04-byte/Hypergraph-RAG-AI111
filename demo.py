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