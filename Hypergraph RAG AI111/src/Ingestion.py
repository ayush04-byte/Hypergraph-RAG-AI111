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