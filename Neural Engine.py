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