# GenPark Gradient Episodic Memory GEM Projector Skill

A-GEM gradient projection engine preventing negative interference and forgetting during continual task adaptation.

Learn more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Current Task Gradient g] --> B{g . g_ref >= 0?}
    B -->|Yes| C[Apply Gradient Directly]
    B -->|No: Interference| D[Orthogonal Projection g_proj = g - (g.g_ref / |g_ref|^2) * g_ref]
    D --> E[g_proj . g_ref = 0: Zero Forgetting Guaranteed]
```

## Features
- Efficient orthogonal subspace projection.
- Eliminates backward transfer penalty.
- Pure Python standard library.
