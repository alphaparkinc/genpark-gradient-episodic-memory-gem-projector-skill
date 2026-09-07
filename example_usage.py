"""
Demonstration of Gradient Episodic Memory GEM Projector Skill
"""

from client import GEMGradientProjector

def main():
    print("=== Averaged Gradient Episodic Memory (A-GEM) Gradient Projection ===")
    projector = GEMGradientProjector()

    # Past task reference gradient
    g_ref = {"param_1": 1.0, "param_2": 2.0, "param_3": -1.0}

    # Case 1: Interfering current gradient (negative dot product)
    g_interfering = {"param_1": -2.0, "param_2": -1.0, "param_3": 0.5}
    dot_before = projector.dot_product(g_interfering, g_ref)
    print(f"Interfering dot product (g . g_ref): {dot_before:.4f} (< 0 causes catastrophic forgetting!)")

    g_proj = projector.project_gradient(g_interfering, g_ref)
    dot_after = projector.dot_product(g_proj, g_ref)
    print(f"Projected Gradient: {g_proj}")
    print(f"Orthogonal dot product after projection: {dot_after:.6f} (>= 0 guarantees safety!)")

    assert dot_after >= -1e-6
    print("A-GEM Gradient Projector Verification PASS!")

if __name__ == "__main__":
    main()
