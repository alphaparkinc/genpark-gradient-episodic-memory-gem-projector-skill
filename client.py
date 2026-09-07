"""
Gradient Episodic Memory GEM Projector Skill Client
Pure Python Standard Library implementation of Averaged Gradient Episodic Memory (A-GEM / Lopez-Paz & Ranzato).
Projects proposed task gradients onto the orthogonal subspace of past task reference gradients
if the inner product is negative (g . g_ref < 0), guaranteeing non-negative transfer.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class GEMGradientProjector:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    @staticmethod
    def dot_product(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        return sum(v1[k] * v2.get(k, 0.0) for k in v1)

    def project_gradient(self, current_grad: Dict[str, float], ref_grad: Dict[str, float]) -> Dict[str, float]:
        """
        If g . g_ref >= 0: no projection needed (positive/neutral transfer).
        If g . g_ref < 0: project g onto {v : v . g_ref >= 0}:
        g_proj = g - ( (g . g_ref) / (g_ref . g_ref) ) * g_ref
        """
        dot_curr_ref = self.dot_product(current_grad, ref_grad)
        if dot_curr_ref >= 0:
            return dict(current_grad)

        dot_ref_ref = self.dot_product(ref_grad, ref_grad) + self.eps
        scale = dot_curr_ref / dot_ref_ref

        projected = {}
        for k, g_val in current_grad.items():
            r_val = ref_grad.get(k, 0.0)
            projected[k] = g_val - scale * r_val

        return projected
