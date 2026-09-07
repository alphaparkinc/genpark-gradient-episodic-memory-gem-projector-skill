"""
MCP Server for Gradient Episodic Memory GEM Projector Skill
"""

import json
import sys
from client import GEMGradientProjector

proj = GEMGradientProjector()

def handle_call(name: str, args: dict) -> dict:
    if name == "project_gradient":
        curr = args.get("current_gradient", {})
        ref = args.get("ref_gradient", {})
        dot_init = proj.dot_product(curr, ref)
        g_res = proj.project_gradient(curr, ref)
        dot_final = proj.dot_product(g_res, ref)
        return {"projected_gradient": g_res, "initial_dot": dot_init, "final_dot": dot_final}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
