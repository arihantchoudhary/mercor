#!/usr/bin/env python3
"""
Get actual HellaSwag examples to understand the context/continuation format.
"""
import json
from lm_eval.tasks import TaskManager, get_task_dict

def get_sample_hellaswag_requests():
    """Get a few sample HellaSwag requests to understand the format."""
    task_manager = TaskManager()
    task_dict = get_task_dict(["hellaswag"], task_manager)
    task = task_dict["hellaswag"]

    # Build docs
    task.build_all_requests(limit=3, rank=0, world_size=1)

    print("=== HellaSwag request format ===")
    instances = task.instances
    for i, inst in enumerate(instances[:12]):
        ctx, cont = inst.args
        print(f"\nInstance {i}:")
        print(f"  request_type: {inst.request_type}")
        print(f"  context (last 200 chars): ...{repr(ctx[-200:])}")
        print(f"  continuation (first 100): {repr(cont[:100])}")
        if i >= 11:
            break
    print()


if __name__ == "__main__":
    get_sample_hellaswag_requests()
