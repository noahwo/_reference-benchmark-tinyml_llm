import json
import os

notebook_path = '/home/han/Projects/reference-benchmark-tinyml_llm/data_analysis/2025/11.06/ipynb/langfuse_export.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Original code snippet to find
original_code_part = """        rows = [
            {
                "num_run": trace["metadata"]["num_run"],
                "name": trace["name"],
                "trace_id": trace["id"],
                "batch_id": trace["session_id"],
                # "latency": cal_time(trace),
                # "latency": round(trace["latency"], 2),
                **extract_observation_details(
                    trace["observations"],
                    trace["id"],
                ),
                "status": (
                    "failure"
                    if trace["output"]["status"].lower() == "failed"
                    else "success"
                ),
                "tags": trace["tags"],
                "timestamp": int(parser.isoparse(trace["timestamp"]).timestamp()),
            }
            for trace in traces
        ]"""

# New code to replace it with
new_code_part = """        rows = []
        for trace in traces:
            details = extract_observation_details(trace["observations"], trace["id"])
            
            # Determine status: prefer output status if available, otherwise use calculated status
            status = details["status"]
            if trace.get("output") and isinstance(trace["output"], dict):
                status = (
                    "failure"
                    if trace["output"].get("status", "").lower() == "failed"
                    else "success"
                )
            
            row = {
                "num_run": trace["metadata"]["num_run"],
                "name": trace["name"],
                "trace_id": trace["id"],
                "batch_id": trace["session_id"],
                # "latency": cal_time(trace),
                # "latency": round(trace["latency"], 2),
                **details,
                "status": status,
                "tags": trace["tags"],
                "timestamp": int(parser.isoparse(trace["timestamp"]).timestamp()),
            }
            rows.append(row)"""

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if original_code_part in source:
            print("Found the target cell.")
            new_source = source.replace(original_code_part, new_code_part)
            # Split back into lines, keeping newlines
            # cell['source'] = [line + '\n' for line in new_source.split('\n')]
            # Actually, splitlines keeps it cleaner but we need to preserve exact structure if possible.
            # But for notebooks, a list of strings is standard.
            # Let's just split by \n and add \n to all but last if needed, or just use splitlines(True)
            cell['source'] = new_source.splitlines(True)
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched the notebook.")
else:
    print("Could not find the code to replace.")
