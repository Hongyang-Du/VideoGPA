import os
import json

def save_json(dataset_name, video_name, out_dir, json_name, results):
    # Filter threshold keys (convert to string first)
    clean_results = {
        str(k): v 
        for k, v in results.items()
        if not str(k).startswith("_")
    }

    data = {
        "dataset": dataset_name,
        "video": video_name,
        "extrinsics": results.get("_extrinsic", None),
        "thresholds": clean_results,
    }

    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVE] Summary JSON → {json_path}")

