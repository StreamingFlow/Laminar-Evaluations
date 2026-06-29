import os
import json
from tqdm import tqdm
from laminar.conversion.ConvertToPE import ConvertToPE
from laminar.llms.LLMConnector import LLMConnector
from laminar.llms.queries_templates import REQUEST_DESCRIPTION_CONTEXT_QUERIES

connector = LLMConnector()
DATASET_NAME = "python_all.jsonl"
OUTPUT_NAME = "python_all_processed.jsonl"

if not os.path.exists(DATASET_NAME):
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("code-search-net/code_search_net", "python")
    combined = concatenate_datasets([ds[split] for split in ds])
    combined.to_json(DATASET_NAME)

# Count lines up front so the progress bar has a total
with open(DATASET_NAME, "r") as f:
    total_lines = sum(1 for _ in f)

if os.path.exists(OUTPUT_NAME):
    with open(OUTPUT_NAME, "r") as f:
        line_skip = sum(1 for _ in f)

with open(OUTPUT_NAME, "w") as outfile:
    with open(DATASET_NAME, "r") as f:
        for line in tqdm(f, total=total_lines, desc="Processing", unit="line"):

            if line_skip > 0:
                line_skip -= 1
                continue

            try:

                line_json = json.loads(line)
                converted = ConvertToPE(line_json["func_code_string"], False)

                if converted.pe:
                    entry = {
                        "ID": line_json["whole_func_string"],
                        'entry_expected_desc': line_json["func_documentation_string"],
                        'entry_obtained_desc': connector.describe(
                            component_name=converted.className, kind="pe", code=converted.pe,
                            provider="openai",
                            context_queries=REQUEST_DESCRIPTION_CONTEXT_QUERIES
                        )
                    }
                    outfile.write(json.dumps(entry) + "\n")

            except Exception:
                pass

