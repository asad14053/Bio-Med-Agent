from pathlib import Path

# Input file (exact path)
INPUT_FILE = Path(
    "data/preprocess/tb_v1/v1/preprocess_manifest_train1.jsonl"
)

# Output directory = same directory as input file
BASE_DIR = INPUT_FILE.parent

# Output files and line ranges (1-indexed)
OUTPUT_RANGES = {
    "preprocess_manifest_base.jsonl": (1, 20000),
    "preprocess_manifest_bquery1.jsonl": (20001, 30000),
    "preprocess_manifest_bquery2.jsonl": (30001, 40000),
    "preprocess_manifest_bquery3.jsonl": (40001, 50000),
}


def split_jsonl(input_file: Path, output_ranges: dict, base_dir: Path):
    output_files = {
        name: (base_dir / name).open("w", encoding="utf-8")
        for name in output_ranges
    }

    try:
        with input_file.open("r", encoding="utf-8") as infile:
            for idx, line in enumerate(infile, start=1):
                for out_name, (start, end) in output_ranges.items():
                    if start <= idx <= end:
                        output_files[out_name].write(line)
                        break
    finally:
        for f in output_files.values():
            f.close()


if __name__ == "__main__":
    split_jsonl(INPUT_FILE, OUTPUT_RANGES, BASE_DIR)
    print("✅ JSONL split completed.")
    print("📁 Output directory:", BASE_DIR)
