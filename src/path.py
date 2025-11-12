from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]

DATA_DIR = ROOT_DIR / "data"

if __name__ == "__main__":
    print(ROOT_DIR)
    print(DATA_DIR)