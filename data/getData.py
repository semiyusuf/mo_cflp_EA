"""
Download the six required OR-Library instances into data/ and check that each
file parses with the expected size (m facilities, n = 50 customers).

    python scripts/download_data.py
"""
import urllib.request

from config import ALL_INSTANCES, DATA_DIR

from src.data_loader import load_instance

BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
EXPECTED_M = {"cap41": 16, "cap42": 16, "cap101": 25, "cap102": 25, "cap121": 50, "cap122": 50}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name in ALL_INSTANCES:
        path = DATA_DIR / f"{name}.txt"
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE_URL}{name}.txt", path)
        inst = load_instance(path)
        assert (inst.m, inst.n) == (EXPECTED_M[name], 50), f"{name}: unexpected size {inst.m}x{inst.n}"
        print(f"{name}: m={inst.m}, n={inst.n}, total demand={inst.total_demand:.0f}, "
              f"total capacity={inst.capacity.sum():.0f}  OK")


if __name__ == "__main__":
    main()
