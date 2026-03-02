import os
import re
import sys
from collections import defaultdict


FINAL_RE = re.compile(
    r'^Final,\s+(.+?)\s+\((-?\d+)\),\s+(.+?)\s+\((-?\d+)\)\s*$'
)


def parse_final_line(line):
    m = FINAL_RE.match(line.strip())
    if not m:
        return None
    bot1, s1, bot2, s2 = m.groups()
    return bot1, int(s1), bot2, int(s2)


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_last_glogs.py <num_matches>")
        sys.exit(1)

    try:
        last_k = int(sys.argv[1])
    except ValueError:
        print("num_matches must be an integer")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")

    if not os.path.isdir(logs_dir):
        print(f"No logs directory found at {logs_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(logs_dir) if f.endswith(".glog")]
    if not files:
        print("No .glog files found.")
        sys.exit(1)

    files.sort()  # timestamped names; lexicographic == chronological
    selected = files[-last_k:]

    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "bankroll": 0, "matches": 0})

    parsed = 0
    for fname in selected:
        path = os.path.join(logs_dir, fname)
        try:
            with open(path, "r") as f:
                final_line = None
                for line in f:
                    if line.startswith("Final,"):
                        final_line = line
                if not final_line:
                    continue
        except OSError:
            continue

        res = parse_final_line(final_line)
        if not res:
            continue

        bot1, s1, bot2, s2 = res
        parsed += 1

        for bot, score in [(bot1, s1), (bot2, s2)]:
            stats[bot]["bankroll"] += score
            stats[bot]["matches"] += 1

        if s1 > s2:
            stats[bot1]["wins"] += 1
            stats[bot2]["losses"] += 1
        elif s2 > s1:
            stats[bot2]["wins"] += 1
            stats[bot1]["losses"] += 1
        else:
            stats[bot1]["draws"] += 1
            stats[bot2]["draws"] += 1

    if parsed == 0:
        print("No valid Final lines parsed.")
        sys.exit(1)

    print(f"Parsed {parsed} matches from the last {last_k} .glog files.\n")
    print(f"{'Bot':10} {'W':>4} {'L':>4} {'D':>4} {'BR':>12} {'Avg/Match':>12}")
    print("-" * 52)
    for bot, st in stats.items():
        m = st["matches"] if st["matches"] else 1
        avg = st["bankroll"] / float(m)
        print(
            f"{bot:10} {st['wins']:4d} {st['losses']:4d} {st['draws']:4d} "
            f"{st['bankroll']:12d} {avg:12.1f}"
        )


if __name__ == "__main__":
    main()

