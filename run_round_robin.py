"""
Round robin: 20 games per pairing between bestbot, bestbotv2, bestbotv2.1, bestbotv2.2.
Pairings: (bestbot,v2), (bestbot,v2.1), (bestbot,v2.2), (v2,v2.1), (v2,v2.2), (v2.1,v2.2).
"""

import os
import re
import subprocess

from config import PYTHON_CMD

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
FINAL_RE = re.compile(r"^Final,\s+(.+?)\s+\((-?\d+)\),\s+(.+?)\s+\((-?\d+)\)\s*$")


def set_config(bot1_name: str, bot1_file: str, bot2_name: str, bot2_file: str) -> None:
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    content = re.sub(
        r"BOT_1_NAME\s*=\s*['\"].+?['\"]",
        f"BOT_1_NAME = '{bot1_name}'",
        content,
    )
    content = re.sub(
        r"BOT_1_FILE\s*=\s*['\"].+?['\"]",
        f"BOT_1_FILE = '{bot1_file}'",
        content,
    )
    content = re.sub(
        r"BOT_2_NAME\s*=\s*['\"].+?['\"]",
        f"BOT_2_NAME = '{bot2_name}'",
        content,
    )
    content = re.sub(
        r"BOT_2_FILE\s*=\s*['\"].+?['\"]",
        f"BOT_2_FILE = '{bot2_file}'",
        content,
    )
    with open(CONFIG_PATH, "w") as f:
        f.write(content)


def run_one_round_robin_pair(
    base_dir: str,
    logs_dir: str,
    bot1_name: str,
    bot1_file: str,
    bot2_name: str,
    bot2_file: str,
    num_matches: int,
    before_files: set,
) -> tuple:
    """Run num_matches for one pairing; return (updated before_files, list of (bot1, s1, bot2, s2))."""
    set_config(bot1_name, bot1_file, bot2_name, bot2_file)
    for i in range(num_matches):
        print(f"  Match {i + 1}/{num_matches}")
        subprocess.run(
            [PYTHON_CMD, "engine.py"],
            cwd=base_dir,
            check=True,
            capture_output=True,
        )
    after = {f for f in os.listdir(logs_dir) if f.endswith(".glog")}
    new_files = sorted(after - before_files)
    results = []
    for fname in new_files:
        path = os.path.join(logs_dir, fname)
        final_line = None
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("Final,"):
                        final_line = line.strip()
                        break
        except OSError:
            continue
        if not final_line:
            continue
        m = FINAL_RE.match(final_line)
        if not m:
            continue
        b1, s1, b2, s2 = m.groups()
        results.append((b1, int(s1), b2, int(s2)))
    return after, results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    num_matches = 20

    pairings = [
        ("bestbot", "./bestbot.py", "bestbotv2", "./bestbotv2.py"),
        ("bestbot", "./bestbot.py", "bestbotv2.1", "./bestbotv2.1.py"),
        ("bestbot", "./bestbot.py", "bestbotv2.2", "./bestbotv2.2.py"),
        ("bestbotv2", "./bestbotv2.py", "bestbotv2.1", "./bestbotv2.1.py"),
        ("bestbotv2", "./bestbotv2.py", "bestbotv2.2", "./bestbotv2.2.py"),
        ("bestbotv2.1", "./bestbotv2.1.py", "bestbotv2.2", "./bestbotv2.2.py"),
    ]

    before = {f for f in os.listdir(logs_dir) if f.endswith(".glog")}
    all_results = []

    for idx, (b1_name, b1_file, b2_name, b2_file) in enumerate(pairings):
        print(f"\n=== Round robin pairing {idx + 1}/6: {b1_name} vs {b2_name} ({num_matches} games) ===")
        before, pair_results = run_one_round_robin_pair(
            base_dir, logs_dir, b1_name, b1_file, b2_name, b2_file, num_matches, before
        )
        all_results.extend(pair_results)
        w1 = sum(1 for a, sa, b, sb in pair_results if sa > sb)
        w2 = sum(1 for a, sa, b, sb in pair_results if sb > sa)
        print(f"  Result: {b1_name} {w1} - {w2} {b2_name}")

    stats = {}
    for b1, s1, b2, s2 in all_results:
        for name, score in ((b1, s1), (b2, s2)):
            if name not in stats:
                stats[name] = {"wins": 0, "losses": 0, "draws": 0, "bankroll": 0}
        stats[b1]["bankroll"] += s1
        stats[b2]["bankroll"] += s2
        if s1 > s2:
            stats[b1]["wins"] += 1
            stats[b2]["losses"] += 1
        elif s2 > s1:
            stats[b2]["wins"] += 1
            stats[b1]["losses"] += 1
        else:
            stats[b1]["draws"] += 1
            stats[b2]["draws"] += 1

    total_games = len(all_results)
    print("\n" + "=" * 60)
    print("ROUND ROBIN SUMMARY (20 games per pairing, 6 pairings = 120 games)")
    print("=" * 60)
    print(f"{'Bot':14} {'W':>4} {'L':>4} {'D':>4} {'Bankroll':>12} {'Avg/Game':>10}")
    print("-" * 60)
    for name in ["bestbot", "bestbotv2", "bestbotv2.1", "bestbotv2.2"]:
        if name not in stats:
            continue
        st = stats[name]
        avg = st["bankroll"] / float(total_games) if total_games else 0
        print(
            f"{name:14} {st['wins']:4} {st['losses']:4} {st['draws']:4} "
            f"{st['bankroll']:12} {avg:10.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
