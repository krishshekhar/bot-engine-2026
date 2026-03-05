import os
import re
import subprocess

from config import PYTHON_CMD


FINAL_RE = re.compile(r"^Final,\s+(.+?)\s+\((-?\d+)\),\s+(.+?)\s+\((-?\d+)\)\s*$")


def run_matches(num_matches: int = 40) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    before = {f for f in os.listdir(logs_dir) if f.endswith(".glog")}

    # Run engine.py num_matches times with current config (bestbot vs bestbotv2)
    for i in range(num_matches):
        print(f"=== Running match {i + 1}/{num_matches} ===")
        subprocess.run(
            [PYTHON_CMD, "engine.py"],
            cwd=base_dir,
            check=True,
        )

    after = {f for f in os.listdir(logs_dir) if f.endswith(".glog")}
    new_files = sorted(after - before)

    print(f"\nCollected {len(new_files)} new .glog files.")
    if len(new_files) != num_matches:
        print("Warning: number of new logs does not match num_matches; continuing with what was found.")

    stats = {}

    parsed = 0
    for fname in new_files:
        path = os.path.join(logs_dir, fname)
        final_line = None
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("Final,"):
                        final_line = line.strip()
        except OSError:
            continue

        if not final_line:
            continue

        m = FINAL_RE.match(final_line)
        if not m:
            continue

        bot1, s1, bot2, s2 = m.groups()
        s1 = int(s1)
        s2 = int(s2)

        # Only aggregate if this match is between the two current config bots.
        pair = {bot1, bot2}
        if len(pair) != 2:
            continue

        for name, score in ((bot1, s1), (bot2, s2)):
            if name not in stats:
                stats[name] = {"wins": 0, "losses": 0, "draws": 0, "bankroll": 0}

        parsed += 1
        stats[bot1]["bankroll"] += s1
        stats[bot2]["bankroll"] += s2

        if s1 > s2:
            stats[bot1]["wins"] += 1
            stats[bot2]["losses"] += 1
        elif s2 > s1:
            stats[bot2]["wins"] += 1
            stats[bot1]["losses"] += 1
        else:
            stats[bot1]["draws"] += 1
            stats[bot2]["draws"] += 1

    print(f"\nParsed {parsed} matches from new logs.")
    if parsed == 0:
        return

    for name, st in stats.items():
        avg = st["bankroll"] / float(parsed)
        print(
            f"{name}: W={st['wins']} L={st['losses']} D={st['draws']} "
            f"Bankroll={st['bankroll']} Avg/Match={avg:.1f}"
        )


if __name__ == "__main__":
    run_matches(40)

