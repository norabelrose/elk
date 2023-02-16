from .files import elk_cache_dir
from datetime import datetime
from prettytable import PrettyTable
import json


def list_runs(args):
    path = elk_cache_dir()
    table = PrettyTable(["Date", "Model", "Dataset", "UUID"])

    # Trivial case
    if not path.exists():
        print(f"No cached runs found; {path} does not exist.")
        return

    # List all cached runs
    subfolders = sorted(
        ((p.stat().st_mtime, p) for p in path.iterdir() if p.is_dir()), reverse=True
    )
    for timestamp, run in subfolders:
        # Read the arguments used to run this experiment
        with open(run / "args.json", "r") as f:
            run_args = json.load(f)

        date = datetime.fromtimestamp(timestamp).strftime("%X %x")
        table.add_row(
            [date, run_args["model"], " ".join(run_args["dataset"]), run.name]
        )

    print(f"Cached runs in \033[1m{path}\033[0m:")  # bold
    print(table)
