import shutil
from dataclasses import dataclass
from pathlib import Path

import rich
from rich.panel import Panel
from simple_parsing import field

from ..files import sweeps_dir
from .visualize import visualize_sweep


def pretty_error(msg):
    rich.print(Panel(f"[red]Error[/red]: {msg}"))


@dataclass
class Plot:
    sweeps: list[Path] = field(positional=True, default_factory=list)
    overwrite: bool = False

    def execute(self):
        sweeps_root_dir = sweeps_dir()
        if not self.sweeps:
            self.sweeps = [
                max(sweeps_root_dir.iterdir(), key=lambda f: f.stat().st_ctime)
            ]
        else:
            self.sweeps = [sweeps_root_dir / sweep for sweep in self.sweeps]

        if self.sweeps and not (sweeps_root_dir / self.sweeps[0]).exists():
            pretty_error(
                f"No sweep with name {{{self.sweeps[0]}}} found in {sweeps_root_dir}"
            )
        elif len(self.sweeps) > 1:
            # TODO support more than one sweep
            pretty_error(
                f"""{len(self.sweeps)} paths specified.
                Only one sweep is supported at this time."""
            )
        elif (self.sweeps[0] / "viz").exists() and not self.overwrite:
            pretty_error(
                f"[blue]{self.sweeps[0] / 'viz'}[/blue] already exists. "
                f"Use --overwrite to overwrite."
            )
        else:
            if self.overwrite:
                shutil.rmtree(self.sweeps[0] / "viz")
            visualize_sweep(self.sweeps[0])
