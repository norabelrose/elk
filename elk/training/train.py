"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from ..metrics import evaluate_preds, to_one_hot
from ..run import Run
from ..training.supervised import train_supervised
from ..utils.typing import assert_type
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import ReporterConfig


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    net: ReporterConfig = subgroups(
        {"ccs": CcsReporterConfig, "eigen": EigenReporterConfig}, default="eigen"
    )
    """Config for building the reporter network."""

    supervised: Literal["none", "single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        # Save the reporter config separately in the reporter directory
        # for convenient loading of reporters later.
        save(self.net, reporter_dir / "cfg.yaml", save_dc_types=True)

        return reporter_dir, lr_dir

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
    ) -> dict[str, pd.DataFrame]:
        """Train a single reporter on a single layer."""

        self.make_reproducible(seed=self.net.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        (first_train_h, train_gt, _), *rest = train_dict.values()
        d = first_train_h.shape[-1]
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        if isinstance(self.net, CcsReporterConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(self.net, d, device=device)
            train_loss = reporter.fit(first_train_h, train_gt)

            (val_h, val_gt, _) = next(iter(val_dict.values()))
            x0, x1 = first_train_h.unbind(2)
            val_x0, val_x1 = val_h.unbind(2)
            pseudo_auroc = reporter.check_separability(
                train_pair=(reporter.neg_norm(x0), reporter.pos_norm(x1)),
                val_pair=(reporter.neg_norm(val_x0), reporter.pos_norm(val_x1)),
            )

            (_, v, k, _) = first_train_h.shape
            reporter.platt_scale(
                to_one_hot(repeat(train_gt, "n -> (n v)", v=v), k).flatten(),
                rearrange(first_train_h, "n v k d -> (n v k) d"),
            )

        elif isinstance(self.net, EigenReporterConfig):
            # We set num_classes to None to enable training on datasets with different
            # numbers of classes. Under the hood, this causes the covariance statistics
            # to be simply averaged across all batches passed to update().
            reporter = EigenReporter(self.net, d, num_classes=None, device=device)

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_gt, _) in train_dict.items():
                (_, v, k, _) = train_h.shape

                # Datasets can have different numbers of variants and different numbers
                # of classes, so we need to flatten them here before concatenating
                hidden_list.append(rearrange(train_h, "n v k d -> (n v k) d"))
                label_list.append(
                    to_one_hot(repeat(train_gt, "n -> (n v)", v=v), k).flatten()
                )
                reporter.update(train_h)

            pseudo_auroc = None
            train_loss = reporter.fit_streaming()
            reporter.platt_scale(
                torch.cat(label_list),
                torch.cat(hidden_list),
            )
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.net)}")

        # Save reporter checkpoint to disk
        reporter.save(reporter_dir / f"layer_{layer}.pt")

        # Fit supervised logistic regression model
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        row_bufs = defaultdict(list)
        for ds_name, (val_h, val_gt, val_lm_preds) in val_dict.items():
            meta = {"dataset": ds_name, "layer": layer}

            val_result = evaluate_preds(val_gt, reporter(val_h))
            row_bufs["eval"].append(
                {
                    **meta,
                    "pseudo_auroc": pseudo_auroc,
                    "train_loss": train_loss,
                    **val_result.to_dict(),
                }
            )

            if val_lm_preds is not None:
                lm_result = evaluate_preds(val_gt, val_lm_preds)
                row_bufs["lm_eval"].append({**meta, **lm_result.to_dict()})

            for i, model in enumerate(lr_models):
                lr_result = evaluate_preds(val_gt, model(val_h))
                row_bufs["lr_eval"].append(
                    {"inlp_iter": i, **meta, **lr_result.to_dict()}
                )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}
