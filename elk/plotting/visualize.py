from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table


@dataclass
class SweepByDsMultiplot:
    """Class for generating line plots with multiple subplots for different datasets."""

    model_name: str

    def render(
        self,
        sweep: "SweepVisualization",
        with_transfer=False,
        ensembles=["full", "none"],
        write=False,
    ) -> go.Figure:
        """Render the multiplot visualization.

        Args:
            sweep: The SweepVisualization instance containing the data.
            with_transfer: Flag indicating whether to include transfer eval data.
            ensembles: Filter for which ensembing options to include.
            write: Flag indicating whether to write the visualization to disk.

        Returns:
            The generated Plotly figure.
        """
        df = sweep.df[sweep.df["model_name"] == self.model_name]
        unique_datasets = df["eval_dataset"].unique()
        num_datasets = len(unique_datasets)
        num_rows = (num_datasets + 2) // 3

        fig = make_subplots(
            rows=num_rows,
            cols=3,
            subplot_titles=unique_datasets,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.1,
            x_title="Layer",
            y_title="AUROC",
        )
        color_map = dict(zip(ensembles, qualitative.Plotly))

        for ensemble in ensembles:
            ensemble_data: pd.DataFrame = df[df["ensembling"] == ensemble]
            if with_transfer:  # TODO write tests
                ensemble_data = ensemble_data.groupby(
                    ["eval_dataset", "layer", "ensembling"], as_index=False
                ).agg({"auroc_estimate": "mean"})
            else:
                ensemble_data = ensemble_data[
                    ensemble_data["eval_dataset"] == ensemble_data["train_dataset"]
                ]

            for i, dataset_name in enumerate(unique_datasets, start=1):
                dataset_data = ensemble_data[
                    ensemble_data["eval_dataset"] == dataset_name
                ]
                dataset_data = dataset_data.sort_values(by="layer")
                # Floor division by 3 is to determine the row num,
                # The + 1 is added to convert the 0-based index to a 1-based index for
                # Plotly's subplot numbering. Similar for col, except that the column
                # position is determined by modulo division by 3.
                row, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1
                fig.add_trace(
                    go.Scatter(
                        x=dataset_data["layer"],
                        y=dataset_data["auroc_estimate"],
                        mode="lines",
                        name=ensemble,
                        showlegend=(
                            False if dataset_name != unique_datasets[0] else True
                        ),
                        line=dict(color=color_map[ensemble]),
                    ),
                    row=row,
                    col=col,
                ).update_yaxes(
                    range=[0.4, 1.1],  # Between 0.5 and 1.0 but with a bit of buffer
                    row=row,
                    col=col,
                )

        fig.update_layout(
            legend=dict(
                title="Ensembling",
            ),
            title=f"AUROC Trend: {self.model_name}",
        )
        if write:
            fig.write_image(
                file=sweep.path / f"{self.model_name}-line-ds-multiplot.png",
                scale=2,
            )
            fig.write_html(
                file=sweep.path / f"{self.model_name}-line-ds-multiplot.html"
            )

        return fig


@dataclass
class TransferEvalHeatmap:
    """Class for generating heatmaps for transfer evaluation results."""

    layer: int
    score_type: str = "auroc_estimate"
    ensembling: str = "full"

    def render(self, df: pd.DataFrame) -> go.Figure:
        """Render the heatmap visualization.

        Args:
            df: The DataFrame containing the transfer evaluation data.

        Returns:
            The generated Plotly figure.
        """
        model_name = df["eval_dataset"].iloc[0]  # infer model name
        # TODO: validate
        pivot = pd.pivot_table(
            df, values=self.score_type, index="eval_dataset", columns="train_dataset"
        )

        fig = px.imshow(pivot, color_continuous_scale="Viridis", text_auto=True)

        fig.update_layout(
            xaxis_title="Train Dataset",
            yaxis_title="Transfer Dataset",
            title=f"AUROC Score Heatmap: {model_name} | Layer {self.layer}",
        )

        return fig


@dataclass
class TransferEvalTrend:
    """Class for generating line plots for the trend of AUROC scores in transfer
    evaluation."""

    dataset_names: list[str] | None
    score_type: str = "auroc_estimate"

    def render(self, df: pd.DataFrame) -> go.Figure:
        """Render the trend plot visualization.

        Args:
            df: The DataFrame containing the transfer evaluation data.

        Returns:
            The generated Plotly figure.
        """
        model_name = df["model_name"].iloc[0]
        if self.dataset_names is not None:
            df = self._filter_transfer_datasets(df, self.dataset_names)
        pivot = pd.pivot_table(
            df, values=self.score_type, index="layer", columns="eval_dataset"
        )

        fig = px.line(pivot, color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(
            xaxis_title="Layer",
            yaxis_title="AUROC Score",
            title=f"AUROC Score Trend: {model_name}",
        )

        avg = pivot.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=avg.index,
                y=avg.values,
                name="average",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        return fig

    @staticmethod
    def _filter_transfer_datasets(df, dataset_names):
        df = df[df["eval_dataset"].isin(dataset_names)]
        df = df[df["train_dataset"].isin(dataset_names)]
        return df


@dataclass
class ModelVisualization:
    """Class representing the visualization for a single model within a sweep."""

    df: pd.DataFrame
    model_name: str
    is_transfer: bool

    @classmethod
    def collect(cls, model_path: Path) -> "ModelVisualization":
        """Collect the evaluation data for a model.

        Args:
            model_path: The path to the model directory.
            sweep_name: The name of the sweep.

        Returns:
            The ModelVisualization instance containing the evaluation data.
        """
        df = pd.DataFrame()
        model_name = model_path.name
        is_transfer = False

        def get_train_dirs(model_path):
            # toplevel is either repo/dataset or dataset
            for toplevel in model_path.iterdir():
                if (toplevel / "cfg.yaml").exists():
                    yield toplevel
                else:
                    for train_dir in toplevel.iterdir():
                        yield train_dir

        for train_dir in get_train_dirs(model_path):
            eval_df = cls._read_eval_csv(train_dir, train_dir.name, train_dir.name)
            df = pd.concat([df, eval_df], ignore_index=True)
            transfer_dir = train_dir / "transfer"
            if transfer_dir.exists():
                is_transfer = True
                for eval_ds_dir in transfer_dir.iterdir():
                    eval_df = cls._read_eval_csv(
                        eval_ds_dir, eval_ds_dir.name, train_dir.name
                    )
                    df = pd.concat([df, eval_df], ignore_index=True)

        df["model_name"] = model_name
        return cls(df, model_name, is_transfer)

    def render_and_save(
        self,
        sweep: "SweepVisualization",
        dataset_names: list[str] | None = None,
        score_type="auroc_estimate",
        ensembling="full",
    ) -> None:
        """Render and save the visualization for the model.

        Args:
            sweep: The SweepVisualization instance.
            dataset_names: List of dataset names to include in the visualization.
            score_type: The type of score to display.
            ensembling: The ensembling option to consider.
        """
        df = self.df
        model_name = self.model_name
        layer_min, layer_max = df["layer"].min(), df["layer"].max()
        model_path = sweep.path / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        if self.is_transfer:
            for layer in range(layer_min, layer_max + 1):
                filtered = df[(df["layer"] == layer) & (df["ensembling"] == ensembling)]
                fig = TransferEvalHeatmap(
                    layer, score_type=score_type, ensembling=ensembling
                ).render(filtered)
                fig.write_image(file=model_path / f"{layer}.png")
        fig = TransferEvalTrend(dataset_names).render(df)
        fig.write_image(file=model_path / "transfer_eval_trend.png")

    @staticmethod
    def _read_eval_csv(path, eval_dataset, train_dataset):
        file = path / "lr_eval.csv"
        eval_df = pd.read_csv(file)
        eval_df["eval_dataset"] = eval_dataset
        eval_df["train_dataset"] = train_dataset
        return eval_df


@dataclass
class SweepVisualization:
    """Class representing the overall visualization for a sweep."""

    name: str
    df: pd.DataFrame
    path: Path
    datasets: list[str]
    models: dict[str, ModelVisualization]

    def model_names(self) -> list[str]:
        """Get the names of all models in the sweep.

        Returns:
            List of model names.
        """
        return list(self.models.keys())

    @staticmethod
    def _get_model_paths(sweep_path: Path) -> list[Path]:
        """Get the paths to the model directories in the sweep.

        Args:
            sweep_path: The path to the sweep directory.

        Returns:
            List of model directory paths.

        Raises:
            Exception: If the sweep has already been visualized.
        """
        folders = []
        for model_repo in sweep_path.iterdir():
            if not model_repo.is_dir():
                raise Exception(f"expected {model_repo} to be a directory")

            # TODO: Use a more robust heuristic
            if model_repo.name.startswith("gpt2"):
                folders += [model_repo]
            else:
                folders += [p for p in model_repo.iterdir() if p.is_dir()]
        return folders

    @classmethod
    def collect(cls, sweep_path: Path) -> "SweepVisualization":
        """Collect the evaluation data for a sweep.

        Args:
            sweep_path: The path to the sweep directory.

        Returns:
            The SweepVisualization instance containing the evaluation data.

        Raises:
            Exception: If the output directory already exists.
        """
        sweep_name = sweep_path.parts[-1]
        sweep_viz_path = sweep_path / "viz"
        if sweep_viz_path.exists():
            raise Exception("This sweep has already been visualized.")
        sweep_viz_path.mkdir(parents=True, exist_ok=True)

        model_paths = cls._get_model_paths(sweep_path)
        models = {
            model_path.name: ModelVisualization.collect(model_path)
            for model_path in model_paths
        }
        df = pd.concat([model.df for model in models.values()], ignore_index=True)
        datasets = list(df["eval_dataset"].unique())
        return cls(sweep_name, df, sweep_viz_path, datasets, models)

    def render_and_save(self):
        """Render and save all visualizations for the sweep."""
        for model in self.models.values():
            model.render_and_save(self)
        self.render_table(write=True)
        self.render_multiplots(write=True)

    def render_multiplots(self, write=False):
        """Render and optionally write the multiplot visualizations.

        Args:
            write: Flag indicating whether to write the visualizations to disk.
        """
        return [
            SweepByDsMultiplot(model).render(self, write=write, with_transfer=False)
            for model in self.models
        ]

    def render_table(
        self, score_type="auroc_estimate", display=True, write=False
    ) -> pd.DataFrame:
        """Render and optionally write the score table.

        Args:
            layer: The layer number (from last layer) to include in the score table.
            score_type: The type of score to include in the table.
            display: Flag indicating whether to display the table to stdout.
            write: Flag indicating whether to write the table to a file.

        Returns:
            The generated score table as a pandas DataFrame.
        """
        df = self.df[self.df["ensembling"] == "full"]

        # For each model, we use the layer whose mean AUROC is the highest
        best_layers, model_dfs = [], []
        for _, model_df in df.groupby("model_name"):
            best_layer = model_df.groupby("layer").auroc_estimate.mean().argmax()

            best_layers.append(best_layer)
            model_dfs.append(model_df[model_df["layer"] == best_layer])

        pivot_table = pd.concat(model_dfs).pivot_table(
            index="eval_dataset",
            columns="model_name",
            values=score_type,
            margins=True,
            margins_name="Mean",
        )
        if display:
            console = Console()
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )

            table.add_column("Dataset")
            for column in pivot_table.columns:
                table.add_column(str(column))

            for index, row in pivot_table.iterrows():
                table.add_row(str(index), *(f"{val:.3f}" for val in row))

            table.add_row("Best Layer", *map(str, best_layers), style="bold")
            console.print(table)

        if write:
            pivot_table.to_csv(f"score_table_{score_type}.csv")
        return pivot_table


def visualize_sweep(sweep_path: Path):
    """Visualize a sweep by generating and saving the visualizations.

    Args:
        sweep_path: The path to the sweep data directory.
    """
    SweepVisualization.collect(sweep_path).render_and_save()
