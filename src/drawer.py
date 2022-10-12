#%%
import plotly
from pathlib import Path
from typing import Literal, Tuple, TypeVar, Union
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.configs.paths import DATA_PATH

Charter = TypeVar("Charter")


class Charter:
    DATE_COL = "order_date"
    PRODUCT_COL = "product"
    sns.set_theme(style="whitegrid")

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Charter:
        path = Path(DATA_PATH / path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix == ".csv":
            return Charter(pd.read_csv(path))
        if path.suffix == ".xlsx":
            return Charter(pd.read_excel(path))
        if path.suffix == ".json":
            return Charter(pd.read_json(path))
        if path.suffix == ".parquet":
            return Charter(pd.read_parquet(path))
        if path.suffix == ".pickle" or path.suffix == ".pkl":
            return Charter(pd.read_pickle(path))
        if path.suffix == ".feather":
            return Charter(pd.read_feather(path))
        raise ValueError(f"Unsupported file type: {path}")

    def corr_heatmap(
        self,
    ) -> plt.axes:
        corr = self.df.corr()

        return ff.create_annotated_heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            annotation_text=corr.values.round(1),
            colorscale="Viridis",
        )

    def _with_rangeslider(
        self, fig: plotly.graph_objs._figure.Figure
    ) -> plotly.graph_objs._figure.Figure:
        fig = fig.update_layout(
            xaxis_rangeslider_visible=True,
            xaxis_type="category",
        )
        return fig

    def _with_rangeselector(
        self,
        fig: plotly.graph_objs._figure.Figure,
    ) -> plotly.graph_objs._figure.Figure:
        fig = fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
        )
        return fig

    def week_of_month(self, dt):
        first_day = dt.replace(day=1)
        dom = dt.day
        adj_dom = dom + first_day.weekday()
        return f"{dt.year}-{dt.month}-W{int(np.ceil(adj_dom/7))}"

    def transaction_timeseries_plotly(self) -> plotly.graph_objects.Figure:
        fig = self.transactions_per_interval_plotly(interval="day", chart_type="line")
        fig = self._with_rangeslider(fig)
        fig = self._with_rangeselector(fig)
        return fig

    def transactions_per_interval_plotly(
        self,
        interval: Literal["day", "week", "month", "quarter", "year"] = "month",
        chart_type: Literal["bar", "line"] = "bar",
    ) -> plotly.graph_objs._figure.Figure:
        interval_text = interval
        interval = interval.upper()[0]
        data = self.df[self.DATE_COL].dt.to_period(interval).value_counts().sort_index()

        if interval == "W":
            data.index = data.index.to_timestamp().map(self.week_of_month)

        chart_factory = getattr(px, chart_type)

        return chart_factory(
            pd.DataFrame(
                np.stack(
                    (
                        data.index.astype(str),
                        data.values.astype(int),
                    ),
                    axis=1,
                ),
                columns=[interval, "#transaction"],
            ),
            x=interval,
            y="#transaction",
            hover_data=[interval, "#transaction"],
            labels={interval: "Date", "#transaction": "Number of transactions"},
            title=f"Number of transactions per {interval_text}",
        )

    def convert_interval(
        self, interval: Literal["day", "week", "month", "quarter", "year"]
    ) -> Tuple[str, str]:
        interval_text = interval
        interval = interval.upper()[0]
        if interval not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError(f"Unsupported interval: {interval}")
        return interval_text, interval

    def num_of_transactions_per_interval(
        self,
        figsize: Tuple[int, int] = (20, 10),
        interval: Literal["day", "week", "month", "quarter", "year"] = "month",
        value_label: bool = True,
        draw_mean_std: bool = True,
        **kwargs,
    ) -> plt.axes:
        """num_of_transactions_per_interval

        Args:
            figsize (Tuple[int, int], optional): The figure size. Defaults to (20, 10).
            interval (Literal["day", "week", "month", "quarter", "year"], optional): The interval type. Defaults to 'month'.
            value_label (bool, optional): Display the value label on chart. Defaults to True.
            draw_mean_std (bool, optional): Draw the mean and +-standard deviation on chart. Defaults to True.

        Returns:
            plt.axes: The chart.

        Examples:
        >>> c = Charter.from_file('data.csv')
        >>> c.num_of_transactions_per_interval(figsize=(20, 20), interval='month')
        <AxesSubplot: title={'center': 'Number of transactions per month'}, xlabel='date', ylabel='#transactions'>
        """
        interval_text, interval = self.convert_interval(interval)

        plot_data = (
            self.df[self.DATE_COL].dt.to_period(interval).value_counts().sort_index()
        )

        x = plot_data.index
        y = plot_data.values

        _ = plt.figure(figsize=figsize)

        pal = sns.color_palette("Oranges_d", len(plot_data))

        rank = plot_data.argsort().argsort()

        ax = sns.barplot(
            x=x,
            y=y,
            palette=np.array(pal)[rank],
        )

        xticks = 5 if interval in ("D", "W") else 1

        ax.set_xticks(ax.get_xticks()[::xticks])
        ax.set_xticklabels(x[::xticks], rotation=45, ha="right")

        ax.set_title(f"Number of transactions per {interval_text}")
        ax.set_xlabel("date")
        ax.set_ylabel("#transactions")

        if value_label:
            _ = ax.bar_label(ax.containers[0], padding=3, fmt="%d")

        if draw_mean_std:
            _ = ax.axhline(y.mean(), color="b", linestyle="-")
            _ = ax.axhline(y.mean() + y.std(), color="r", linestyle="--")
            _ = ax.axhline(y.mean() - y.std(), color="g", linestyle="--")
            _ = ax.text(x=0, y=y.mean(), s=f"mean: {y.mean():.2f}", color="b")
            _ = ax.text(
                x=0,
                y=y.mean() + y.std(),
                s=f"+std: {y.mean() + y.std():.2f}",
                color="r",
            )
            _ = ax.text(
                x=0,
                y=y.mean() - y.std(),
                s=f"-std: {y.mean() - y.std():.2f}",
                color="g",
            )

        return ax

    def num_of_transactions_per_target(
        self,
        target: str,
        interval: Literal["day", "week", "month", "quarter", "year"] = "month",
        chart_type: Literal["bar", "line", "area", "hist", "pie", "box"] = "bar",
        figsize: Tuple[int, int] = (20, 10),
        k: int = 10,
        value_label: bool = True,
        stacked: bool = True,
        subplots: bool = False,
        layout: Tuple[int, int] = (2, 5),
        sharex: bool = True,
        sharey: bool = True,
        draw_mean_std: bool = True,
    ):
        """num_of_transactions_per_target Plot the number of transactions per target.

        Args:
            target (str): The target column.
            interval (Literal[day, week, month, quarter, year], optional): The interval for counting. Defaults to 'month'.
            chart_type (Literal[bar, line, area, hist, pie, box], optional): The chart type. Defaults to 'bar'.
            figsize (Tuple[int, int], optional): The figure size. Defaults to (20, 10).
            k (int, optional): Top k target to display. Defaults to 10.
            value_label (bool, optional): Display value labels for each value. Defaults to True.
            stacked (bool, optional): Stack the target value to display. Defaults to True.
            subplots (bool, optional): Splitted display with subplots for each target. Defaults to False.
            layout (Tuple[int, int], optional): Layout for subplots. Defaults to (2, 5).
            sharex (bool, optional): Share x-axis in subplots display. Defaults to True.
            sharey (bool, optional): Share y-axis in subplots display. Defaults to True.
            draw_mean_std (bool, optional): Display the mean +- standard deviation in chart. Defaults to True.

        Returns:
            _type_: _description_
        """
        assert target in self.df.columns, f"The target {target} is not in the data"

        if subplots:
            assert (layout[0] * layout[1]) >= (
                len(self.df[target].unique()) if k == -1 else k
            ), f"The layout is not enough to display all {target}"

        interval_text, interval = self.convert_interval(interval)

        target_cnt = self.df[target].value_counts(sort=True)

        if k != -1:
            topk_target_idx = target_cnt.index[:k]
            topk_target = self.df[np.in1d(self.df[target], topk_target_idx)]

            target_cnt = (
                topk_target.groupby(self.df[self.DATE_COL].dt.to_period(interval))[
                    target
                ]
                .value_counts()
                .unstack()
            )

        ax = target_cnt.plot(
            figsize=figsize,
            subplots=subplots,
            layout=layout,
            sharex=sharex,
            sharey=sharey,
            kind=chart_type,
            stacked=stacked,
            title=f"Number of transactions per {interval_text} for each {target}",
            xlabel="date",
            ylabel="#transactions",
            legend=True,
        )

        try:
            if subplots:
                for i in range(len(ax)):
                    for j in range(len(ax[i])):
                        _ = ax[i][j].set_title(f"{ax[i][j].get_title()}")
                if value_label:
                    for i in range(layout[0]):
                        for j in range(layout[1]):
                            _ = ax[i, j].bar_label(
                                ax[i, j].containers[0], padding=3, fmt="%d"
                            )
                if draw_mean_std:
                    for i in range(layout[0]):
                        for j in range(layout[1]):
                            y = target_cnt.iloc[:, j]
                            _ = ax[i, j].axhline(y.mean(), color="b", linestyle="-")
                            _ = ax[i, j].axhline(
                                y.mean() + y.std(), color="r", linestyle="--"
                            )
                            _ = ax[i, j].axhline(
                                y.mean() - y.std(), color="g", linestyle="--"
                            )
                            _ = ax[i, j].text(
                                x=0, y=y.mean(), s=f"mean: {y.mean():.2f}", color="b"
                            )
                            _ = ax[i, j].text(
                                x=0,
                                y=y.mean() + y.std(),
                                s=f"+std: {y.mean() + y.std():.2f}",
                                color="r",
                            )
                            _ = ax[i, j].text(
                                x=0,
                                y=y.mean() - y.std(),
                                s=f"-std: {y.mean() - y.std():.2f}",
                                color="g",
                            )

            else:
                if value_label and chart_type not in ["line"]:
                    for i in range(len(target_cnt.columns)):
                        _ = ax.bar_label(ax.containers[i], padding=3, fmt="%d")
                if draw_mean_std:
                    mean = target_cnt.mean(axis=1)
                    mean.index = mean.index.astype(str)
                    std = target_cnt.std(axis=1)
                    std.index = std.index.astype(str)
                    _ = ax.plot(mean, color="b", linestyle="-")
                    _ = ax.plot(mean + std, color="r", linestyle="--")
                    _ = ax.plot(mean - std, color="g", linestyle="--")
        except IndexError:
            ...

        return ax


if __name__ == "__main__":
    c = Charter.from_file("./carrefour.pkl")
    ax = c.num_of_transactions_per_target(
        target="city",
        k=3,
        chart_type="area",
        subplots=False,
        interval="m",
        stacked=False,
        sharex=False,
        sharey=False,
    )
    ax = c.num_of_transactions_per_interval()
    ax = c.transaction_timeseries_plotly()
    ax = c.corr_heatmap()
