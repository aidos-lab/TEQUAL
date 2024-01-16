import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import utils
from topology.tequal import TEQUAL

pio.renderers.default = "chrome"


def visualize_embeddings(
    T: TEQUAL, key_val, filter_name, configs, labels=None, x_axis=None, y_axis=None
) -> go.Figure:
    N = len(T.point_clouds)

    params = utils.read_parameter_file()

    # Formatting Plot Axis Labels and titles

    subplot_titles = list(map(str, [cfg.meta.id for cfg in configs]))

    if x_axis and y_axis:
        title = f"{params.experiment}: {filter_name} = {key_val}"
        x_labels = list(map(str, params[x_axis[0]][x_axis[1]]))
        if y_axis == 1:
            y_labels = [str(key_val)]
            y_title = filter_name
        else:
            y_labels = list(map(str, params[y_axis[0]][y_axis[1]]))
            y_title = y_axis[1]
        x_title = x_axis[1]

        num_cols = len(x_labels)
        num_rows = len(y_labels)
    else:
        title = f"{params.experiment}"
        x_labels, y_labels = None, None
        num_cols = N
        num_rows = 1

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        column_titles=x_labels,
        row_titles=y_labels,
        x_title=x_title,
        y_title=y_title,
    )
    fig.print_grid()
    row = 1
    col = 1
    for i, embedding in enumerate(T.point_clouds):
        if embedding.shape[1] > 0:
            data = np.squeeze(embedding)
            df = pd.DataFrame(data, columns=["x", "y"])
            # Sampling for plotly outputs
            if labels is None:
                labels = np.zeros(shape=(N, len(df)))
            df["labels"] = labels[i].T[0]

            sub_df = df.sample(n=(int(len(df) / 10)))

            fig.add_trace(
                go.Scatter(
                    x=sub_df["x"],
                    y=sub_df["y"],
                    mode="markers",
                    opacity=0.4,
                    marker=dict(
                        size=1,
                        # color=sub_df["labels"],
                        colorscale="jet",
                    ),
                ),
                row=row,
                col=col,
            )
            row += 1
            if row == num_rows + 1:
                row = 1
                col += 1

    fig.update_layout(
        width=1000,
        height=500,
        template="simple_white",
        showlegend=False,
        font=dict(color="black"),
        title=None,
    )

    fig.update_annotations(font_size=8)
    fig.update_xaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
        tickfont=dict(family="CMU Sans Serif Demi Condensed", size=10),
    )
    fig.update_yaxes(
        showticklabels=False,
        tickwidth=0,
        tickcolor="rgba(0,0,0,0)",
        tickfont=dict(family="CMU Sans Serif Demi Condensed", size=10),
    )
    return fig


def visualize_dendrogram(
    T: TEQUAL,
    configs,
):
    labels = list(
        map(
            str,
            [f"{cfg.meta.id}" for cfg in configs],
        )
    )

    def diagram_distance(_):
        return squareform(T.distance_relation)

    fig = ff.create_dendrogram(
        X=np.arange(len(T.diagrams)),
        labels=labels,
        distfun=diagram_distance,
        colorscale=px.colors.qualitative.Plotly,
        linkagefun=lambda x: linkage(x, T.linkage),
        color_threshold=T.epsilon,
    )
    fig.update_layout(
        title="CIFAR-10 Compression",
        xaxis_title="Probes",
        yaxis_title="Landscape Distance",
        width=800,
        height=500,
        template="simple_white",
        showlegend=False,
        font=dict(color="black", size=10),
    )

    fig.update_xaxes(
        showticklabels=False,
    )
    fig.update_yaxes()

    ticktext = fig["layout"]["xaxis"]["ticktext"]
    tickvals = fig["layout"]["xaxis"]["tickvals"]
    colormap = {}
    reference = dict(zip(tickvals, ticktext))

    # Extracting Dendrogram Colors
    for trace in fig["data"]:
        if 0 in trace["y"]:
            xs = trace["x"][np.argwhere(trace["y"] == 0)]
            # This catch will ensure plots are generated, but empty plots may indicate you
            # have mismatched info between old runs and your params.json. Clean and rerun
            tickers = [reference[x[0]] if x[0] in reference.keys() else 0 for x in xs]
            for ticker in tickers:
                colormap[ticker] = trace["marker"]["color"]
    return fig, colormap


def save_visualizations_as_html(visualizations, output_file):
    """
    Saves a list of Plotly visualizations as an HTML file.

    Parameters:
    -----------
    visualizations : list
        A list of Plotly visualizations (plotly.graph_objects.Figure).
    output_file : str
        The path to the output HTML file.
    """

    # Create the HTML file and save the visualizations
    with open(output_file, "w") as f:
        f.write("<html>\n<head>\n</head>\n<body>\n")
        for i, viz in enumerate(visualizations):
            div_str = pio.to_html(viz, full_html=False, include_plotlyjs="cdn")
            f.write(f'<div id="visualization{i+1}">{div_str}</div>\n')
        f.write("</body>\n</html>")
