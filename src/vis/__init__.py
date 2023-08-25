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


def visualize_embeddings(T: TEQUAL, labels) -> go.Figure:

    N = len(T.point_clouds)
    params = utils.read_parameter_file()

    architechtures = params.data_params.batch_sizes
    num_cols = len(architechtures)
    num_rows = int(N / num_cols)

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        column_titles=list(map(str, architechtures)),
        x_title="Batch Sizes",
        row_titles=None,
        y_title="Training Params",
        subplot_titles=(),
    )

    row = 1
    col = 1
    for i, embedding in enumerate(T.point_clouds):
        data = np.squeeze(embedding)
        df = pd.DataFrame(data, columns=["x", "y"])
        sample_size = int(len(df) / 5)
        # Sampling for plotly outputs
        sub_df = df.sample(n=sample_size)
        df["labels"] = labels[i]
        fig.add_trace(
            go.Scatter(
                x=sub_df["x"],
                y=sub_df["y"],
                mode="markers",
                opacity=0.4,
                marker=dict(
                    size=3,
                    color=df["labels"],
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
        width=2000,
        template="simple_white",
        showlegend=False,
        font=dict(color="black"),
        title="Hyperparameter Gridsearch",
    )

    fig.update_xaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    fig.update_yaxes(showticklabels=False, tickwidth=0, tickcolor="rgba(0,0,0,0)")
    return fig


def visualize_dendrogram(T: TEQUAL):
    def diagram_distance(_):
        return squareform(T.distance_relation)

    fig = ff.create_dendrogram(
        np.arange(len(T.diagrams)),
        distfun=diagram_distance,
        colorscale=px.colors.qualitative.Plotly,
        linkagefun=lambda x: linkage(x, T.linkage),
        color_threshold=T.epsilon,
    )
    fig.update_layout(
        width=1500,
        height=1000,
        template="simple_white",
        showlegend=False,
        font=dict(color="black", size=10),
        title="Persistence Based Clustering",
    )

    fig.update_xaxes(title=dict(text=f"Models"))
    fig.update_yaxes(title=dict(text=f"{T.metric} homological distance"))

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
