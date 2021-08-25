import pathlib
from os import path

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import torch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
import tensorflow_hub as hub

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# https://www.sbert.net/docs/pretrained_models.html#choosing-the-right-model
transformer_model = 'distiluse-base-multilingual-cased-v2'
muse_embedder = SentenceTransformer(transformer_model)

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
guse_embedder = hub.load(module_url)

print("module %s loaded" % module_url)


def get_similar_neighbors(query: str, dataset, k_neighbors, embedding_mode):
    if dataset == "SNOW_CSM":
        labels_file_path = "data/snow/NLU Data/CSM/labels.txt"
        utterance_file_path = "data/snow/NLU Data/CSM/utterance_modified.txt"
        embeddings_file_path = f'data/snow/NLU Data/CSM/utterance_{embedding_mode}_embeddings.pt'
    elif dataset == "SNOW_HR":
        labels_file_path = "data/snow/NLU Data/HR/labels.txt"
        utterance_file_path = "data/snow/NLU Data/HR/utterance_modified.txt"
        embeddings_file_path = f'data/snow/NLU Data/HR/utterance_{embedding_mode}_embeddings.pt'
    elif dataset == "SNOW_ITSM":
        labels_file_path = "data/snow/NLU Data/ITSM/labels.txt"
        utterance_file_path = "data/snow/NLU Data/ITSM/utterance_modified.txt"
        embeddings_file_path = f'data/snow/NLU Data/ITSM/utterance_{embedding_mode}_embeddings.pt'

    # We assume embeddings exists on disk as Pytorch Tensors (created while loading the dataset)
    # for both GUSE and MUSE
    embeddings = torch.load(embeddings_file_path)

    print(f"Loaded {embeddings_file_path} - Shape: {embeddings.shape}")

    with open(utterance_file_path) as f:
        utterances = f.read().splitlines()
    with open(labels_file_path) as f:
        labels = f.read().splitlines()

    top_x_utterances_label = []
    top_x_utterances_score = []
    top_x_utterances_utt = []

    if query != "":
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(k_neighbors+1, len(utterances) - 1)  # Check: why -1?

        print("TSNE")
        if embedding_mode == "MUSE":
            query_embedding = muse_embedder.encode(query, convert_to_tensor=True)
            xuse_embeddings = torch.cat((embeddings, query_embedding.unsqueeze(0)), 0)
        elif embedding_mode == "GUSE":
            guse_embedding = guse_embedder([query])
            query_embedding = torch.tensor(guse_embedding.numpy())
            xuse_embeddings = torch.cat((embeddings, query_embedding), 0)

        # This has been removed since we're having issues refreshing the TSNE to include the text field utterance.
        # X = xuse_embeddings.numpy()
        # tsne_embed = TSNE(n_components=3).fit_transform(X)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, xuse_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            if idx < len(utterances):
                print(f"{utterances[idx]} - Score: {score:.4f} - {labels[idx]}")
                top_x_utterances_label.insert(0, labels[idx])
                top_x_utterances_score.insert(0, score)
                top_x_utterances_utt.insert(0, utterances[idx])

        top_x_utterances_label.pop(0)
        top_x_utterances_score.pop(0)
        top_x_utterances_utt.pop(0)

    return top_x_utterances_label, \
           top_x_utterances_score, \
           top_x_utterances_utt


def load_dataset(utterance_file_path: str = "data/snow/NLU Data/CSM/utterance_modified.txt",
                 embeddings_muse_file_path: str = None,
                 embeddings_guse_file_path: str = None,
                 labels_file_path: str = "data/snow/NLU Data/CSM/labels.txt"):

    with open(utterance_file_path) as f:
        utterance_list = f.read().splitlines()

        # LOADING-GENERATING GUSE EMBEDDINGS
        if embeddings_muse_file_path is not None:
            if path.exists(embeddings_muse_file_path):
                print(f"Loading Existing mUSE embeddings from Tensor {embeddings_muse_file_path}")
                embeddings = torch.load(embeddings_muse_file_path)
            else:
                print(f"Loading utterances from {utterance_file_path}")
                print(f"Generating mUSE Embeddings")
                embeddings = muse_embedder.encode(utterance_list, convert_to_tensor=True)
                torch.save(embeddings, embeddings_muse_file_path)

        # LOADING-GENERATING GUSE EMBEDDINGS
        if embeddings_guse_file_path is not None:
            if path.exists(embeddings_guse_file_path):
                print(f"Loading Existing guse embeddings from Tensor {embeddings_guse_file_path}")
                embeddings = torch.load(embeddings_guse_file_path)
            else:
                print(f"Loading utterances from {utterance_file_path}")
                guse_embedding = guse_embedder(utterance_list)
                embeddings = torch.tensor(guse_embedding.numpy())
                torch.save(embeddings, embeddings_guse_file_path)

    np_embeddings = embeddings.numpy()
    df = pd.DataFrame(data=np_embeddings)
    df.insert(loc=0, column='utt', value=utterance_list)

    with open(labels_file_path) as f:
        labels_list = f.read().splitlines()
        df['label'] = labels_list

    return df


def load_all_datasets(embedding_type="MUSE"):
    data_dictionary = None
    if embedding_type == "MUSE":
        data_dictionary = {
            "SNOW_CSM": load_dataset(utterance_file_path="data/snow/NLU Data/CSM/utterance_modified.txt",
                                     embeddings_muse_file_path=f'data/snow/NLU Data/CSM/utterance_MUSE_embeddings.pt',
                                     labels_file_path="data/snow/NLU Data/CSM/labels.txt"),
            "SNOW_HR": load_dataset(utterance_file_path="data/snow/NLU Data/HR/utterance_modified.txt",
                                    embeddings_muse_file_path=f'data/snow/NLU Data/HR/utterance_MUSE_embeddings.pt',
                                    labels_file_path="data/snow/NLU Data/HR/labels.txt"),
            "SNOW_ITSM": load_dataset(utterance_file_path="data/snow/NLU Data/ITSM/utterance_modified.txt",
                                      embeddings_muse_file_path=f'data/snow/NLU Data/ITSM/utterance_MUSE_embeddings.pt',
                                      labels_file_path="data/snow/NLU Data/ITSM/labels.txt"),
        }
    elif embedding_type == "GUSE":
        data_dictionary = {
            "SNOW_CSM": load_dataset(utterance_file_path="data/snow/NLU Data/CSM/utterance_modified.txt",
                                     embeddings_guse_file_path=f'data/snow/NLU Data/CSM/utterance_GUSE_embeddings.pt',
                                     labels_file_path="data/snow/NLU Data/CSM/labels.txt"),
            "SNOW_HR": load_dataset(utterance_file_path="data/snow/NLU Data/HR/utterance_modified.txt",
                                    embeddings_guse_file_path=f'data/snow/NLU Data/HR/utterance_GUSE_embeddings.pt',
                                    labels_file_path="data/snow/NLU Data/HR/labels.txt"),
            "SNOW_ITSM": load_dataset(utterance_file_path="data/snow/NLU Data/ITSM/utterance_modified.txt",
                                      embeddings_guse_file_path=f'data/snow/NLU Data/ITSM/utterance_GUSE_embeddings.pt',
                                      labels_file_path="data/snow/NLU Data/ITSM/labels.txt"),
        }
    return data_dictionary


muse_data_dict = load_all_datasets(embedding_type="MUSE")
guse_data_dict = load_all_datasets(embedding_type="GUSE")
WORD_EMBEDDINGS = ("SNOW_CSM", "SNOW_HR", "SNOW_ITSM")

with open(PATH.joinpath("muse_demo_intro.md"), "r") as file:
    demo_intro_md = file.read()

with open(PATH.joinpath("muse_demo_description.md"), "r") as file:
    demo_description_md = file.read()


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("hack.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Understanding NLU",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px", "background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            Card(
                                [
                                    html.B("Select dataset"),
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "ServiceNow CSM",
                                                "value": "SNOW_CSM",
                                            },
                                            {
                                                "label": "ServiceNow HR",
                                                "value": "SNOW_HR",
                                            },
                                            {
                                                "label": "ServiceNow ITSM",
                                                "value": "SNOW_ITSM",
                                            },
                                        ],
                                        placeholder="Select a dataset",
                                        value="SNOW_CSM",
                                    ),
                                    html.Br(),
                                    html.Div(
                                        id="div-wordemb-controls",
                                        style={"display": "none"},
                                        children=[
                                            NamedInlineRadioItems(
                                                name="Embedding Type",
                                                short="wordemb-display-mode",
                                                options=[
                                                    {
                                                        "label": " MUSE",
                                                        "value": "MUSE",
                                                    },
                                                    {
                                                        "label": " GUSE",
                                                        "value": "GUSE",
                                                    },
                                                ],
                                                val="GUSE",
                                            ),
                                            html.Br(),
                                            html.B("Select the utterance for neighbour analysis"),
                                            dcc.Dropdown(
                                                id="dropdown-word-selected",
                                                placeholder="Select utterance to display its neighbors",
                                                style={"background-color": "#f2f3f4"},
                                            ),
                                        ],
                                    ),
                                    html.Br(),
                                    html.B("Edit or enter utterance for neighbour analysis"),
                                    html.Div([
                                        dcc.Input(id='input-utterance', type='text', value='',
                                                  style={"padding": "auto", "width": "660px"}),
                                        html.Button(id='submit-button-state', n_clicks=0, children='Submit',
                                                    style={"padding": "auto", "margin-left": "20px",
                                                           "margin-top" : "10px"}),
                                        html.Div(id='output-state')
                                    ]),
                                    NamedSlider(
                                        name="# Nearest neighbors",
                                        short="nearest_neighbors",
                                        min=5,
                                        max=30,
                                        step=None,
                                        val=5,
                                        marks={i: str(i) for i in [5,10,15,20,25,30]},
                                    ),
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(id="div-plot-click-wordemb"),
                                ]
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


def demo_callbacks(app):
    # Scatter Plot of the t-SNE datasets
    def generate_figure_word_vec(embedding_df,
                                 layout,
                                 wordemb_display_mode,
                                 dataset,
                                 tsne_embeddings=None):

        try:
            data_dict = load_all_datasets(embedding_type=wordemb_display_mode)
            labels = list(data_dict[dataset]['label'])

            if tsne_embeddings is None:
                xuse_embeddings = embedding_df.drop(columns=['utt', 'label'])
                tsne_embeddings = TSNE(n_components=3).fit_transform(xuse_embeddings.to_numpy())
            else:  # we provide a tsne only when a query has been added to the dataset... (to verify!)
                labels.append("_")              # for the text field utterance...

            embedding_df["label"] = labels
            embedding_df['x'] = tsne_embeddings[:, 0]
            embedding_df['y'] = tsne_embeddings[:, 1]
            embedding_df['z'] = tsne_embeddings[:, 2]

            groups = embedding_df.groupby("label")
            data = []

            for idx, embedding_df in groups:
                scatter = go.Scatter3d(
                    name=idx,
                    x=embedding_df['x'],
                    y=embedding_df['y'],
                    z=embedding_df['z'],
                    text=embedding_df['utt'],
                    textposition="middle center",
                    showlegend=False,
                    mode="markers",
                    marker=dict(size=3, symbol="circle"),
                )
                data.append(scatter)

            figure = go.Figure(data=data, layout=layout)

            return figure
        except KeyError as error:
            print(error)
            raise PreventUpdate

    @app.callback(
        Output("div-wordemb-controls", "style"), [Input("dropdown-dataset", "value")]
    )
    def show_wordemb_controls(dataset):
        if dataset in WORD_EMBEDDINGS:
            return None
        else:
            return {"display": "none"}

    @app.callback(
        Output("dropdown-word-selected", "options"),
        [Input("dropdown-dataset", "value"),
        ],
    )
    def fill_dropdown_word_selection_options(dataset):
        if dataset in WORD_EMBEDDINGS:
            return [
                {"label": i, "value": i} for i in guse_data_dict[dataset].iloc[:, 0].tolist()
            ]
        else:
            return []

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),
            Input("radio-wordemb-display-mode", "value"),
        ],
    )
    def display_3d_scatter_plot(
            dataset,
            wordemb_display_mode,
    ):

        if wordemb_display_mode == "MUSE":
            embedding_df = muse_data_dict[dataset]
        elif wordemb_display_mode == "GUSE":
            embedding_df = guse_data_dict[dataset]

        # Plot layout
        axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )

        if dataset in WORD_EMBEDDINGS:
            figure = generate_figure_word_vec(
                embedding_df=embedding_df,
                layout=layout,
                wordemb_display_mode=wordemb_display_mode,
                dataset=dataset,
            )
        else:
            figure = go.Figure()

        return figure

    @app.callback(
        Output("div-plot-click-wordemb", "children"),
        Output("graph-3d-plot-tsne", "clickData"),
        Output('input-utterance', 'value'),
        Output('dropdown-word-selected', 'value'),
        [Input("graph-3d-plot-tsne", "clickData"),
         Input("dropdown-dataset", "value"),
         Input('submit-button-state', 'n_clicks'),
         Input("slider-nearest_neighbors", "value"),
         Input('dropdown-word-selected', 'value'),
         Input('radio-wordemb-display-mode', 'value'),
         State('input-utterance', 'value')],
    )
    def display_click_word_neighbors(clickData, dataset, n_clicks, k_neighbors, utter, embedding_mode, query):
        selected_word = None
        if utter != "" and utter is not None:
            selected_word = utter
        elif dataset in WORD_EMBEDDINGS and clickData:
            selected_word = clickData["points"][0]["text"]
        elif query != "" and query is not None:
            selected_word = query

        if selected_word is not None:
            print(f"Selected word: {selected_word}")

            try:
                top_x_utterances_label, \
                top_x_utterances_score, \
                top_x_utterances_utt = get_similar_neighbors(selected_word, dataset, k_neighbors+1, embedding_mode)

                layout = go.Layout(
                    title="Intents in nearest neighbours",
                    margin=go.layout.Margin(l=60, r=60, t=100, b=0),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                pie_fig = go.Figure(data=[go.Pie(labels=top_x_utterances_label, values=top_x_utterances_score,
                                                 textinfo='label+percent', insidetextorientation='radial')],
                                    layout=layout
                                    )

                trace = go.Bar(
                    x=top_x_utterances_score,
                    y=[str(x) + " --> " + str(y) for (x, y) in zip(top_x_utterances_utt, top_x_utterances_label)],
                    width=0.5,
                    orientation="h",
                    marker=dict(color="rgb(50, 102, 193)"),
                )

                layout = go.Layout(
                    title=f'{k_neighbors} nearest neighbors of "{selected_word}"',
                    xaxis=dict(title="Similarity Score"),
                    margin=go.layout.Margin(l=60, r=60, t=35, b=35),
                )
                fig = go.Figure(data=[trace], layout=layout)

                if k_neighbors > 5:
                    neighbors_bar_chart_height = str(k_neighbors*3)+"vh"
                else:
                    neighbors_bar_chart_height = "20vh"
                return ((dcc.Graph(
                    id="graph-bar-nearest-neighbors-word",
                    figure=fig,
                    style={"height": neighbors_bar_chart_height},
                    config={"displayModeBar": False},
                ), dcc.Graph(id="pie-chart", figure=pie_fig)), None, selected_word, None)
            except KeyError as error:
                raise PreventUpdate

        return (None, None, "", None)

    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-tsne", "clickData"), Input("dropdown-dataset", "value")],
    )
    def display_click_message(clickData, dataset):
        # Displays message shown when a point in the graph is clicked
        if dataset in WORD_EMBEDDINGS:
            if clickData:
                return None
            else:
                return "Click an utterance on the plot to see its top n neighbors."
