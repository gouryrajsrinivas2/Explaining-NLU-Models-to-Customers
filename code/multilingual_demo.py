import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from os import path
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.manifold import TSNE


# https://www.sbert.net/docs/pretrained_models.html#choosing-the-right-model
transformer_model = 'distiluse-base-multilingual-cased-v2'
embedder = SentenceTransformer(transformer_model)


def generate_muse_embeddings(utterance_file_path: str = "data/snow/NLU Data/CSM/utterance_modified.txt",
                             embeddings_file_path: str = 'data/mUSE/utterance_embeddings.pt'):
    embeddings = None
    with open(utterance_file_path) as f:
        utterance_list = f.read().splitlines()
        print(utterance_list)
        if path.exists(embeddings_file_path):
            print(f"Loading Existing mUSE embeddings from Tensor {embeddings_file_path}")
            embeddings = torch.load(embeddings_file_path)
        else:
            print(f"Loading utterances from {utterance_file_path}")
            print(f"Generating mUSE Embeddings")
            embeddings = embedder.encode(utterance_list, convert_to_tensor=True)
            torch.save(embeddings, embeddings_file_path)

    return embeddings, utterance_list


def load_labels(labels_file_path: str = "data/snow/NLU Data/CSM/labels.txt"):
    labels_list = None
    with open(labels_file_path) as f:
        labels_list = f.read().splitlines()

    return labels_list


muse_embeddings, utterances = generate_muse_embeddings()
labels = load_labels()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='input-utterance', type='text', value=''),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state')
])

@app.callback(Output('output-state', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('input-utterance', 'value'))
def update_output(n_clicks, utterance):
    output_value = get_similar_neighbors(utterance, muse_embeddings)
    return 'Output: {}'.format(output_value)


def get_similar_neighbors(query: str, muse_embeddings):

    top_x_utterances = []

    if query != "":
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(6, len(utterances)-1)   # Check: why -1?
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        print("TSNE")
        muse_embeddings = torch.cat((muse_embeddings, query_embedding.unsqueeze(0)), 0)
        X = muse_embeddings.numpy()
        tsne_embeddings = TSNE(n_components=3).fit_transform(X)
        print(tsne_embeddings)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, muse_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            if idx < len(utterances):
                print(f"{utterances[idx]} - Score: {score:.4f} - {labels[idx]}")
                top_x_utterances.append(utterances[idx])
                top_x_utterances.append(labels[idx])

    return top_x_utterances


if __name__ == '__main__':
    app.run_server(debug=True, port=8055)