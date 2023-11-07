# %%
import numpy as np
import pandas as pd

# %%
import logomaker
from sklearn.cluster import DBSCAN
from umap import UMAP

# %%
def extract_motif(importances, k=7, norm=True):
    importances = np.array(importances)
    if len(importances.shape) != 2:
        raise ValueError(f'Expected matrix with ndim=2, got ndim={len(importances.shape)}.')

    best_i = -1
    best_score = -np.inf

    for i in range(0, importances.shape[0] - k + 1):
        importances_i = importances[i:(i+k), :]
        score = np.sum(importances_i)

        if score > best_score:
            best_i = i
            best_score = score

    motif = importances[best_i:(best_i + k), :]
    if norm:
        motif = motif / np.sum(motif)

    return motif

# %%
def extract_motifs(importances, k=7, norm=True):
    importances = np.array(importances)
    if len(importances.shape) != 3:
        raise ValueError(f'Expected matrix with ndim=3, got ndim={len(importances.shape)}.')
    
    return np.stack([extract_motif(x, k=k, norm=norm) for x in importances])

# %%
def motifs_flatten(motifs_3d):
    motifs_n, motifs_len, motifs_depth = motifs_3d.shape
    return motifs_3d.reshape((motifs_n, motifs_len*motifs_depth))

# %%
def motifs_embed(motifs_2d, embed_fn = lambda x: UMAP().fit_transform(x)):
    return embed_fn(motifs_2d)

# %%
def motifs_cluster(motifs_2d, clust_fn = lambda x: DBSCAN().fit(x).labels_):
    return clust_fn(motifs_2d)

# %%
def extract_meta_motifs(importances, k=7, embed_fn=None, clust_fn=None):
    if embed_fn is None:
        embed_fn = lambda x: UMAP().fit_transform(x)
    if clust_fn is None:
        clust_fn = lambda x: DBSCAN().fit(x).labels_

    motifs = extract_motifs(importances, k=k, norm=True)

    motifs_n, motifs_len, motifs_depth = motifs.shape

    motifs_flattened = motifs_flatten(motifs)
    motifs_embedding = motifs_embed(motifs_flattened, embed_fn)
    motifs_clustering = motifs_cluster(motifs_embedding, clust_fn)

    motifs_df = pd.DataFrame(motifs_flattened)
    motifs_df['cluster'] = motifs_clustering

    motifs_metaclusters = motifs_df.groupby(['cluster']).mean()
    motifs_metaclusters = np.array(motifs_metaclusters).reshape(len(motifs_metaclusters), motifs_len, motifs_depth)

    return motifs_metaclusters

# %%
def plot_motif(motif_2d, sigma=['A', 'C', 'G', 'T'], title=''):
    motif_2d_df = pd.DataFrame(motif_2d, columns=sigma)

    # create Logo object
    logo = logomaker.Logo(motif_2d_df, shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold')

    # style using Logo methods
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)

    # style using Axes methods
    logo.ax.set_ylabel("", labelpad=-1)
    logo.ax.set_title(title)

    return logo