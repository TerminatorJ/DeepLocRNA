
# %%
import matplotlib.pyplot as plt
import pandas as pd
import logomaker

# %%
def plot_motif(motif_array, sigma=['A', 'C', 'G', 'U'], ax=None, figsize=(16, 9/2), title=None, title_x=.50, title_y=1.05, ylab=None):
    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=figsize)
    df = pd.DataFrame(motif_array, columns=sigma)
    logomaker.Logo(df, font_name='Arial Rounded MT Bold', ax=axs) # , shade_below=.5 fade_below=.5, 
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    
    if title is not None:
        axs.set_title(title, x = title_x, y = title_y, fontsize=16)
        
    if ylab is not None:
        axs.set_ylabel(ylab, fontsize=16)
    
    #axs.set_yticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=26)
    axs.set_xticks(ticks = [])
    
    if ax is None:
        return fig