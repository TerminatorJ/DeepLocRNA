import matplotlib.pyplot as plt
import logomaker
import pandas as pd
def plot_motif(motif_array, sigma=['A', 'C', 'G', 'U'], ax=None, figsize=(16, 9/2), title=None, title_x=.50, title_y=1.05, ylab=None, start = 0, fontsize = 16):
    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=figsize)
    df = pd.DataFrame(motif_array, columns=sigma)
    logomaker.Logo(df, ax=axs) # , shade_below=.5 fade_below=.5, 
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    
    if title is not None:
        axs.set_title(title, x = title_x, y = title_y, fontsize=16)
        
    if ylab is not None:
        axs.set_ylabel(ylab, fontsize=16)
    
    #axs.set_yticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=26)
    # axs.set_xticks(ticks = [i for i in range(motif_array.shape[0])])
    # axs.set_xticklabels([str(i + 2854) for i in range(motif_array.shape[0])], rotation=45, ha='right')
    x_positions = range(0, motif_array.shape[0], 10)  # Define the x-tick positions every 100 numbers
    x_labels = [str(x+start) for x in x_positions]  # Create x-tick labels
    # print(x_labels)
    axs.set_xticks(ticks=x_positions)
    axs.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=fontsize)  
    axs.set_yticklabels(axs.get_yticklabels(), fontsize=fontsize)
    if ax is None:
        return fig
def plot_line(flat_att):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plt.plot([i for i in range(len(flat_att))], flat_att, linestyle='-', color = "blue")  # 'o' for markers, '-' for line style
    plt.title("Positions vs IG score")
    plt.xlabel("Positions", fontsize=16)
    plt.ylabel("IG score", fontsize=16)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.set_yticklabels(axs.get_yticklabels(), fontsize=16)
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=16)
    plt.tight_layout()
    return fig