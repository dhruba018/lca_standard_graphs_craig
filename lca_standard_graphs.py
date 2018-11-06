import pandas as pd
import numpy as np
import seaborn as sns
import pdb
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

def build_comparison_table(df1, df2, df1_name, df2_name, level_name='Scenarios', fillna=None):

    a = df1.copy(deep=True)
    a.insert(0, level_name, df1_name, allow_duplicates=True)

    b = df2.copy(deep=True)
    b.insert(0, level_name, df2_name, allow_duplicates=True)

    comp = pd.concat([a, b], sort=False)
    comp.sort_index(inplace=True)
    comp.set_index(level_name, append=True, inplace=True)

    if fillna is not None:
        comp.fillna(fillna, inplace=True)

    return comp



def plot_grouped_stackedbar_comparison(df, ix_categories, ix_entities_compared, norm='max', orient='h', palette_def=('pastel', 'deep', 'dark')):
    if norm is not None:
        if norm == 'max':
            df = _normalize_impacts(df,
                                    ix_norm_level=ix_categories,
                                    ix_ref_level=ix_entities_compared)
        else:
            df = _normalize_impacts(df,
                                    ix_norm_level=ix_categories,
                                    ix_ref_level=ix_entities_compared,
                                    ref_name=norm)

    df = _calc_cumsum_tidy_df(df, var_name='stages')
    ax, fig = _plot_grouped_stackedbars_from_tidycumsum(df,
                                                        categories=ix_categories,
                                                        stacked_portions='stages',
                                                        values='value',
                                                        entities_compared=ix_entities_compared,
                                                        orient=orient,
                                                        palette_def=palette_def)
    return ax, fig



def _normalize_impacts(df, ix_norm_level, ix_ref_level, ref_name=None):

    if ref_name:
        ref_imp = df.xs(ref_name, axis=0, level=ix_ref_level).sum(axis=1)
    else:

        grouped_levels = list({i for i in df.index.names} - {ix_norm_level})
        ref_imp = df.sum(axis=1).reset_index(grouped_levels, drop=True).groupby(ix_norm_level).max()
    ref_imp = ref_imp.reindex(df.index, level=ix_norm_level)

    df = df.divide(ref_imp, axis=0) * 100

    return df


def _calc_cumsum_tidy_df(df, var_name='stages'):

    # Turn each row in to a sequential cummulative sum
    df = df.cumsum(axis=1)

    # Save names of indexes, and reset
    ix_names = df.index.names
    df = df.reset_index()

    # Turn into a tidy format
    df = df.melt(id_vars=ix_names, var_name=var_name)

    return df

def _plot_grouped_stackedbars_from_tidycumsum(cumsum_df, categories, stacked_portions, values,
        entities_compared, orient='h', palette_def=('pastel', 'deep', 'dark'), ci=95): 


    # Initializations
    fig = plt.figure(facecolor='white')
    ax = plt.subplot(111)
    legend_elements = []

    # Determine number of colors
    n_entities_compared = len(set(cumsum_df[entities_compared]))
    n_stacked_portions = len(set(cumsum_df[stacked_portions]))
    n_palettes = len(palette_def)

    # Choose colors & palette accordingly
    colors = sns.color_palette(palette_def[-1], n_colors=n_stacked_portions)
    palettes=[]
    if n_entities_compared <= n_palettes:
        for i in range(n_palettes - n_entities_compared, n_palettes):
            palettes += [sns.color_palette(palette_def[i], n_colors=n_stacked_portions)]
    else:
        palettes = None


    # Swap x and y axes, for vertical or horizontal bars
    if orient == 'h':
        x = values
        y = categories
    else:
        x = categories
        y = values


    for i, j in enumerate(cumsum_df[stacked_portions].unique()):

        bo = cumsum_df[stacked_portions] == j
        g = cumsum_df.loc[bo]

        if n_entities_compared > n_palettes:
            color = colors[i]
            palette = None
        else:
            color = None
            palette = [p[i] for p in palettes]

        if i == n_entities_compared:
            ci = ci
        else:
            ci = None

        ax = sns.barplot(data=g,
                 x=x,
                 y=y,
                 hue=entities_compared,
                 color=color,
                 palette=palette,
                 zorder=n_stacked_portions - i,
                 edgecolor="k",
                 ci=ci)

        if i == 0:
            legend_elements = generate_legend(g[entities_compared].unique())
        try:
            legend_elements += [Patch(facecolor=palette[-1], label=j),]
        except TypeError:
            legend_elements += [Patch(facecolor=color, label=j),]


    # Remove context-dependent stuff
    ax.legend_.remove() # remove the redundant legends
    ax.xaxis.set_label_text('')
    ax.yaxis.set_label_text('')


    legend_elements.reverse()
    plt.legend(handles=legend_elements,
              bbox_to_anchor=(1, 0.5),
              bbox_transform=fig.transFigure,
              # ncol=3,
              loc="center left")

    return ax, fig


def generate_legend(entities):
    legend_elements = []


    if len(entities) > 3:
        legend_elements += [Patch(edgecolor='black',
                                  facecolor='white',
                                  label="Gradient: {} (lightest), {}, {} (darkest)".format(
                                      entities[0],
                                      ', '.join(entities[1: -1]),
                                      entities[-1]))]

    else:

        legend_elements += [Patch(edgecolor='black',
                                  facecolor='white',
                                  label='light colors: ' + entities[0])]
        if len(entities) == 3:
            legend_elements += [Patch(edgecolor='black', 
                                      facecolor='white', 
                                      label='medium colors: ' + entities[1])]

        legend_elements += [Patch(edgecolor='black', 
                                  facecolor='white', 
                                  label='dark colors: ' + entities[-1])]
        legend_elements.reverse()
    return legend_elements


# It might be pertinent to define and lighten/darken colormaps ourselves:
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html

