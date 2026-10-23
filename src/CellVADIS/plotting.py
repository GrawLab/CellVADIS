"""Visualization helpers for CellVADIS fate outputs."""

import os

import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


def plot_tissue_transition_proportions(
    adata,
    output_key="stitchOT",
    colors=None,
    save_path=None,
    figsize=(5, 5),
    show=True,
    return_proportions=False,
    return_fig=False,
):
    """Plot the overall fraction of source cells assigned to each target tissue.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the fate result under ``adata.uns[output_key]``.
    output_key : str, default="stitchOT"
        Key in ``adata.uns`` used to access the fate matrix and target labels.
    colors : list of str or None, optional
        Colors for the pie-slice labels. If omitted, a default Matplotlib palette is used.
    save_path : str or None, optional
        File path used to save the figure.
    figsize : tuple, default=(5, 5)
        Figure size in inches.
    show : bool, default=True
        If True, display the figure.
    return_proportions : bool, default=False
        If True, return the transition proportions in addition to plotting.
    return_fig : bool, default=False
        If True, return the Matplotlib figure object.

    Returns
    -------
    np.ndarray or matplotlib.figure.Figure or None
        Transition proportions when ``return_proportions=True``; otherwise the figure
        when ``return_fig=True``; otherwise None.
    """
    fate_matrix = adata.uns[output_key]["fate_matrix"]
    transition_proportions = np.sum(fate_matrix, axis=0) / np.sum(fate_matrix)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(transition_proportions)))
        if "Unobserved" in adata.uns[output_key]["target_names"]:
            colors[-1] = "lightgray"

    ax.pie(
        transition_proportions,
        labels=adata.uns[output_key]["target_names"],
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
    )

    if return_fig:
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        return fig

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    elif show:
        plt.show()

    adata.uns[output_key]["transition_proportions"] = transition_proportions
    if return_proportions:
        return transition_proportions
    return None



def plot_dominant_fates(
    adata,
    output_key="stitchOT",
    colors=None,
    show=True,
    save_path=None,
    return_fig=False,
    **kwargs,
):
    """UMAP of source cells colored by their dominant inferred fate.

    Parameters
    ----------
    adata : AnnData
        Annotated dataset containing the source cells and their OT fate result.
    output_key : str, default="stitchOT"
        Key in ``adata.uns`` storing the inferred fate information.
    colors : list of str or None, optional
        Colors assigned to each target label. If omitted, a default palette is used.
    show : bool, default=True
        If True, display the figure immediately.
    save_path : str or None, optional
        Output path for the saved plot.
    return_fig : bool, default=False
        If True, return the Matplotlib figure object.
    **kwargs
        Additional keyword arguments forwarded to ``scanpy.pl.umap``.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure handle when ``return_fig=True``; otherwise None.
    """
    source_name = adata.uns[output_key]["source_name"]
    target_names = adata.uns[output_key]["target_names"]
    adata_source = adata[adata.obs["location"] == source_name].copy()
    sc.pp.neighbors(adata_source, n_neighbors=30)
    sc.tl.umap(adata_source, min_dist=0.9, spread=0.9, random_state=42)

    dominant_fate = np.argmax(adata.uns[output_key]["fate_matrix"], axis=1)
    dominant_fate = [target_names[i] for i in dominant_fate]
    adata_source.obs[f"{output_key}_dominant_fate"] = dominant_fate

    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(target_names)))
        if "Unobserved" in target_names:
            colors[-1] = "lightgray"

    color_dict = {target: color for target, color in zip(target_names, colors)}
    return sc.pl.umap(
        adata_source,
        color=f"{output_key}_dominant_fate",
        palette=color_dict,
        show=show,
        save=save_path,
        return_fig=return_fig,
        **kwargs,
    )





def plot_fate_probabilities_with_pie_charts(
    adata,
    output_key="stitchOT",
    entropy_fade=False,
    n_cells_to_plot=100,
    indices=None,
    s=500,
    outline=1.5,
    background=None,
    colors=None,
    ground_truth_locations=None,
    color_map_ground_truth=None,
    plot_legend=True,
    show=True,
    save_path=None,
    return_fig=False,
):
    """Plot fate probabilities as pie charts projected on a UMAP embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated dataset containing the source cells and their inferred fate matrix.
    output_key : str, default="stitchOT"
        Key in ``adata.uns`` that stores the fate matrix and target labels.
    entropy_fade : bool, default=False
        If True, fade the background points according to fate entropy.
    n_cells_to_plot : int, default=100
        Number of source cells to show as pie charts. A subset is sampled when the
        dataset is larger than this value.
    indices : array-like or None, optional
        Explicit indices of cells to plot instead of randomly sampling from the source set.
    s : int, default=500
        Marker size used for pie-chart plotting.
    outline : float, default=1.5
        Line width used for outlines when ground-truth labels are drawn.
    background : object or None, optional
        Background color passed through the plotting code path when provided.
    colors : list of str or None, optional
        Colors assigned to each fate target. If omitted, a Matplotlib palette is used.
    ground_truth_locations : array-like or None, optional
        Optional ground-truth target labels used to outline each pie chart.
    color_map_ground_truth : dict or None, optional
        Mapping from target integer labels to colors used for the pie-chart outlines.
    plot_legend : bool, default=True
        If True, show a legend for the target tissues.
    show : bool, default=True
        If True, display the figure.
    save_path : str or None, optional
        Output path for saving the plot.
    return_fig : bool, default=False
        If True, return the created matplotlib figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object when ``return_fig=True``; otherwise None.
    """
    fig, ax = plt.subplots()

    source_name = adata.uns[output_key]["source_name"]
    target_names = adata.uns[output_key]["target_names"]
    fate_matrix = adata.uns[output_key]["fate_matrix"]
    adata_source = adata[adata.obs["location"] == source_name].copy()
    sc.pp.neighbors(adata_source, n_neighbors=30)
    sc.tl.umap(adata_source, min_dist=0.9, spread=0.9, random_state=42)
    umap_coords = adata_source.obsm["X_umap"]

    if entropy_fade:
        entropies = -np.sum(fate_matrix * np.log2(fate_matrix), axis=1)

    umap_coords = (umap_coords - umap_coords.min()) / (umap_coords.max() - umap_coords.min())
    num_data = umap_coords.shape[0]

    if indices is not None:
        umap_coords_sampled = umap_coords[indices]
    elif n_cells_to_plot < num_data:
        #In future versions it would be nice to be able to sample cells in a way that they minimize overlap of pie charts
        umap_coords_sampled = umap_coords[np.random.choice(num_data, size=n_cells_to_plot, replace=False), :]
    else:
        umap_coords_sampled = umap_coords

    if colors is None:
        colors = plt.cm.tab10(range(fate_matrix.shape[1]))
        if "Unobserved" in target_names:
            colors[-1] = "lightgray"

    if entropy_fade:
        ax.scatter(
            umap_coords_sampled[:, 0],
            umap_coords_sampled[:, 1],
            color=[0.9, 0.9, 0.9],
            s=s,
        )

    for i, row in enumerate(umap_coords_sampled):
        x, y = row[0], row[1]
        probs = fate_matrix[i]
        radius = np.sqrt(s) / (2 * 3 * 72)
        ax.set_aspect("equal")

        wedges = [
            patches.Wedge(
                [x, y],
                radius,
                360 * np.sum(probs[:k]),
                360 * np.sum(probs[:k]) + 360 * probs[k],
                linewidth=0,
                color=colors[k],
            )
            for k in range(len(probs))
        ]

        for wedge in wedges:
            ax.add_patch(wedge)

        if ground_truth_locations is not None and color_map_ground_truth is not None:
            outline_color = color_map_ground_truth[int(ground_truth_locations[i])]
            outline = patches.Wedge(
                [x, y],
                radius,
                theta1=0,
                theta2=360,
                fill=False,
                edgecolor=outline_color,
                linewidth=outline,
            )
            ax.add_patch(outline)

    ax.set_ylim(np.min(umap_coords[:, 1]) - 0.05, np.max(umap_coords[:, 1]) + 0.05)
    ax.set_xlim(np.min(umap_coords[:, 0]) - 0.05, np.max(umap_coords[:, 0]) + 0.05)
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, target_names)
    ]
    if plot_legend:
        ax.legend(
            handles=legend_patches,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="Target Tissues",
            frameon=False,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    if return_fig:
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    elif show:
        plt.show()
    else:
        plt.close(fig)

    return None



def plot_avg_fate_per_celltype(
    adata,
    output_key="stitchOT",
    celltype_key="celltype",
    show=True,
    save_path=None,
    return_fig=False,
):
    """Plot the average fate probability by cell type as a heatmap.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the inferred fate probabilities.
    output_key : str, default="stitchOT"
        Key in ``adata.uns`` containing the fate result metadata.
    celltype_key : str, default="celltype"
        Column in ``adata.obs`` that stores the cell-type annotation.
    show : bool, default=True
        If True, display the heatmap.
    save_path : str or None, optional
        Output path used to save the figure.
    return_fig : bool, default=False
        When True, return the figure handle instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure when ``return_fig=True``; otherwise None.
    """
    fate_matrix = adata.uns[output_key]["fate_matrix"]
    target_names = adata.uns[output_key]["target_names"]
    avg_fate_per_celltype = {}
    adata_source = adata[adata.obs["location"] == adata.uns[output_key]["source_name"]].copy()

    for celltype in adata_source.obs[celltype_key].unique():
        celltype_mask = adata_source.obs[celltype_key] == celltype
        avg_fate_per_celltype[celltype] = np.mean(fate_matrix[celltype_mask], axis=0)

    avg_fates = pd.DataFrame(avg_fate_per_celltype, index=target_names)
    avg_fates.index = target_names

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        avg_fates,
        annot=False,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Avg fate probability", "shrink": 0.8},
    )
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel("Target tissues")
    plt.xlabel("Cell types")

    fig = plt.gcf()

    if return_fig:
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    elif show:
        plt.show()
    else:
        plt.close(fig)

    return None




def plot_homing_niches(
    adata,
    output_key="jointOT",
    colors=None,
    figsize=(12, 3),
    show=True,
    save_path=None,
    return_fig=False,
):
    """Project high-probability source cells onto each target tissue UMAP.

    Parameters
    ----------
    adata : AnnData
        Annotated dataset containing the OT result and tissue-level embeddings.
    output_key : str, default="jointOT"
        Key in ``adata.uns`` storing the OT plan and fate metadata.
    colors : list of str or None, optional
        Colors assigned to each tissue. If omitted, a default palette is used.
    figsize : tuple, default=(12, 3)
        Figure size for the panel of tissue-specific projections.
    show : bool, default=True
        If True, display the figure.
    save_path : str or None, optional
        Output file path used to save the figure.
    return_fig : bool, default=False
        If True, return the created figure handle.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object when ``return_fig=True``; otherwise None.
    """
    fate_matrix = adata.uns[output_key]["fate_matrix"]
    target_names = adata.uns[output_key]["target_names"]

    if "Unobserved" in target_names:
        unobserved_idx = target_names.index("Unobserved")
        fate_matrix = np.delete(fate_matrix, unobserved_idx, axis=1)
        target_names = [name for name in target_names if name != "Unobserved"]

    coupling_matrix = adata.uns[output_key]["norm_OT_plan"]
    fate_groups = adata.uns[output_key]["fate_groups"]

    if colors is None:
        tissue_colors = {
            tissue: color
            for tissue, color in zip(
                target_names,
                plt.cm.tab20(np.linspace(0, 1, len(target_names))),
            )
        }
    else:
        tissue_colors = {tissue: color for tissue, color in zip(target_names, colors)}

    fig, ax = plt.subplots(ncols=len(target_names), figsize=figsize, layout="constrained")

    for i, tissue in enumerate(target_names):
        adata_tissue = adata[adata.obs["location"] == tissue].copy()
        sc.pp.neighbors(adata_tissue, n_neighbors=30)
        sc.tl.umap(adata_tissue, min_dist=0.9, spread=0.9, random_state=42)
        umap_tissue = adata_tissue.obsm["X_umap"]

        current_ax = ax[i] if len(target_names) > 1 else ax
        current_ax.scatter(
            umap_tissue[:, 0],
            umap_tissue[:, 1],
            c="darkgray",
            s=0.2 * (120000 / len(umap_tissue)),
            alpha=0.6,
            edgecolor=None,
            linewidth=0,
            rasterized=True,
        )
        current_ax.set_xticks([])
        current_ax.set_yticks([])

        source_cells_fate_prob = fate_matrix[:, target_names.index(tissue)]
        source_cells_high_fate_coupling = coupling_matrix[source_cells_fate_prob > 1 / 3, :][:, fate_groups[i]]

        row_sums = source_cells_high_fate_coupling.sum(axis=1, keepdims=True)
        normalized_weights = np.divide(
            source_cells_high_fate_coupling,
            row_sums,
            out=np.zeros_like(source_cells_high_fate_coupling),
            where=row_sums > 0,
        )

        source_cells_high_fate_projected = normalized_weights @ umap_tissue

        current_ax.scatter(
            source_cells_high_fate_projected[:, 0],
            source_cells_high_fate_projected[:, 1],
            c=tissue_colors[tissue],
            s=0.5 * (120000 / len(umap_tissue)),
            label="source cells projected to " + tissue,
            alpha=0.95,
            edgecolor=None,
            linewidth=0,
            rasterized=True,
        )
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        current_ax.set_title(tissue, fontsize=8)
        current_ax.spines["top"].set_visible(False)
        current_ax.spines["right"].set_visible(False)
        current_ax.spines["bottom"].set_visible(False)
        current_ax.spines["left"].set_visible(False)

    if save_path is not None:
        if save_path.lower().endswith(".pdf"):
            save_target = save_path
        else:
            save_target = os.path.join(save_path, f"umap_{tissue}_projection_.pdf")
        plt.savefig(save_target, dpi=600, rasterized=True, bbox_inches="tight")

    if return_fig:
        if show:
            plt.show()
        return fig

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None