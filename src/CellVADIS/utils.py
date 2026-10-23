"""Utility helpers for fate aggregation and correlation analysis.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_fates(ot_matrix, fate_groups):
    """Aggregate OT mass into one value per source cell and fate group.

    Parameters
    ----------
    ot_matrix : np.ndarray
        OT matrix with shape (n_sources, n_targets). Each row describes the mass
        assigned from one source cell to all targets.
    fate_groups : sequence of sequence of int
        Grouping of target indices defining each fate, e.g. [[0, 1], [2, 3]].

    Returns
    -------
    np.ndarray
        Array of shape (n_sources, len(fate_groups)); each column contains the
        summed transport mass for one fate group.
    """
    n_source = ot_matrix.shape[0]
    n_fates = len(fate_groups)
    fate_matrix = np.zeros((n_source, n_fates))

    for i, group in enumerate(fate_groups):
        fate_matrix[:, i] = np.sum(ot_matrix[:, group], axis=1)

    return fate_matrix


def obs_key_wise_subsampling(
    adata,
    obs_keys,
    N_target,
    N_source,
    random_state=0,
    source_name=None,
    in_place=False,
):
    """Subsample observations to a fixed number of cells per annotation group.

    Parameters
    ----------
    adata : anndata.AnnData
        Input dataset whose observations are grouped by ``obs_keys``.
    obs_keys : str or list of str
        Column names in ``adata.obs`` used to define groups. For example,
        ["timepoint", "location"] creates groups for each time/location pair.
    N_target : int or None
        Desired number of cells to keep in each non-source group. If a group is
        smaller, all cells are kept. When set to None, no target-side downsampling
        is applied.
    N_source : int or None
        Desired number of cells to keep in the special source group identified by
        ``source_name``. When ``source_name`` is not provided, this argument is
        ignored. When set to None, no source-side downsampling is applied.
    random_state : int, default=0
        Seed used to make the subsampling reproducible.
    source_name : str or None, default=None
        Optional label used to identify the source group. When present, that group
        is subsampled to ``N_source`` instead of ``N_target``.
    in_place : bool, default=False
        If True, the dataset is modified in place; otherwise, a copy is returned.

    Returns
    -------
    anndata.AnnData
        A subsampled AnnData object. If ``in_place=True``, the same object is
        updated and returned.
    """
    if isinstance(obs_keys, str):
        obs_keys = [obs_keys]

    if N_target is not None:
        group_sizes = adata.obs.groupby(obs_keys).size()
        too_small = group_sizes[group_sizes < N_target]
        if len(too_small) > 0:
            warnings.warn(
                f"Subsample size too large; using minimum group size of {too_small.min()}.",
                UserWarning,
            )
            N_target = int(too_small.min())

    rng = np.random.default_rng(random_state)
    sampled_indices = []

    for _, df_group in adata.obs.groupby(obs_keys):
        if source_name in df_group[obs_keys[0]].values and source_name is not None:
            if N_source is None or len(df_group) < N_source:
                sampled_indices.append(df_group.index.to_numpy())
                continue
            sampled_indices.append(
                rng.choice(df_group.index.to_numpy(), size=N_source, replace=False)
            )
            continue

        if N_target is None or len(df_group) < N_target:
            sampled_indices.append(df_group.index.to_numpy())
            continue

        sampled_indices.append(
            rng.choice(df_group.index.to_numpy(), size=N_target, replace=False)
        )

    selection = np.concatenate(sampled_indices)
    if in_place:
        adata._inplace_subset_obs(selection)
        return adata
    return adata[selection].copy()






def corr_marker_with_fates(
    adata,
    marker_genes,
    obsm_key=None,
    output_key="stitchOT",
    return_as_df=True,
    sort_df_by="mean",
    location_key="location",
):
    """Correlate marker-gene expression with fate probabilities.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing the source cells and the CellVADIS result in
        ``adata.uns[output_key]``.
    marker_genes : sequence of str or None
        Genes to test. If None, all genes in the dataset are used. This can be memory
        intensive for large datasets.
    obsm_key : str or None, default=None
        Optional key in ``adata.obsm`` containing prepared expression values. If
        provided, those values are used instead of raw ``adata.X``.
    output_key : str, default="stitchOT"
        Key in ``adata.uns`` that stores the fate result dictionary.
    return_as_df : bool, default=True
        If True, return a pandas DataFrame of correlations. If False, only store the
        result in ``adata.varm``.
    sort_df_by : str, default="mean"
        Sorting mode for the returned DataFrame. Use "mean" to sort by the average
        absolute correlation across all fates, or pass a target label to sort by a
        specific fate column.
    location_key : str, default="location"
        Observation column used to identify the source subset when ``obsm_key`` is
        not provided. This makes the function work with datasets whose location label
        is stored under a different column name.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame indexed by marker genes when ``return_as_df=True``; otherwise,
        None. The correlation matrix is also stored in ``adata.varm["marker_fate_correlation"]``.
    """
    if marker_genes is None:
        marker_genes = adata.var_names

    fate_matrix = adata.uns[output_key]["fate_matrix"]
    source_name = adata.uns[output_key]["source_name"]
    results = np.zeros((len(marker_genes), fate_matrix.shape[1]))

    for i, gene in enumerate(marker_genes):
        if gene not in adata.var_names:
            warnings.warn(f"Gene {gene} not found in adata.var_names. Skipping.")
            continue

        gene_idx = adata.var_names.get_loc(gene)

        if obsm_key is not None:
            gene_expression = adata.obsm[obsm_key][:, gene_idx]
        else:
            source_mask = adata.obs[location_key] == source_name
            gene_expression = (
                adata[source_mask, gene].X.todense()
                if hasattr(adata.X, "todense")
                else adata[source_mask, gene].X
            )
            gene_expression = (
                np.asarray(gene_expression).flatten()
                if hasattr(gene_expression, "flatten")
                else gene_expression
            )

        for fate_idx in range(fate_matrix.shape[1]):
            fate_probs = fate_matrix[:, fate_idx]
            corr_coef, _ = spearmanr(gene_expression, fate_probs, nan_policy="omit")
            results[i, fate_idx] = corr_coef

    results_full = np.zeros((adata.shape[1], adata.uns[output_key]["fate_matrix"].shape[1]))
    for i, gene in enumerate(marker_genes):
        if gene in adata.var_names:
            gene_idx = adata.var_names.get_loc(gene)
            results_full[gene_idx, :] = results[i, :]

    adata.varm["marker_fate_correlation"] = results_full

    if return_as_df:
        results_df = pd.DataFrame(
            results,
            index=marker_genes,
            columns=adata.uns[output_key]["target_names"],
        )
        if sort_df_by == "mean":
            results_df = (
                results_df.assign(mean_correlation=np.abs(results_df.mean(axis=1)))
                .sort_values(by="mean_correlation", ascending=False)
                .drop(columns=["mean_correlation"])
            )
        elif sort_df_by in adata.uns[output_key]["target_names"]:
            results_df = (
                results_df.assign(abs_correlation=np.abs(results_df[sort_df_by]))
                .sort_values(by="abs_correlation", ascending=False)
                .drop(columns=["abs_correlation"])
            )
        return results_df

    return None



def corr_marker_with_target_marginals(
    adata,
    marker_genes,
    output_key="jointOT",
    obsm_key=None,
    return_as_df=True,
    sort_df_by="mean",
    location_key="location",
):
    """Correlate marker genes with target marginal abundance.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated dataset containing an OT result stored under ``output_key`` in
        ``adata.uns``.
    marker_genes : sequence of str or None
        Marker genes to evaluate. If None, all genes in the dataset are used.
    output_key : str, default="jointOT"
        Key in ``adata.uns`` containing the OT result dictionary.
    obsm_key : str or None, default=None
        Optional expression matrix in ``adata.obsm`` to use instead of ``adata.X``.
    return_as_df : bool, default=True
        If True, return a pandas DataFrame; otherwise only store the result in
        ``adata.varm``.
    sort_df_by : str, default="mean"
        Sorting mode for the returned DataFrame. Use "mean" for average absolute
        correlation across targets or pass a target label to sort by one target.
    location_key : str, default="location"
        Observation column used to define which cells belong to the target groups.
        This makes the function compatible with datasets that store the location label
        under a different column name.

    Returns
    -------
    pandas.DataFrame or None
        Correlation table for marker genes vs. target marginals when
        ``return_as_df=True``. Results are also stored in
        ``adata.varm["marker_target_marginal_correlation"]``.
    """
    if marker_genes is None:
        marker_genes = adata.var_names

    fate_matrix = adata.uns[output_key]["fate_matrix"]
    target_marginals = adata.uns[output_key]["OT_plan"].sum(axis=0) #change this to be renormalized against source cells with >1/3 fate towards this tissue
    fate_groups = adata.uns[output_key]["fate_groups"]
    target_names = adata.uns[output_key]["target_names"]

    if "Unobserved" in target_names:
        fate_matrix = np.delete(fate_matrix, -1, axis=1)
        target_names = target_names[:-1]

    results = np.zeros((len(marker_genes), fate_matrix.shape[1]))

    if obsm_key is not None:
        gene_expression_matrix = adata.obsm[obsm_key]
    else:
        gene_expression_matrix = adata.X.todense() if hasattr(adata.X, "todense") else adata.X
        if hasattr(gene_expression_matrix, "toarray"):
            gene_expression_matrix = gene_expression_matrix.todense().toarray()

    target_mask = adata.obs[location_key].isin(target_names)
    gene_expression_matrix = gene_expression_matrix[target_mask, :]

    for i, gene in enumerate(marker_genes):
        if gene not in adata.var_names:
            warnings.warn(f"Gene {gene} not found in adata.var_names. Skipping.")
            continue

        gene_idx = adata.var_names.get_loc(gene)

        for j, target_idx in enumerate(fate_groups):
            gene_expression = (
                gene_expression_matrix[target_idx, gene_idx]
                if len(gene_expression_matrix.shape) > 1
                else gene_expression_matrix
            )
            marginals = target_marginals[target_idx]
            corr_coef, _ = spearmanr(
                gene_expression,
                np.log10(marginals + 1e-10),
                nan_policy="omit",
            )
            results[i, j] = corr_coef

    results_full = np.zeros((adata.shape[1], len(target_names)))
    for i, gene in enumerate(marker_genes):
        if gene in adata.var_names:
            gene_idx = adata.var_names.get_loc(gene)
            results_full[gene_idx, :] = results[i, :]

    adata.varm["marker_target_marginal_correlation"] = results_full

    if return_as_df:
        results_df = pd.DataFrame(results, index=marker_genes, columns=target_names)
        if sort_df_by == "mean":
            results_df = (
                results_df.assign(mean_correlation=np.abs(results_df.mean(axis=1)))
                .sort_values(by="mean_correlation", ascending=False)
                .drop(columns=["mean_correlation"])
            )
        elif sort_df_by in target_names:
            results_df = (
                results_df.assign(abs_correlation=np.abs(results_df[sort_df_by]))
                .sort_values(by="abs_correlation", ascending=False)
                .drop(columns=["abs_correlation"])
            )
        return results_df

    return None