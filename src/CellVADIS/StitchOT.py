import warnings

import numpy as np
import ot

from .utils import get_fates


def StitchOT(
    adata,
    obsm_key,
    source_name,
    target_names,
    tau_b=0.5,
    tau_a=0.8,
    epsilon=0.005,
    rescale_OT_plans=True,
    use_true_fractions=False,
    assign_unsampled_probs=False,
    ignore_warnings=True,
    cost_scaling="median",
    cost_metric="sqeuclidean",
    float_type=np.float64,
    store_in_adata=True,
    output_key="stitchOT",
):
    """
    StitchOT method for cell fate inference.
    This method performs a series of OT solves from each source cell to each target location,
    and then stitches the results together into a single fate matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing gene expression and a ``location`` annotation in
        ``adata.obs``.
    obsm_key : str or None
        Name of the feature matrix to use from ``adata.obsm``. If ``None``, ``adata.X`` is
        used instead.
    source_name : str
        Label in ``adata.obs["location"]`` that identifies the source compartment.
    target_names : list of str
        Values in ``adata.obs["location"]`` identifying the target compartments.
    tau_b : float, default=0.995
        Unbalanced OT parameter for the target marginal. Values closer to 1 allow less mass creation/destruction.
    tau_a : float, default=0.5
        Unbalanced OT parameter for the source marginal.
    epsilon : float, default=0.005
        Entropic regularization strength used by the unbalanced Sinkhorn solver.
    rescale_OT_plans : bool, default=True
        If ``True``, rescale each OT plan by the fraction of source mass that is transported to each target location.
    use_true_fractions : array-like, default=None
        Optional per-target fractions used to initialize the target marginal. Must match ``target_names`` in length and will overwrite
        ``rescale_OT_plans`` to True if provided.
    assign_unsampled_probs : bool, default=False
        If ``True``, append an extra "unsampled" fate column estimated from the residual mass.
    ignore_warnings : bool, default=True
        If True, suppress OT warnings during the solve.
    cost_scaling : {"median", "mean", None}, default="median"
        Normalization applied to the cost matrix.
    cost_metric : str, default="sqeuclidean"
        Distance metric passed to ``ot.dist``.
    float_type : numpy dtype, default=np.float64
        Floating-point dtype used for OT computation. Higher precision may improve stability of sinkhorn solves.
    store_in_adata : bool, default=True
        If True, store the result dictionaries in ``adata.uns`` under ``output_key``.
    output_key : str, default="stitchOT"
        Key prefix used when writing results to ``adata.uns``.

    Returns
    -------
    fate_matrix : ndarray
        Fate probability matrix for the source cells, grouped by ``target_names``.
    norm_OT_plan : ndarray
        Row-normalized transport plan.
    OT_plan : ndarray
        Raw transport plan returned by the unbalanced Sinkhorn solver.
    mask : ndarray of bool
        Boolean mask over the original target subset indicating which target cells were kept after subsampling.

    Notes
    -----
    When ``store_in_adata=True``, the following keys are written:

    * ``adata.obsm[f"{output_key}_fate_matrix"]``: full-sized fate matrix aligned to ``adata.obs_names``.
    * ``adata.obs[f"{output_key}_is_source"]``: boolean flag for source cells.
    * ``adata.obs[f"{output_key}_target_sampled"]``: boolean flag for retained target cells.
    * ``adata.uns[output_key]``: dictionary with metadata, indices, and the raw OT outputs.
    """
    if ignore_warnings:
        warnings.filterwarnings("ignore")

    if obsm_key is not None:
        data_source = adata[adata.obs["location"] == source_name].obsm[obsm_key]
        data_target = adata[adata.obs["location"].isin(target_names)].obsm[obsm_key]
    else:
        data_source = adata[adata.obs["location"] == source_name].X
        data_target = adata[adata.obs["location"].isin(target_names)].X

    target_locations = (
        adata[adata.obs["location"].isin(target_names)].obs["location"].values
    )

    fate_groups_fractions = np.zeros(len(target_names), dtype=float_type)
    OT_plan = np.zeros((data_source.shape[0], data_target.shape[0]))
    source_marginals_threshold = np.zeros((data_source.shape[0], len(target_names)))

    for i, loc in enumerate(target_names):
        C = ot.dist(
            data_source, data_target[target_locations == loc], metric=cost_metric
        )
        if cost_scaling == "median":
            C /= np.median(C)
        elif cost_scaling == "mean":
            C /= np.mean(C)

        a = np.ones((C.shape[0])) / C.shape[0]
        b = np.ones((C.shape[1])) / C.shape[1]

        OT_plan_loc = ot.sinkhorn_unbalanced(
            a.astype(float_type),
            b.astype(float_type),
            C.astype(float_type),
            reg=epsilon,
            verbose=False,
            log=False,
            reg_m=((epsilon * tau_a) / (1 - tau_a), (epsilon * tau_b) / (1 - tau_b)),
            reg_type="entropy",
        )
        a = OT_plan_loc.sum(axis=1)
        source_marginals_threshold[:, i] = 2 * a < 1 / data_source.shape[0]
        fate_groups_fractions[i] = np.sum(a > (1 / C.shape[0])) / C.shape[0]
        OT_plan[:, target_locations == loc] = OT_plan_loc

    fate_groups_fractions[fate_groups_fractions < 0.01] = 0.01
    fate_groups_fractions = fate_groups_fractions / np.sum(fate_groups_fractions)

    if rescale_OT_plans:
        if use_true_fractions is not False:
            fate_groups_fractions = use_true_fractions
        for i, loc in enumerate(target_names):
            OT_plan[:, target_locations == loc] *= fate_groups_fractions[i]

    if np.isnan(OT_plan).any():
        warnings.warn(
            "OT_plan contains NaN values; replacing them with 0.", UserWarning
        )
    OT_plan = np.nan_to_num(OT_plan, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = OT_plan.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0):
        zero_row_indices = np.where(row_sums == 0)[0]
        warnings.warn(
            f"The following rows in OT_plan have zero mass: {zero_row_indices}",
            UserWarning,
        )

    norm_OT_plan = OT_plan / row_sums
    norm_OT_plan = np.nan_to_num(norm_OT_plan, nan=0.0, posinf=0.0, neginf=0.0)

    fate_groups = [np.where(target_locations == loc)[0] for loc in target_names]
    fate_matrix = get_fates(OT_plan, fate_groups)

    if assign_unsampled_probs:
        unsampled_prob = (
            np.all(source_marginals_threshold, axis=1, keepdims=True).astype(float)
            * 1
            / (C.shape[0])
        )
        fate_matrix = np.hstack((fate_matrix, unsampled_prob))

    fate_matrix = np.nan_to_num(fate_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    fate_row_sums = fate_matrix.sum(axis=1, keepdims=True)
    fate_matrix = fate_matrix / fate_row_sums

    if store_in_adata:
        if assign_unsampled_probs:
            target_names = target_names + ["Unobserved"]
        adata.uns[output_key] = {
            "fate_matrix": fate_matrix,
            "OT_plan": OT_plan,
            "norm_OT_plan": norm_OT_plan,
            "fate_groups": fate_groups,
            "target_names": target_names,
            "source_name": source_name,
            "assign_unsampled_probs": assign_unsampled_probs,
        }
        return None

    return fate_matrix, norm_OT_plan, OT_plan
