import warnings

import numpy as np
import ot

from .utils import get_fates


def JointOT(
    adata,
    obsm_key,
    source_name,
    target_names,
    tau_b=0.5,
    tau_a=0.995,
    epsilon=0.005,
    use_true_fractions=None,
    assign_unsampled_probs=False,
    lambda1=100,
    ignore_warnings=True,
    cost_scaling="median",
    cost_metric="sqeuclidean",
    float_type=np.float64,
    store_in_adata=True,
    output_key="jointOT",
):
    """Perform a single joint optimal-transport solve for cell-fate inference.

    This method builds one global source-to-target transport plan across all target cells and
    then aggregates it to a fate matrix grouped by tissue/location.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing expression values and a ``location`` annotation in
        ``adata.obs``.
    obsm_key : str or None
        Name of the feature matrix in ``adata.obsm``. If ``None``, ``adata.X`` is used.
    source_name : str
        Label in ``adata.obs["location"]`` identifying the source compartment.
    target_names : list of str
        Labels in ``adata.obs["location"]`` identifying the target compartments.
    tau_b : float, default=0.5
        Sinkhorn regularization parameter for the target-side marginal.
    tau_a : float, default=0.995
        Sinkhorn regularization parameter for the source-side marginal.
    epsilon : float, default=0.005
        Entropic regularization strength used by the unbalanced Sinkhorn solver.
    use_true_fractions : array-like or None, default=None
        Optional per-target fractions used to initialize the target marginal. If provided, they
        override the uniform target distribution.
    assign_unsampled_probs : bool, default=False
        If True, append an extra ``Unobserved`` fate column representing the unresolved mass.
    lambda1 : float, default=100
        Scaling factor used when deriving the unsampled probability if
        ``assign_unsampled_probs=True``.
    ignore_warnings : bool, default=True
        If True, suppress OT warnings during the solve.
    cost_scaling : {"median", "mean", None}, default="median"
        Normalization applied to the cost matrix.
    cost_metric : str, default="sqeuclidean"
        Distance metric passed to ``ot.dist``.
    float_type : numpy.dtype, default=np.float64
        Floating-point dtype used during the OT solve.
    store_in_adata : bool, default=True
        If True, store outputs under ``adata.uns[output_key]``.
    output_key : str, default="jointOT"
        Key prefix used when writing results into ``adata.uns``.

    Returns
    -------
    fate_matrix : ndarray
        Fate probability matrix for the source cells, grouped by ``target_names``.
    norm_OT_plan : ndarray
        Row-normalized transport plan.
    OT_plan : ndarray
        Raw transport plan returned by the unbalanced Sinkhorn solver.

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

    a = np.ones(data_source.shape[0]) / data_source.shape[0]
    b = np.ones(data_target.shape[0]) / data_target.shape[0]

    for j, loc in enumerate(target_names):
        loc_mask = target_locations == loc
        marginal = np.sum(loc_mask)
        if use_true_fractions is not None:
            # Use ground truth location fractions
            b[loc_mask] = use_true_fractions[j] / marginal
        else:
            # Default: uniform distribution across locations
            b[loc_mask] = (1 / len(target_names)) / marginal

    C = ot.dist(data_source, data_target, metric=cost_metric)
    if cost_scaling == "median":
        C /= np.median(C)
    elif cost_scaling == "mean":
        C /= np.mean(C)

    OT_plan = ot.sinkhorn_unbalanced(
        a.astype(float_type),
        b.astype(float_type),
        C.astype(float_type),
        reg=epsilon,
        numItermax=100000,
        verbose=False,
        log=False,
        stopThr=1e-10,
        reg_m=((epsilon * tau_a) / (1 - tau_a), (epsilon * tau_b) / (1 - tau_b)),
        reg_type="entropy",
    )

    if np.isnan(OT_plan).any():
        warnings.warn(
            "OT_plan contains NaN values; replacing them with 0.", UserWarning
        )
    OT_plan = np.nan_to_num(OT_plan, nan=0.0, posinf=0.0, neginf=0.0)

    norm_OT_plan = OT_plan / OT_plan.sum(axis=1, keepdims=True)
    norm_OT_plan = np.nan_to_num(norm_OT_plan, nan=0.0, posinf=0.0, neginf=0.0)

    fate_groups = [np.where(target_locations == loc)[0] for loc in target_names]
    fate_matrix = get_fates(OT_plan, fate_groups)

    if assign_unsampled_probs:
        diff_marginals = np.clip(
            (1 / C.shape[0] - np.sum(OT_plan, axis=1)) * lambda1, 0, 1
        )
        unsampled_prob = diff_marginals.reshape(-1, 1)
        fate_matrix = np.hstack((fate_matrix, unsampled_prob))

    fate_matrix = np.nan_to_num(fate_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    fate_matrix = fate_matrix / np.sum(fate_matrix, axis=1, keepdims=True)

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
