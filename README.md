# CellVADIS

<p align="center">
    <img src="docs/CellVADIS_logo.png" width="80%" alt="CellVADIS logo"/>
</p>

CellVADIS is a Python toolkit based on optimal transport for inferring cross-tissue immune cell dynamics from single-cell snapshots, as obtained by e.g. scRNA-seq, flow cytometry or mass cytometry.

It relates to the following publication:
* Wicker L., Varga S., Hasenauer J., Graw F.: **CellVADIS: Inferring cross-tissue immune cell trafficking dynamics from single-cell snapshots**, bioRxiv 2026, https://doi.org/10.64898/2026.11.18.767220.

By connecting single-cell measurements across tissue sites, CellVADIS enables the prediction of individual cell migration fates, the identification of tissue-specific trafficking signatures on circulating cells, as well as the determination of tissue-specific 'homing niches' for tissue-entering cells, which supports the inference of location-specific cell developmental trajectories.
It is designed for scenarios in which a source compartment is defined (e.g., blood) and one wants to estimate how these source cells are distributed across one or more target tissues (or states).

## What CellVADIS can do:

- Infer tissue-specific fate probabilities from a source population
- Score how gene expression correlates with inferred tissue fates for source cells (trafficking signatures), as well as the mass received by tissue cells (homing niches).
- Summarize transitions with plotting functions for tissues, dominant fates, and homing niches
- Work naturally with `AnnData` objects and `scanpy` or `pytometry` workflows

Typical use cases include:

- immune-cell trafficking from blood to tissues
- cross-tissue mapping from single-cell atlases
- identifying markers associated with tissue homing or destination preference
- Inferring the proportion of source cells trafficking to specific target tissues


## Installation

### From source

```bash
git clone https://github.com/GrawLab/CellVADIS.git
cd CellVADIS
pip install -e .
```

### Requirements

CellVADIS is intended for Python 3.11+ and has the following base dependencies:

- numpy
- pandas
- scipy
- scanpy
- matplotlib
- seaborn
- POT

If you are running the example notebooks, you may also need:

- harmonypy
- pytometry

## Quick start example

```python
import CellVADIS

# Assume adata is an AnnData object with:
# - expression matrix/latent space representation used for distance calculation
#   in adata.X or adata.obsm[...]
# - a tissue/location annotation in adata.obs["location"],
#   for this example this should include SOURCE and TARGET1 to TARGET4

source_name = "SOURCE"
target_names = ["TARGET1", "TARGET2", "TARGET3", "TARGET4"]

# Recommended: balance the dataset before running CellVADIS. This helps reduce
# the dominance of large tissue groups, speeds up OT, and is often preferable to
# applying CellVADIS on the full dataset. Too large cell numbers (>50k) might also lead to memory issues.
adata = CellVADIS.utils.obs_key_wise_subsampling(
    adata,
    obs_keys="location",
    N_target=5000, #Number of cells to keep for each target tissue
    N_source=10000, #Number of cells to keep in source tissue
    source_name=source_name,
)

# Run the OT models on the subsampled data.
CellVADIS.JointOT(
    adata=adata,
    obsm_key="X_pca",
    source_name=source_name,
    target_names=target_names,
)

CellVADIS.StitchOT(
    adata=adata,
    obsm_key="X_pca",
    source_name=source_name,
    target_names=target_names,
)

# Plot overall tissue-specific transition proportions
CellVADIS.plotting.plot_tissue_transition_proportions(
    adata,
    output_key="stitchOT",
    return_proportions=True,
)

# Plot dominant fate per source cell in a UMAP embedding
CellVADIS.plotting.plot_dominant_fates(
    adata,
    output_key="stitchOT",
)

# Plot fate probabilities projected on individual source cells as pie charts
CellVADIS.plotting.plot_fate_probabilities_with_pie_charts(
    adata,
    output_key="stitchOT",
    n_cells_to_plot=100,
)

# Plot average fate probability by cell type as a heatmap (if cell type information is available)
CellVADIS.plotting.plot_avg_fate_per_celltype(
    adata,
    output_key="stitchOT",
    celltype_key="celltype",
)

# Correlate marker genes with inferred source-cell fate probabilities
#  to identify trafficking signatures
marker_fate_corr = CellVADIS.utils.corr_marker_with_fates(
    adata,
    marker_genes=["CD3D", "CXCR6", "CCR7"],
    output_key="stitchOT",
    sort_df_by="mean",
)

# Plot homing niches: target-side tissue projections of high-probability source cells
CellVADIS.plotting.plot_homing_niches(
    adata,
    output_key="jointOT",
)

# Correlate marker genes with target marginals to quantify homing niches
marker_target_corr = CellVADIS.utils.corr_marker_with_target_marginals(
    adata,
    marker_genes=["CD3D", "CXCR6", "CCR7"],
    output_key="jointOT",
)
```


## Output structure

By default, CellVADIS stores the analysis results in `adata.uns` under the chosen `output_key`.

For example:

```python
adata.uns["stitchOT"]["fate_matrix"]
adata.uns["stitchOT"]["OT_plan"]
adata.uns["stitchOT"]["norm_OT_plan"]
adata.uns["stitchOT"]["target_names"]
adata.uns["stitchOT"]["source_name"]
```

These entries contain the fate probabilities, coupling matrices, grouping information, and metadata needed for downstream plotting and statistics.

## Example notebooks

The repository includes example notebooks for two common scenarios:

- `docs/notebooks/example_TICA.ipynb`: Exemplary workflow on scRNA-seq data from tissue immune cell atlas by Conde et al.
- `docs/notebooks/example_MASS.ipynb`: cytometry-based workflow on the MASS atlas by Wong et al.

These notebooks show how to prepare the data, run the models, and visualize the results.


## Notes on interpretation

The inferred output is a source-to-target probability distribution for each source cell. In practice:

- Values near 1 in one target column indicate strong destination preference,
- Mixed distributions may reflect broad or uncertain trafficking behavior,
- Correlation analysis can help connect fate probabilities to marker expression to identify trafficking signatures,
- Relating expressions levels on tissue cells with their received mass as inferred by the unbalanced optimal transport methods might reveal homing niches and identify tissue-entry points.
- Both Joint- and StitchOT can also be run with `assign_unsampled_probs=True`, which takes into account that not all tissues a source cell could traffick to are sampled and assigns an additional probability representing the fate potential towards these unobserved tissues.
- While benchmarking on synthetic data revealed ([Repo link to paper results](https://github.com/GrawLab/CellVADIS_analysis)) that JointOT and StitchOT are both similar in accuracy, we suggest to choose the method based on the downstream tasks. JointOT is preferred for analyses from the viewpoint of the target cells (identification of homing niches etc.), while StitchOT might be preferred for source-cell based analysis, such as identifying trafficking signatures.


## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation
