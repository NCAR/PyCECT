import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # UF-ECT Setup Framework Examples
    This notebook contains examples of the setup framework described in Price-Broncucia et al, 2024. Most of the figures in the paper can be produced directly using data in the repository and this notebook.
    """
    )
    return


@app.cell
def _():
    import matplotlib.ticker as mtick
    import new_model_routine as newmr
    from matplotlib import pyplot as plt

    return mtick, newmr, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CESM Beta 3 ne30
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 1: Generate Experimental Ensemble Summary Files
    This first step of the setup framework involves generating a large experimental ensemble to test with. This process will be the most model dependent as the particular way to create ensembles will depend on each model. An example of a program to generate an ensemble for MPAS-A is included in the PyCECT repository for reference. CESM ensemble generation can be found in the CESM and associated repositories. After runs are created, spatial means can be calculated with the ECT code for each timeslice (example found in `MPAS/mpas_430_pyEnsSumMPAS.sh`).
    """
    )
    return


@app.cell
def _():
    ne30_folder = "../data/cesm_beta_3_setup_sums"
    ne30_file_prefix = "cesm_beta_3_2800_sum_"
    ne30_file_suffix = ""
    model_title = "CESM 3 beta 3 (ne30)"
    return model_title, ne30_file_prefix, ne30_file_suffix, ne30_folder


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 2: Summary Files
    With summary files created they should be read in.
    This finds which variables were shared across the timeseries, and creates an array of timesteps taking into account the save frequency and first save file (in case the first save file is not the IC's).
    """
    )
    return


@app.cell
def _(ne30_file_prefix, ne30_file_suffix, ne30_folder, newmr):
    # This set does not include initial conditions
    ne30_all_standard_means, ne30_timesteps, ne30_shared_vars = newmr.read_summary_series(
        ne30_folder, ne30_file_prefix, ne30_file_suffix, 1, 1
    )
    return ne30_all_standard_means, ne30_shared_vars, ne30_timesteps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 3: Determine Appropriate Timeslice
    Use Shapiro-Wilk P-score over time to determine an appropriate timeslice. We want a choose a late enough timestep to enable the number of non-normal variables to stabilize. Based on this plot set `new_timestep` equal to selected timestep. Keep in mind if your model only saves some variables on odd timesteps or similar.
    """
    )
    return


@app.cell
def _(ne30_all_standard_means, ne30_shared_vars, ne30_timesteps, newmr):
    ne30_shapiro_timesteps, ne30_non_normal_vars_percentage = newmr.shapiro_wilks_over_time(
        ne30_all_standard_means, ne30_timesteps, ne30_shared_vars
    )
    return ne30_non_normal_vars_percentage, ne30_shapiro_timesteps


@app.cell
def _(
    model_title,
    ne30_non_normal_vars_percentage,
    ne30_shapiro_timesteps,
    newmr,
):
    title = f"Non-Normal Variables for {model_title} vs Model Runtime"
    new_timestep = 9
    newmr.plot_shapiro_wilks_over_time(
        ne30_shapiro_timesteps,
        ne30_non_normal_vars_percentage,
        title=title,
        mark_timestep=new_timestep,
    )
    return (new_timestep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 4, Determine Number of PC Dimensions to Use
    Determine the required number of PC dimensions by increasing ensemble size until estimate of PCs required to explain 95% of the variance stabilizes. Based on plot, set `new_pc` and `new_ensemble`
    """
    )
    return


@app.cell
def _(ne30_file_prefix, ne30_folder, ne30_timesteps, new_timestep, newmr):
    sample_sizes, min_pca_included = newmr.stable_PCs_required(
        ne30_folder, ne30_file_prefix, '', new_timestep, ne30_timesteps, 0.95
    )
    return min_pca_included, sample_sizes


@app.cell
def _(min_pca_included, model_title, newmr, sample_sizes):
    title_1 = f'Minimum PCA Dimensions to Explain \n 95% Variance in {model_title} vs Ensemble Size'
    new_pc_count = 187
    new_ensemble = 2200
    newmr.plot_stable_PCs(
        sample_sizes,
        min_pca_included,
        vertical_line=new_ensemble,
        horizontal_line=new_pc_count,
        title=title_1,
    )
    return new_ensemble, new_pc_count


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 5, Determine if $m_\sigma$ Needs to be Adjusted
    By calculating the theoretical minimum false positive rate at the selected number of PCs, we can adjust $m_\sigma$ using a heuristic if needed.
    """
    )
    return


@app.cell
def _(new_pc_count, newmr):
    new_m = newmr.find_theoretical_m(new_pc_count, 0.002)
    new_m
    return (new_m,)


@app.cell
def _(mtick, new_m, newmr, plt):
    low_pc_plot = 150
    high_pc_plot = 200
    theorectical_fpr_high = newmr.theoretical_fpr_range(low_pc_plot, high_pc_plot, new_m)
    pca_vals = list(range(low_pc_plot, high_pc_plot))
    plt.plot(pca_vals, theorectical_fpr_high)
    plt.axhline(y=0.005, label="Goal: 0.5% FPR", color="red", linestyle="dashed")
    plt.title(fr"Theoretical False Positive Rate with $m_\sigma = {new_m:.2f}$")
    plt.xlabel("PC's Included")
    plt.ylabel("False Positive Rate (FPR)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 6: Ensure Ensemble Size is Sufficient for FPR
    Test the false positive rate at increasing ensemble sizes to determine if it yields a suffieciently low FPR. For comparison we can also show the increasing false positive rate at lower ensemble sizes.
    """
    )
    return


@app.cell
def _(
    model_title,
    ne30_file_prefix,
    ne30_folder,
    ne30_timesteps,
    new_ensemble,
    new_m,
    new_pc_count,
    new_timestep,
    newmr,
):
    title_2 = f'{model_title} False Positive Rate by Ensemble Size \n {new_pc_count} Principal Components, $m_\\sigma = {new_m:.2f}$'
    ne30_sample_sizes, ne30_fpr_by_ensemble_size = newmr.run_fpr_tests(
        ne30_folder,
        ne30_file_prefix,
        '',
        new_timestep,
        ne30_timesteps,
        new_pc_count,
        new_ensemble - 100,
        new_m,
        title=title_2,
    )
    return ne30_fpr_by_ensemble_size, ne30_sample_sizes


@app.cell
def _(
    model_title,
    ne30_fpr_by_ensemble_size,
    ne30_sample_sizes,
    new_ensemble,
    new_m,
    new_pc_count,
    newmr,
):
    title_3 = f'{model_title} False Positive Rate by Ensemble Size \n {new_pc_count} Principal Components, $m_\\sigma = {new_m:.2f}$'
    newmr.plot_fpr_tests(ne30_sample_sizes, ne30_fpr_by_ensemble_size, title_3, new_ensemble)
    return (title_3,)


@app.cell
def _(
    ne30_file_prefix,
    ne30_folder,
    ne30_timesteps,
    new_m,
    new_pc_count,
    new_timestep,
    newmr,
    title_3,
):
    long_sample_sizes, long_fpr_by_ensemble_size = newmr.run_fpr_tests(
        ne30_folder,
        ne30_file_prefix,
        '',
        new_timestep,
        ne30_timesteps,
        new_pc_count,
        500,
        new_m,
        title=title_3,
    )
    return long_fpr_by_ensemble_size, long_sample_sizes


@app.cell
def _(
    long_fpr_by_ensemble_size,
    long_sample_sizes,
    new_ensemble,
    newmr,
    title_3,
):
    newmr.plot_fpr_tests(long_sample_sizes, long_fpr_by_ensemble_size, title_3, new_ensemble)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
