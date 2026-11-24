"""
NeuroSpike Visualization Module
===============================

Publication-quality visualization functions for Spiking Neural Networks.
Provides spike raster plots, membrane potential traces, firing rate analysis,
weight distributions, STDP windows, training curves, confusion matrices,
energy comparisons, and comprehensive dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Consistent colour palette
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

_STYLE = "seaborn-v0_8-darkgrid"


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a NumPy array safely."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _get_or_create_ax(
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax), creating them if *ax* is None."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ===================================================================
# 1. Spike Raster Plot
# ===================================================================
def plot_spike_raster(
    spikes: Union[torch.Tensor, np.ndarray],
    num_neurons: int = 100,
    time_range: Optional[Tuple[int, int]] = None,
    title: str = "Spike Raster Plot",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a neuroscience-style spike raster.

    Parameters
    ----------
    spikes : array-like, shape ``(time_steps, neurons)``
        Binary spike tensor (1 = spike, 0 = no spike).
    num_neurons : int
        Maximum number of neurons to display.
    time_range : tuple[int, int], optional
        ``(start, end)`` time-step window.  ``None`` uses the full range.
    title : str
        Figure title.
    ax : matplotlib Axes, optional
        Pre-existing axes.  If ``None`` a new figure is created.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    spikes_np = _to_numpy(spikes)

    # Clamp neuron count
    num_neurons = min(num_neurons, spikes_np.shape[1])
    spikes_np = spikes_np[:, :num_neurons]

    # Apply time range
    if time_range is not None:
        start, end = time_range
        start = max(0, start)
        end = min(spikes_np.shape[0], end)
        spikes_np = spikes_np[start:end, :]
    else:
        start = 0

    with plt.style.context(_STYLE):
        fig, ax = _get_or_create_ax(ax, figsize=(12, 6))

        # Compute per-neuron firing rate for colour-coding
        firing_rates = spikes_np.mean(axis=0)
        fr_min = firing_rates.min() if firing_rates.max() > 0 else 0.0
        fr_max = firing_rates.max() if firing_rates.max() > 0 else 1.0
        norm = mcolors.Normalize(vmin=fr_min, vmax=fr_max)
        cmap = plt.cm.coolwarm  # blue (low) → red (high)

        for neuron_idx in range(num_neurons):
            spike_times = np.where(spikes_np[:, neuron_idx] > 0.5)[0] + start
            colour = cmap(norm(firing_rates[neuron_idx]))
            ax.scatter(
                spike_times,
                np.full_like(spike_times, neuron_idx),
                s=1.0,
                color=colour,
                marker="|",
                linewidths=0.8,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Firing Rate", fontsize=10)

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Neuron Index", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(-0.5, num_neurons - 0.5)
        fig.tight_layout()

    return fig, ax


# ===================================================================
# 2. Membrane Potential Dynamics
# ===================================================================
def plot_membrane_potential(
    membrane_traces: Union[torch.Tensor, np.ndarray],
    neuron_indices: Optional[Sequence[int]] = None,
    threshold: float = 1.0,
    title: str = "Membrane Potential Dynamics",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot membrane potential over time for selected neurons.

    Parameters
    ----------
    membrane_traces : array-like, shape ``(time_steps, neurons)``
    neuron_indices : list[int], optional
        Which neurons to plot.  Defaults to first 5.
    threshold : float
        Spiking threshold shown as a dashed red line.
    title : str
        Figure title.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mem_np = _to_numpy(membrane_traces)

    if neuron_indices is None:
        neuron_indices = list(range(min(5, mem_np.shape[1])))

    with plt.style.context(_STYLE):
        fig, ax = _get_or_create_ax(ax, figsize=(12, 5))

        time = np.arange(mem_np.shape[0])
        colours = plt.cm.tab10(np.linspace(0, 1, len(neuron_indices)))

        for idx, neuron_id in enumerate(neuron_indices):
            trace = mem_np[:, neuron_id]
            ax.plot(
                time,
                trace,
                color=colours[idx],
                linewidth=1.0,
                alpha=0.85,
                label=f"Neuron {neuron_id}",
            )

            # Mark spike events (membrane ≥ threshold)
            spike_mask = trace >= threshold
            ax.scatter(
                time[spike_mask],
                trace[spike_mask],
                color=PALETTE["danger"],
                s=30,
                zorder=5,
                marker="o",
                edgecolors="white",
                linewidths=0.5,
            )

        ax.axhline(
            y=threshold,
            color=PALETTE["danger"],
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold ({threshold})",
        )

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Membrane Potential", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()

    return fig, ax


# ===================================================================
# 3. Layer-wise Firing Rates
# ===================================================================
def plot_firing_rates(
    spike_data: Union[List[Union[torch.Tensor, np.ndarray]], Dict[str, Union[torch.Tensor, np.ndarray]]],
    layer_names: Optional[List[str]] = None,
    title: str = "Layer-wise Firing Rates",
) -> Tuple[plt.Figure, plt.Axes]:
    """Violin + strip plot showing firing-rate distributions per layer.

    Parameters
    ----------
    spike_data : list of arrays or dict mapping name → array
        Each array has shape ``(time_steps, neurons)`` (binary spikes).
    layer_names : list[str], optional
        Names for each layer.
    title : str

    Returns
    -------
    fig, ax
    """
    # Normalise input to list of numpy arrays
    if isinstance(spike_data, dict):
        if layer_names is None:
            layer_names = list(spike_data.keys())
        arrays = [_to_numpy(v) for v in spike_data.values()]
    else:
        arrays = [_to_numpy(s) for s in spike_data]

    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(arrays))]

    # Compute per-neuron firing rates
    rates = [arr.mean(axis=0) for arr in arrays]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(rates)), 6))

        parts = ax.violinplot(rates, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE["primary"])
            pc.set_alpha(0.6)

        # Annotate mean and std
        for i, r in enumerate(rates):
            mean_val = np.mean(r)
            std_val = np.std(r)
            ax.text(
                i + 1,
                mean_val + std_val + 0.01,
                f"μ={mean_val:.3f}\nσ={std_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["gray"],
            )

        ax.set_xticks(range(1, len(layer_names) + 1))
        ax.set_xticklabels(layer_names, fontsize=11)
        ax.set_ylabel("Firing Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    return fig, ax


# ===================================================================
# 4. Weight Distribution
# ===================================================================
def plot_weight_distribution(
    model: nn.Module,
    title: str = "Weight Distribution Evolution",
    reference_model: Optional[nn.Module] = None,
) -> plt.Figure:
    """Plot weight histograms for each layer of *model*.

    Parameters
    ----------
    model : nn.Module
        The trained (or current) model.
    title : str
    reference_model : nn.Module, optional
        A second model (e.g. pre-training snapshot) for comparison.

    Returns
    -------
    fig : matplotlib Figure
    """
    named_params = [
        (n, p) for n, p in model.named_parameters() if "weight" in n and p.dim() >= 2
    ]
    if not named_params:
        named_params = [(n, p) for n, p in model.named_parameters() if p.dim() >= 1]

    n_layers = len(named_params)
    if n_layers == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No weight parameters found", ha="center", va="center")
        return fig

    ncols = min(3, n_layers)
    nrows = (n_layers + ncols - 1) // ncols

    ref_params: Dict[str, np.ndarray] = {}
    if reference_model is not None:
        ref_params = {
            n: _to_numpy(p)
            for n, p in reference_model.named_parameters()
            if "weight" in n and p.dim() >= 2
        }

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, (name, param) in enumerate(named_params):
            ax = axes_flat[idx]
            w = _to_numpy(param).flatten()
            ax.hist(w, bins=50, alpha=0.7, color=PALETTE["primary"], label="Current")
            if name in ref_params:
                w_ref = ref_params[name].flatten()
                ax.hist(w_ref, bins=50, alpha=0.4, color=PALETTE["secondary"], label="Reference")
                ax.legend(fontsize=8)
            short_name = name.rsplit(".", 1)[-1] if "." in name else name
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Weight value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)

        # Hide unused axes
        for idx in range(n_layers, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

    return fig


# ===================================================================
# 5. STDP Learning Window
# ===================================================================
def plot_stdp_window(
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    A_plus: float = 0.01,
    A_minus: float = 0.012,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the classic STDP learning window.

    Parameters
    ----------
    tau_plus, tau_minus : float
        Time constants for potentiation / depression (ms).
    A_plus, A_minus : float
        Maximum weight change magnitudes.

    Returns
    -------
    fig, ax
    """
    dt = np.linspace(-60, 60, 500)

    dw = np.where(
        dt >= 0,
        A_plus * np.exp(-dt / tau_plus),
        -A_minus * np.exp(dt / tau_minus),
    )
    # Exact zero at dt=0 is ambiguous; keep the positive side convention
    dw[np.argmin(np.abs(dt))] = 0.0

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        # LTP region (dt > 0)
        pos_mask = dt >= 0
        neg_mask = dt < 0
        ax.fill_between(dt[pos_mask], 0, dw[pos_mask], alpha=0.3, color=PALETTE["accent"], label="LTP (potentiation)")
        ax.fill_between(dt[neg_mask], 0, dw[neg_mask], alpha=0.3, color=PALETTE["danger"], label="LTD (depression)")
        ax.plot(dt, dw, color="black", linewidth=1.5)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        # Annotations
        ax.annotate(
            "Pre before Post\n(Causal → LTP)",
            xy=(15, A_plus * np.exp(-15 / tau_plus)),
            xytext=(30, A_plus * 0.8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]),
            color=PALETTE["accent"],
        )
        ax.annotate(
            "Post before Pre\n(Anti-causal → LTD)",
            xy=(-15, -A_minus * np.exp(-15 / tau_minus)),
            xytext=(-50, -A_minus * 0.8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=PALETTE["danger"]),
            color=PALETTE["danger"],
        )

        ax.set_xlabel("Δt = t_post − t_pre  (ms)", fontsize=12)
        ax.set_ylabel("ΔW (weight change)", fontsize=12)
        ax.set_title("Spike-Timing-Dependent Plasticity (STDP) Window", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)

        # Parameter box
        textstr = (
            f"τ₊ = {tau_plus:.1f} ms\n"
            f"τ₋ = {tau_minus:.1f} ms\n"
            f"A₊ = {A_plus}\n"
            f"A₋ = {A_minus}"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right", bbox=props)

        fig.tight_layout()

    return fig, ax


# ===================================================================
# 6. Training Curves
# ===================================================================
def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
) -> plt.Figure:
    """Multi-panel training metrics plot.

    Parameters
    ----------
    history : dict
        Expected keys (all optional): ``train_loss``, ``val_loss``,
        ``train_acc``, ``val_acc``, ``spike_rate``, ``lr``.
    title : str

    Returns
    -------
    fig : matplotlib Figure
    """
    panels: List[Tuple[str, List[Tuple[str, str, str]]]] = []

    if "train_loss" in history or "val_loss" in history:
        entries: List[Tuple[str, str, str]] = []
        if "train_loss" in history:
            entries.append(("train_loss", "Train Loss", PALETTE["primary"]))
        if "val_loss" in history:
            entries.append(("val_loss", "Val Loss", PALETTE["secondary"]))
        panels.append(("Loss", entries))

    if "train_acc" in history or "val_acc" in history:
        entries = []
        if "train_acc" in history:
            entries.append(("train_acc", "Train Acc", PALETTE["accent"]))
        if "val_acc" in history:
            entries.append(("val_acc", "Val Acc", PALETTE["danger"]))
        panels.append(("Accuracy (%)", entries))

    if "spike_rate" in history:
        panels.append(("Spike Rate", [("spike_rate", "Spike Rate", PALETTE["purple"])]))

    if "lr" in history:
        panels.append(("Learning Rate", [("lr", "LR", PALETTE["brown"])]))

    n_panels = max(len(panels), 1)
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        for ax, (ylabel, entries) in zip(axes, panels):
            for key, label, colour in entries:
                data = history[key]
                epochs = range(1, len(data) + 1)
                ax.plot(epochs, data, label=label, color=colour, linewidth=1.5)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    return fig


# ===================================================================
# 7. Confusion Matrix
# ===================================================================
def plot_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray, List[int]],
    y_pred: Union[torch.Tensor, np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    class_names : list[str], optional
    normalize : bool
        Row-normalise the matrix so values represent percentages.
    title : str

    Returns
    -------
    fig, ax
    """
    y_true_np = np.asarray(_to_numpy(y_true) if isinstance(y_true, torch.Tensor) else y_true).flatten()
    y_pred_np = np.asarray(_to_numpy(y_pred) if isinstance(y_pred, torch.Tensor) else y_pred).flatten()

    labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
    n_classes = len(labels)

    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true_np, y_pred_np):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums * 100.0

    if class_names is None:
        class_names = [str(l) for l in labels]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * n_classes), max(5, 0.55 * n_classes)))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        fmt_str = ".1f" if normalize else ".0f"
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                colour = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:{fmt_str}}", ha="center", va="center", color=colour, fontsize=8)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    return fig, ax


# ===================================================================
# 8. Energy Efficiency Comparison
# ===================================================================
def plot_energy_comparison(
    snn_ops: Union[int, float],
    ann_ops: Union[int, float],
    title: str = "Energy Efficiency: SNN vs ANN",
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart comparing SNN synaptic ops vs ANN MAC ops.

    Uses standard energy estimates:
    - SynOp ≈ 0.9 pJ  (45 nm technology)
    - MAC   ≈ 4.6 pJ

    Parameters
    ----------
    snn_ops : int | float
        Number of SNN synaptic operations.
    ann_ops : int | float
        Number of ANN multiply-accumulate operations.
    title : str

    Returns
    -------
    fig, ax
    """
    E_SYNOP_PJ = 0.9   # pJ per synaptic operation
    E_MAC_PJ = 4.6      # pJ per MAC

    snn_energy = snn_ops * E_SYNOP_PJ
    ann_energy = ann_ops * E_MAC_PJ
    savings = (1.0 - snn_energy / ann_energy) * 100.0 if ann_energy > 0 else 0.0

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Panel 1: Operations ---
        bars1 = axes[0].bar(
            ["SNN\n(SynOps)", "ANN\n(MACs)"],
            [snn_ops, ann_ops],
            color=[PALETTE["primary"], PALETTE["secondary"]],
            edgecolor="white",
            linewidth=1.2,
        )
        axes[0].set_ylabel("Operations", fontsize=12)
        axes[0].set_title("Operation Count", fontsize=12, fontweight="bold")
        for bar, val in zip(bars1, [snn_ops, ann_ops]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # --- Panel 2: Energy ---
        bars2 = axes[1].bar(
            ["SNN", "ANN"],
            [snn_energy, ann_energy],
            color=[PALETTE["accent"], PALETTE["danger"]],
            edgecolor="white",
            linewidth=1.2,
        )
        axes[1].set_ylabel("Energy (pJ)", fontsize=12)
        axes[1].set_title("Estimated Energy Consumption", fontsize=12, fontweight="bold")
        for bar, val in zip(bars2, [snn_energy, ann_energy]):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2e} pJ",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Savings annotation
        axes[1].annotate(
            f"SNN saves {savings:.1f}%\nenergy",
            xy=(0.5, 0.85),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            color=PALETTE["accent"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["accent"], alpha=0.8),
        )

        # Footnotes
        fig.text(
            0.5,
            -0.02,
            "Energy estimates: SynOp ≈ 0.9 pJ, MAC ≈ 4.6 pJ (45 nm technology, Horowitz 2014)",
            ha="center",
            fontsize=9,
            color=PALETTE["gray"],
        )

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    return fig, axes  # type: ignore[return-value]


# ===================================================================
# 9. Comprehensive Dashboard
# ===================================================================
def create_dashboard(
    model: nn.Module,
    spike_data: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
    training_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a 3×3 multi-panel dashboard combining multiple visualisations.

    Parameters
    ----------
    model : nn.Module
        Trained SNN model.
    spike_data : array-like or dict
        Either a single spike tensor ``(time_steps, neurons)`` or a dict
        mapping layer names to spike tensors.
    training_history : dict
        Training metrics history (see :func:`plot_training_curves`).
    save_path : str, optional
        If given, saves the figure at this path (300 dpi PNG).

    Returns
    -------
    fig : matplotlib Figure
    """
    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        # ---- (0, 0): Spike raster ----
        ax_raster = fig.add_subplot(gs[0, 0])
        if isinstance(spike_data, dict):
            first_key = next(iter(spike_data))
            raster_data = _to_numpy(spike_data[first_key])
        else:
            raster_data = _to_numpy(spike_data)
        plot_spike_raster(raster_data, num_neurons=50, title="Spike Raster", ax=ax_raster)

        # ---- (0, 1): Membrane potential (synthetic from spikes) ----
        ax_mem = fig.add_subplot(gs[0, 1])
        # Create approximate membrane traces from spikes using exponential filter
        mem_traces = np.zeros_like(raster_data[:, :5], dtype=np.float32)
        beta = 0.9
        for t in range(1, raster_data.shape[0]):
            mem_traces[t] = beta * mem_traces[t - 1] * (1 - raster_data[t - 1, :5]) + raster_data[t, :5] * 0.5
        plot_membrane_potential(mem_traces, threshold=1.0, title="Membrane Traces (approx)", ax=ax_mem)

        # ---- (0, 2): Firing rates ----
        ax_fr = fig.add_subplot(gs[0, 2])
        if isinstance(spike_data, dict):
            fr_list = [_to_numpy(v) for v in spike_data.values()]
            fr_names = list(spike_data.keys())
        else:
            # Split into pseudo-layers
            n = raster_data.shape[1]
            chunk = max(1, n // 3)
            fr_list = [raster_data[:, i * chunk: (i + 1) * chunk] for i in range(3)]
            fr_names = [f"Layer {i}" for i in range(3)]
        # Reuse the violin plot but on this ax
        rates = [arr.mean(axis=0) for arr in fr_list]
        parts = ax_fr.violinplot(rates, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE["primary"])
            pc.set_alpha(0.6)
        ax_fr.set_xticks(range(1, len(fr_names) + 1))
        ax_fr.set_xticklabels(fr_names, fontsize=8)
        ax_fr.set_title("Firing Rates", fontsize=11, fontweight="bold")

        # ---- (1, 0): Training loss ----
        ax_loss = fig.add_subplot(gs[1, 0])
        if "train_loss" in training_history:
            ax_loss.plot(training_history["train_loss"], label="Train", color=PALETTE["primary"])
        if "val_loss" in training_history:
            ax_loss.plot(training_history["val_loss"], label="Val", color=PALETTE["secondary"])
        ax_loss.set_title("Loss", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend(fontsize=8)

        # ---- (1, 1): Training accuracy ----
        ax_acc = fig.add_subplot(gs[1, 1])
        if "train_acc" in training_history:
            ax_acc.plot(training_history["train_acc"], label="Train", color=PALETTE["accent"])
        if "val_acc" in training_history:
            ax_acc.plot(training_history["val_acc"], label="Val", color=PALETTE["danger"])
        ax_acc.set_title("Accuracy", fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.legend(fontsize=8)

        # ---- (1, 2): Spike rate over epochs ----
        ax_sr = fig.add_subplot(gs[1, 2])
        if "spike_rate" in training_history:
            ax_sr.plot(training_history["spike_rate"], color=PALETTE["purple"], linewidth=1.5)
        ax_sr.set_title("Spike Rate / Epoch", fontsize=11, fontweight="bold")
        ax_sr.set_xlabel("Epoch")

        # ---- (2, 0): Weight distribution (first layer) ----
        ax_wd = fig.add_subplot(gs[2, 0])
        first_weight = None
        for n, p in model.named_parameters():
            if "weight" in n and p.dim() >= 2:
                first_weight = _to_numpy(p).flatten()
                ax_wd.set_title(f"Weights: {n}", fontsize=9, fontweight="bold")
                break
        if first_weight is not None:
            ax_wd.hist(first_weight, bins=40, color=PALETTE["primary"], alpha=0.7)
        else:
            ax_wd.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax_wd.transAxes)

        # ---- (2, 1): STDP window ----
        ax_stdp = fig.add_subplot(gs[2, 1])
        dt_arr = np.linspace(-60, 60, 300)
        dw_arr = np.where(dt_arr >= 0, 0.01 * np.exp(-dt_arr / 20.0), -0.012 * np.exp(dt_arr / 20.0))
        ax_stdp.fill_between(dt_arr[dt_arr >= 0], 0, dw_arr[dt_arr >= 0], alpha=0.3, color=PALETTE["accent"])
        ax_stdp.fill_between(dt_arr[dt_arr < 0], 0, dw_arr[dt_arr < 0], alpha=0.3, color=PALETTE["danger"])
        ax_stdp.plot(dt_arr, dw_arr, "k-", linewidth=1.2)
        ax_stdp.axhline(0, color="gray", linewidth=0.5)
        ax_stdp.set_title("STDP Window", fontsize=11, fontweight="bold")
        ax_stdp.set_xlabel("Δt (ms)")

        # ---- (2, 2): LR schedule ----
        ax_lr = fig.add_subplot(gs[2, 2])
        if "lr" in training_history:
            ax_lr.plot(training_history["lr"], color=PALETTE["brown"], linewidth=1.5)
        ax_lr.set_title("Learning Rate", fontsize=11, fontweight="bold")
        ax_lr.set_xlabel("Epoch")

        fig.suptitle("NeuroSpike — Training Dashboard", fontsize=16, fontweight="bold", y=0.99)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig
