import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
_COLORS = {
    'Adelie': '#1f77b4',
    'Chinstrap': '#ff7f0e',
    'Gentoo': '#2ca02c'
}
def plot_convergence(model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    history = np.asarray(getattr(model, 'accuracy_history', []), dtype=float)
    epochs = range(1, len(history) + 1)
    color = '#0f2f45'
    ax.plot(epochs, history * 100, '-', color=color, linewidth=2, zorder=3)
    if len(history) > 0:
        ax.fill_between(epochs, history * 100, alpha=0.1, color=color, zorder=2)
    ax.set_xlabel('Epoch', fontsize=11, color='#2f4a5f')
    ax.set_ylabel('Training Accuracy (%)', fontsize=11, color='#2f4a5f')
    ax.set_title(f'Network Convergence ({len(history)} epochs)', fontsize=13, fontweight='bold', pad=15, color='#0a2030')
    ax.set_ylim(-2, 102)
    for name, spine in ax.spines.items():
        spine.set_color('#b4c6d3')
        if name in ('top', 'right'):
            spine.set_visible(False)
    ax.tick_params(colors='#2f4a5f', labelsize=9)
    ax.grid(alpha=0.3, linestyle='--', color='gray', zorder=1)
    return fig, ax

def plot_confusion_matrix(cm, class_names, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    cm = np.array(cm)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cyan", ["#ffffff", "#52c6e3"])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10, color='#2f4a5f')