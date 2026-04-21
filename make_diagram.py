"""Generate a pipeline architecture diagram for the Methods section."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('white')

def box(ax, x, y, w, h, label, sublabel='', color='#D6EAF8', fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#2C3E50')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))

# Title
ax.text(6, 6.6, 'EA-Based Fraud Rule Discovery Pipeline',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#2C3E50')

# Row 1: Input
box(ax, 0.3, 5.0, 2.2, 0.9, 'Raw Dataset', 'CSV with fraud_label', color='#FDEBD0')
arrow(ax, 2.5, 5.45, 3.2, 5.45)

# Row 1: Preprocessing
box(ax, 3.2, 5.0, 2.2, 0.9, 'Preprocessing', 'Scale, clean, split', color='#FDEBD0')
arrow(ax, 5.4, 5.45, 6.1, 5.45)

# Row 1: ElasticNet
box(ax, 6.1, 5.0, 2.5, 0.9, 'ElasticNet Model', 'Baseline risk scores', color='#D5F5E3')
arrow(ax, 7.35, 5.0, 7.35, 4.35)

# EA Box (center, large)
box(ax, 1.5, 2.2, 9.0, 1.9, '', color='#EBF5FB')
ax.text(6.0, 3.95, 'Evolutionary Algorithm Loop  (generations)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#1A5276')

# EA internals
box(ax, 1.7, 2.4, 1.6, 1.3, 'Initialize\nPopulation', '100 random rules', color='#D6EAF8', fontsize=8)
arrow(ax, 3.3, 3.05, 3.6, 3.05)
box(ax, 3.6, 2.4, 1.6, 1.3, 'Evaluate\nFitness', 'Precision+Recall\n+ElasticNet agree', color='#D6EAF8', fontsize=8)
arrow(ax, 5.2, 3.05, 5.5, 3.05)
box(ax, 5.5, 2.4, 1.6, 1.3, 'K-Means\nClustering', 'Diversity\npreservation', color='#D6EAF8', fontsize=8)
arrow(ax, 7.1, 3.05, 7.4, 3.05)
box(ax, 7.4, 2.4, 1.5, 1.3, 'Tournament\nSelection', 'Per cluster', color='#D6EAF8', fontsize=8)
arrow(ax, 8.9, 3.05, 9.1, 3.05)
box(ax, 9.1, 2.4, 1.2, 1.3, 'Crossover\n+Mutate', 'New offspring', color='#D6EAF8', fontsize=8)

# Loop back arrow
ax.annotate('', xy=(1.7, 2.55), xytext=(10.15, 2.55),
            arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.5,
                            connectionstyle='arc3,rad=-0.35'))
ax.text(6.0, 1.9, 'next generation', ha='center', fontsize=8, color='#1A5276', style='italic')

# Output
arrow(ax, 6.0, 2.2, 6.0, 1.5)
box(ax, 3.5, 0.6, 5.0, 0.9, 'Top-k Fraud Rules', 'Human-readable interpretable rules + performance metrics', color='#FADBD8')

# Legend
legend_items = [
    mpatches.Patch(facecolor='#FDEBD0', edgecolor='#2C3E50', label='Data Processing'),
    mpatches.Patch(facecolor='#D5F5E3', edgecolor='#2C3E50', label='ElasticNet Baseline'),
    mpatches.Patch(facecolor='#D6EAF8', edgecolor='#2C3E50', label='EA Components'),
    mpatches.Patch(facecolor='#FADBD8', edgecolor='#2C3E50', label='Output'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('results/pipeline_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/pipeline_diagram.png")
