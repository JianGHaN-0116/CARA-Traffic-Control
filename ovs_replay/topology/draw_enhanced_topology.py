"""Draw the enhanced three-switch OVS/Mininet replay topology."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 6)
ax.set_aspect("equal")
ax.axis("off")

switch_style = dict(boxstyle="round,pad=0.3", facecolor="#4472C4", edgecolor="#2F5496", linewidth=1.5)
host_style = dict(boxstyle="round,pad=0.25", facecolor="#E2EFDA", edgecolor="#548235", linewidth=1.2)
ids_style = dict(boxstyle="round,pad=0.25", facecolor="#FCE4D6", edgecolor="#C55A11", linewidth=1.2)
server_style = dict(boxstyle="round,pad=0.25", facecolor="#D6DCE4", edgecolor="#44546A", linewidth=1.2)

def draw_node(ax, x, y, label, style, fontsize=9):
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", bbox=style)

def draw_link(ax, x1, y1, x2, y2, label="", color="#404040", lw=1.5, style="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, linestyle=style))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.25, label, ha="center", va="bottom", fontsize=7, color="#404040", style="italic")

draw_node(ax, 1, 4.5, "Benign 1\n10.0.1.1", host_style)
draw_node(ax, 1, 3.0, "Benign 2\n10.0.1.2", host_style)
draw_node(ax, 1, 1.5, "Attacker 1\n10.0.1.3", ids_style)
draw_node(ax, 1, 0.0, "Attacker 2\n10.0.1.4", ids_style)
draw_node(ax, 1, 5.5, "IDS/Mirror\n10.0.1.5", ids_style)

draw_node(ax, 4, 3.0, "s1\n(Edge)", switch_style, fontsize=10)
draw_node(ax, 6.5, 3.0, "s2\n(Agg)", switch_style, fontsize=10)
draw_node(ax, 9, 3.0, "s3\n(Core)", switch_style, fontsize=10)

draw_node(ax, 10.5, 3.0, "Server\n10.0.3.2", server_style)

draw_link(ax, 1.8, 4.5, 3.2, 3.3, "100M/2ms")
draw_link(ax, 1.8, 3.0, 3.2, 3.0, "100M/2ms")
draw_link(ax, 1.8, 1.5, 3.2, 2.7, "100M/2ms")
draw_link(ax, 1.8, 0.0, 3.2, 2.4, "100M/2ms")
draw_link(ax, 1.8, 5.5, 3.2, 3.5, "100M/2ms")

draw_link(ax, 4.8, 3.0, 5.7, 3.0, "40M/5ms\n(bottleneck)", color="#C00000", lw=2.5)

draw_link(ax, 7.3, 3.0, 8.2, 3.0, "100M/2ms")
draw_link(ax, 9.8, 3.0, 9.8, 3.0, "")
ax.annotate("", xy=(9.8, 3.0), xytext=(9.8, 3.0),
            arrowprops=dict(arrowstyle="-", color="#404040", lw=1.5))
draw_link(ax, 9.8, 3.0, 9.8, 3.0, "100M/2ms")

ax.annotate("", xy=(9.8, 3.0), xytext=(9.8, 3.0),
            arrowprops=dict(arrowstyle="-", color="#404040", lw=1.5))

from matplotlib.patches import FancyArrowPatch
arrow_server = FancyArrowPatch((9.8, 3.0), (9.8, 3.0), arrowstyle="-", color="#404040", lw=1.5)

ax.annotate("100M/2ms", xy=(9.4, 3.0), fontsize=7, color="#404040", style="italic", ha="center", va="bottom")

mirror_box = mpatches.FancyBboxPatch((3.0, 4.2), 2.0, 0.8, boxstyle="round,pad=0.1",
                                      facecolor="#FFF2CC", edgecolor="#BF8F00", linewidth=1)
ax.add_patch(mirror_box)
ax.text(4.0, 4.6, "Mirror port", ha="center", va="center", fontsize=7, color="#BF8F00")

ax.annotate("", xy=(1.8, 5.3), xytext=(3.0, 4.8),
            arrowprops=dict(arrowstyle="-", color="#BF8F00", lw=1, linestyle="--"))

legend_elements = [
    mpatches.Patch(facecolor="#4472C4", edgecolor="#2F5496", label="OVS Switch"),
    mpatches.Patch(facecolor="#E2EFDA", edgecolor="#548235", label="Benign Host"),
    mpatches.Patch(facecolor="#FCE4D6", edgecolor="#C55A11", label="Attacker / IDS"),
    mpatches.Patch(facecolor="#D6DCE4", edgecolor="#44546A", label="Server"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

ax.set_title("Enhanced OVS/Mininet Replay Topology (3 switches, 7 hosts, 500 steps)",
             fontsize=11, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig(r"g:\Inet\Label-Leakage-Safe Evaluation of Detector-Assisted Traffic Control in Edge-IIoT NetworksA Calibration-Aware Detector-Assisted Traffic Control Framework for Service-Preserving Edge-IIoT Networks\figures\enhanced_ovs_topology.png", dpi=300, bbox_inches="tight")
plt.savefig(r"g:\Inet\Label-Leakage-Safe Evaluation of Detector-Assisted Traffic Control in Edge-IIoT NetworksA Calibration-Aware Detector-Assisted Traffic Control Framework for Service-Preserving Edge-IIoT Networks\figures\enhanced_ovs_topology.pdf", bbox_inches="tight")
print("Saved enhanced_ovs_topology.png and .pdf")
