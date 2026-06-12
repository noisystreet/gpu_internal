"""
GPU Internal — batch generate matplotlib figures for Sphinx docs

Usage:
    python source/figures/generate_all.py

Output:
    source/figures/*.svg — referenced by .. figure:: in RST
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

# ---- Common color palette ------------------------------------------------
ORANGE  = "#e65100"   # NVIDIA / highlight
BLUE    = "#1565c0"   # CUDA Core / primary
GREEN   = "#2e7d32"   # success / positive
PURPLE  = "#7b1fa2"   # Tensor Core / special
BROWN   = "#6d4c41"   # fixed function / misc
GRAY    = "#9e9e9e"   # other / inactive
RED     = "#d62728"   # attention / annotation

# KERNEL_COLORS — shared between roofline and arithmetic intensity
KERNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", RED, "#9467bd", "#8c564b"]

# ---- Utility functions ---------------------------------------------------

def save_figure(fig, name, dpi=150, **kwargs):
    """Save figure to OUTPUT_DIR/name, close, and log success."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    print(f"  [OK] {name}")


def annotate_bars(ax, bars, offset=0, fmt="int", color=None):
    """Add value labels above bars in a bar chart."""
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        label = f"{int(h)}" if fmt == "int" else f"{h:.1f}"
        c = color or bar.get_facecolor()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                label, ha="center", fontsize=9, fontweight="bold", color=c)


def draw_pyramid_layers(ax, layers, height=0.9, put_detail_outside=None):
    """Draw stacked colored rectangles with title and detail text.
    
    layers: list of (width, title, detail, color)
    put_detail_outside: set of layer indices whose detail text goes to the right
    """
    if put_detail_outside is None:
        put_detail_outside = set()
    for i, (width, title, detail, color) in enumerate(layers):
        y = i
        rect = plt.Rectangle((-width / 2, y), width, height,
                             facecolor=color, alpha=0.85,
                             edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(0, y + height * 0.5, title, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i in put_detail_outside:
            # Place detail outside (to the right) for narrow layers
            ax.text(width / 2 + 0.03, y + height / 2, detail,
                    ha="left", va="center", fontsize=7, color=color,
                    fontweight="bold")
        else:
            ax.text(0, y + height * 0.12, detail, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.9)


def setup_axis_off(ax, title, xlim=(-0.8, 0.8), ylim=(-0.5, 5.5)):
    """Turn off axis and set title + limits for clean layout."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)


# ---- Figure generation functions -----------------------------------------

def roofline():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    peak_fp32, peak_bw = 67.0, 2.0
    ridge = peak_fp32 / peak_bw

    x = np.logspace(-1, 4, 400)
    roof = np.minimum(peak_bw * x, np.full_like(x, peak_fp32))

    ax.loglog(x, roof, "k-", linewidth=2.5, label="H100 SXM theoretical peak")
    ax.fill_between(x, 0, roof, alpha=0.08, color="gray")
    ax.axvline(ridge, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(ridge * 0.5, 0.3, "Memory-bound", ha="center", fontsize=10, color=RED, alpha=0.8)
    ax.text(ridge * 3, 30, "Compute-bound", ha="center", fontsize=10, color=RED, alpha=0.8)

    kernels = [
        (0.08, 0.16, "Vector Add"), (0.125, 0.25, "SAXPY"),
        (0.6, 1.2, "Softmax\n(N=1024)"), (15, 30, "Attention"),
        (50, 50, "Conv 3x3\nFP16"), (20000, 67, "GEMM\nFP32"),
    ]
    for (ai, flops, name), color in zip(kernels, KERNEL_COLORS):
        ax.plot(ai, flops, "o", markersize=9, color=color, zorder=5)
        ox, oy = (0.5, 0.7) if ai > 100 else (1.8, 1.6) if ai < 0.2 else (1.6, 1.6)
        ax.annotate(name, (ai, flops), (ai * ox, flops * oy),
                    fontsize=8.5, ha="center", color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.6))

    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title("Roofline Model \u2014 H100 SXM", fontsize=13, fontweight="bold")
    ax.set_xlim(0.08, 50000)
    ax.set_ylim(0.1, 120)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "roofline.svg")


def memory_pyramid():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal")

    levels = [
        (1.0,  "Register",            "~256 B/thread\n~1 cycle",       PURPLE),
        (0.8,  "Shared Mem / L1",     "~128 KB/SM\n~5-10 cycles",     BLUE),
        (0.6,  "L2 Cache",            "~4-40 MB\n~100 cycles",         GREEN),
        (0.4,  "Global Memory (HBM)",  "~80 GB\n~2 TB/s\n~400 cycles", ORANGE),
        (0.2,  "Host CPU DRAM",       "~512 GB\n~50 GB/s\nPCIe 4.0",  RED),
    ]
    draw_pyramid_layers(ax, levels, put_detail_outside={3, 4})

    ax.annotate("", xy=(0, -0.1), xytext=(0, 5.2),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="gray"))
    ax.text(-0.7, 2.5, "Capacity \u2191\nSpeed \u2193\nLatency \u2191",
            fontsize=9, color="gray", va="center", ha="center", rotation=90)
    setup_axis_off(ax, "GPU Memory Hierarchy")
    fig.tight_layout()
    save_figure(fig, "memory_pyramid.svg")


def throughput_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    precisions = ["FP64", "FP32", "TF32", "FP16", "BF16", "FP8", "INT8"]
    tensor = [67, 67, 989, 989, 989, 1979, 1979]
    cuda   = [67, 67,   0,   0,   0,    0,    0]

    x = np.arange(len(precisions))
    width = 0.35
    bars1 = ax.bar(x - width/2, tensor, width, label="Tensor Core",
                   color=ORANGE, alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cuda, width, label="CUDA Core",
                   color=BLUE, alpha=0.9, edgecolor="white", linewidth=0.5)

    annotate_bars(ax, bars1, offset=30, color=ORANGE)
    annotate_bars(ax, bars2, offset=30, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(precisions, fontsize=11)
    ax.set_ylabel("Peak Throughput (TFLOPS)", fontsize=11)
    ax.set_title("H100 SXM: Tensor Core vs CUDA Core Throughput", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 2300)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "tensor_vs_cuda_throughput.svg")


def nvlink_evolution():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gens = ["NVLink 1.0\nPascal", "NVLink 2.0\nVolta",
            "NVLink 3.0\nAmpere", "NVLink 4.0\nHopper", "NVLink 5.0\nBlackwell"]
    per_link = [20, 25, 50, 50, 100]
    aggr = [160, 300, 600, 900, 1800]

    x = np.arange(len(gens))
    width = 0.3
    bars1 = ax.bar(x - width/2, per_link, width, label="Per-link BW",
                   color=ORANGE, alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width/2, aggr, width, label="8 GPU aggregate BW",
                   color=BLUE, alpha=0.9, edgecolor="white")

    annotate_bars(ax, bars1, offset=20, color=ORANGE)
    annotate_bars(ax, bars2, offset=30, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(gens, fontsize=10)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax.set_title("NVLink Evolution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 2200)
    fig.tight_layout()
    save_figure(fig, "nvlink_evolution.svg")


def kernel_arithmetic_intensity():
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Vector Add", "SAXPY", "Softmax\nN=1024", "Attention\nN=4096",
              "Conv 3x3\nFP16", "GEMM\nFP32"]
    ai = [0.08, 0.125, 0.6, 15, 50, 20000]

    bars = ax.barh(labels, ai, color=KERNEL_COLORS, alpha=0.85,
                   edgecolor="white", height=0.6)
    for bar, v in zip(bars, ai):
        label = f"{v:.1f}" if v < 100 else f"{int(v)}"
        ax.text(v * 1.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, fontweight="bold")

    ax.axvline(33.5, color=RED, linestyle="--", linewidth=1.5, alpha=0.7,
               label="Ridge Point (H100)")
    ax.text(33.5, len(labels) - 0.5, "Ridge\n33.5", color=RED,
            fontsize=8, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte) \u2014 log scale", fontsize=11)
    ax.set_title("Kernel Arithmetic Intensity Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "kernel_arithmetic_intensity.svg")


def chip_area_allocation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    def _draw_pie(ax, sizes, colors, explode, title):
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct="%d%%", colors=colors,
            startangle=90, explode=explode,
            textprops={"fontsize": 10, "fontweight": "bold"})
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        return wedges

    gpu_colors = [ORANGE, BLUE, GREEN, PURPLE, BROWN, GRAY]
    _draw_pie(ax1, [45, 20, 15, 8, 7, 5], gpu_colors, (0.05, 0, 0, 0, 0, 0),
              "Ampere GA100 Chip Area")
    _draw_pie(ax2, [50, 25, 20, 5], [BLUE, BROWN, ORANGE, GRAY],
              (0.05, 0, 0, 0), "CPU Die Area (Typical)")

    gpu_labels = ["SM / CUDA Core\n(Computation)", "SRAM\n(L2 + Register)",
                   "HBM PHY + I/O\n(Memory IF)", "NVLink / PCIe PHY\n(Interconnect)",
                   "Fixed Function\n(Raster/Video)", "Other\n(Clock/Scan)"]
    fig.legend(gpu_labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, "chip_area_allocation.svg", bbox_inches="tight")


def comm_hierarchy():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    levels = [
        (0.9, "Intra-Node\nNVLink / IF",  "600-900 GB/s",  BLUE),
        (0.6, "Inter-Node\nNVSwitch / IF", "100-400 GB/s", ORANGE),
        (0.3, "Inter-Rack\nIB / RoCE",     "25-200 GB/s",  BROWN),
    ]
    draw_pyramid_layers(ax, levels, height=1.0, put_detail_outside={2})

    ax.text(-0.55, 1.0, "Bandwidth\n\u2193", fontsize=10, color="gray",
            ha="center", va="center")
    ax.text(0.55, 1.0, "Latency\n\u2191", fontsize=10, color="gray",
            ha="center", va="center")

    setup_axis_off(ax, "GPU Communication Hierarchy",
                   xlim=(-0.7, 0.7), ylim=(-0.3, 3.3))
    fig.tight_layout()
    save_figure(fig, "comm_hierarchy.svg")


def ring_allreduce():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    steps = [
        ("Step 1", [(0, 1, "C0"), (1, 2, "C1"), (2, 3, "C2"), (3, 0, "C0")]),
        ("Step 2", [(0, 1, "C1"), (1, 2, "C2"), (2, 3, "C0"), (3, 0, "C1")]),
        ("Step 3", [(0, 1, "C2"), (1, 2, "C0"), (2, 3, "C1"), (3, 0, "C2")]),
    ]
    chunk_colors = {"C0": ORANGE, "C1": BLUE, "C2": GREEN}
    # Diamond-shaped GPU positions
    gpu_xy = [(0, 0.5), (-0.45, -0.15), (0, -0.5), (0.45, 0.15)]

    for idx, (title, transfers) in enumerate(steps):
        ax = axes[idx]
        for i, (x, y) in enumerate(gpu_xy):
            circle = plt.Circle((x, y), 0.2, facecolor="#e3f2fd",
                                edgecolor=BLUE, linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"GPU{i}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=BLUE)

        for src, dst, chunk in transfers:
            sx, sy = gpu_xy[src]
            dx, dy = gpu_xy[dst]
            color = chunk_colors[chunk]
            ax.annotate("",
                xy=(dx * 0.7 + sx * 0.3, dy * 0.7 + sy * 0.3),
                xytext=(sx * 0.7 + dx * 0.3, sy * 0.7 + dy * 0.3),
                arrowprops=dict(arrowstyle="->", color=color, lw=2,
                               connectionstyle="arc3,rad=0.2"))
            mx, my = (sx + dx) / 2, (sy + dy) / 2 - 0.12
            ax.text(mx, my, chunk, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"ReduceScatter {title}", fontsize=10, fontweight="bold")

    fig.suptitle("Ring AllReduce (4 GPUs, 3 steps)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "ring_allreduce.svg", bbox_inches="tight")


# ============================================================
# 9. Warp Scheduling Timing (Greedy Oldest Ready)
# ============================================================
def warp_scheduling():
    fig, ax = plt.subplots(figsize=(8, 3.5))

    cycles = 8
    warps = [0, 1, 2, 3]
    # Greedy Oldest Ready schedule: W0→W1→W3→W3→W0→W1→W3→W3
    schedule = [(0, 0), (1, 1), (3, 2), (3, 3), (0, 4), (1, 5), (3, 6), (3, 7)]

    for w in warps:
        ax.barh(w, cycles, left=0, height=0.6, color="#f5f5f5", edgecolor="none", zorder=1)

    for warp_id, cycle in schedule:
        ax.barh(warp_id, 1, left=cycle, height=0.6, color=ORANGE, alpha=0.85,
                edgecolor="white", linewidth=1, zorder=2)
        ax.text(cycle + 0.5, warp_id, "I", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    # Warp 2 is stalled — add note
    ax.annotate("Stalled\n(mem wait)", xy=(3.5, 2), xytext=(5.5, 2.8),
                fontsize=8, color=BROWN, ha="center",
                arrowprops=dict(arrowstyle="->", color=BROWN, lw=1))

    ax.set_yticks(warps)
    ax.set_yticklabels([f"Warp {w}" for w in warps], fontsize=10)
    ax.set_xticks(range(cycles))
    ax.set_xlabel("Clock Cycle", fontsize=11)
    ax.set_title("Greedy Oldest Ready Scheduling (4 warps, 1 issue/cycle)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, cycles - 0.5)
    ax.set_ylim(-0.8, 3.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "warp_scheduling.svg")


# ============================================================
# 10. Warp Latency Hiding
# ============================================================
def warp_latency_hiding():
    fig, ax = plt.subplots(figsize=(8, 3.5))

    cycles = 8
    warps = [0, 1, 2, 3]
    # Each warp issues every 4 cycles (hides ~400 cycle memory latency with 100 warps)
    schedule = {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]}

    for w in warps:
        ax.barh(w, cycles, left=0, height=0.6, color="#f5f5f5", edgecolor="none", zorder=1)

    for w, cycles_list in schedule.items():
        for c in cycles_list:
            ax.barh(w, 1, left=c, height=0.6, color=BLUE, alpha=0.85,
                    edgecolor="white", linewidth=1, zorder=2)
            ax.text(c + 0.5, w, "I", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        # Connect instructions with arrow to show spacing
        if len(cycles_list) > 1:
            ax.annotate("", xy=(cycles_list[1], w), xytext=(cycles_list[0] + 1, w),
                        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5, linestyle="--"))

    ax.set_yticks(warps)
    ax.set_yticklabels([f"Warp {w}" for w in warps], fontsize=10)
    ax.set_xticks(range(cycles))
    ax.set_xlabel("Clock Cycle", fontsize=11)
    ax.set_title("Zero-overhead Latency Hiding (4 cycles between same-warp issues)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, cycles - 0.5)
    ax.set_ylim(-0.8, 3.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "warp_latency_hiding.svg")


# ============================================================
# 11. vGPU Architecture
# ============================================================
def vgpu_architecture():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # VM layer
    vm_y = 4.2
    for i, (name, x) in enumerate([("VM 0\nvGPU Driver", -1.2), ("VM 1\nvGPU Driver", 0), ("VM 2\nvGPU Driver", 1.2)]):
        rect = plt.Rectangle((x - 0.45, vm_y), 0.9, 0.6, facecolor="#e3f2fd",
                             edgecolor=BLUE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, vm_y + 0.3, name, ha="center", va="center", fontsize=8, fontweight="bold", color=BLUE)

    # vGPU Manager layer
    mgr_y = 2.8
    mgr = plt.Rectangle((-1.7, mgr_y), 3.4, 0.9, facecolor=ORANGE, alpha=0.85,
                        edgecolor="white", linewidth=2)
    ax.add_patch(mgr)
    ax.text(0, mgr_y + 0.45, "NVIDIA vGPU Manager (Hypervisor)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    ax.text(0, mgr_y + 0.12, "Time-slicing Scheduler  |  Memory Overcommit  |  Frame Buffer Mgmt",
            ha="center", va="center", fontsize=7, color="white", alpha=0.9)

    # GPU Hardware layer
    hw_y = 1.2
    hw = plt.Rectangle((-1.7, hw_y), 3.4, 1.0, facecolor=BROWN, alpha=0.85,
                       edgecolor="white", linewidth=2)
    ax.add_patch(hw)
    ax.text(0, hw_y + 0.5, "NVIDIA GPU Hardware", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    ax.text(0, hw_y + 0.15, "SM  |  HBM  |  NVLink", ha="center", va="center",
            fontsize=8, color="white", alpha=0.9)

    # Arrows: VM → Manager
    for x in [-1.2, 0, 1.2]:
        ax.annotate("", xy=(x, mgr_y + 0.9), xytext=(x, vm_y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # Arrow: Manager → GPU
    ax.annotate("", xy=(0, hw_y + 1.0), xytext=(0, mgr_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    # Title
    ax.set_title("vGPU Architecture", fontsize=13, fontweight="bold", pad=15, y=0.95)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.5, 5.5)
    fig.tight_layout()
    save_figure(fig, "vgpu_architecture.svg")


# ============================================================
# 12. MIG Multi-tenant Configuration (Table / Heatmap)
# ============================================================
def mig_tenants():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")

    tenants = [
        ("Tenant A", "3g.40gb", "LLM Training", "42 SM", "80 GB/s", ORANGE),
        ("Tenant B", "2g.20gb", "CPU Inference", "28 SM", "40 GB/s", BLUE),
        ("Tenant C", "1g.20gb", "GPU Inference", "14 SM", "20 GB/s", GREEN),
        ("Tenant D", "Unallocated", "—", "—", "—", GRAY),
    ]

    cols = ["Tenant", "MIG Profile", "Workload", "Compute", "Mem BW"]
    ncols = len(cols)
    nrows = len(tenants) + 1
    cell_w = 1.4
    cell_h = 0.55

    # Header
    for j, col in enumerate(cols):
        x = j * cell_w
        rect = plt.Rectangle((x, nrows * cell_h), cell_w, cell_h,
                             facecolor="#333", edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, nrows * cell_h + cell_h / 2, col,
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    # Data rows
    for i, (name, profile, workload, compute, bw, color) in enumerate(tenants):
        row = nrows - 1 - i
        bg = color if i < 3 else "#f5f5f5"
        fc = "white" if i < 3 else "#333"
        for j, val in enumerate([name, profile, workload, compute, bw]):
            x = j * cell_w
            rect = plt.Rectangle((x, row * cell_h), cell_w, cell_h,
                                 facecolor=bg, alpha=0.85 if i < 3 else 0.5,
                                 edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x + cell_w / 2, row * cell_h + cell_h / 2, val,
                    ha="center", va="center", fontsize=9, fontweight="bold" if j == 0 else "normal",
                    color=fc)

    ax.set_xlim(0, ncols * cell_w)
    ax.set_ylim(0, nrows * cell_h)
    ax.set_title("MIG Multi-Tenant Configuration (A100 80GB)", fontsize=12,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, "mig_tenants.svg")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("GPU Internal figure generator")
    print("=" * 50)
    roofline()
    memory_pyramid()
    throughput_comparison()
    nvlink_evolution()
    kernel_arithmetic_intensity()
    chip_area_allocation()
    comm_hierarchy()
    ring_allreduce()
    warp_scheduling()
    warp_latency_hiding()
    vgpu_architecture()
    mig_tenants()
    print("=" * 50)
    print("All done")
    print("=" * 50)
