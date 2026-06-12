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


# ============================================================
# 1. Roofline Model
# ============================================================
def roofline():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    peak_fp32 = 67.0
    peak_bw = 2.0
    ridge = peak_fp32 / peak_bw

    x = np.logspace(-1, 4, 400)
    mem_roof = peak_bw * x
    comp_roof = np.full_like(x, peak_fp32)
    roof = np.minimum(mem_roof, comp_roof)

    ax.loglog(x, roof, "k-", linewidth=2.5, label="H100 SXM theoretical peak")
    ax.fill_between(x, 0, roof, alpha=0.08, color="gray")
    ax.axvline(ridge, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(ridge * 0.5, 0.3, "Memory-bound", ha="center", fontsize=10, color="red", alpha=0.8)
    ax.text(ridge * 3, 30, "Compute-bound", ha="center", fontsize=10, color="red", alpha=0.8)

    kernels = [
        (0.08, 0.16, "Vector Add", "#1f77b4"),
        (0.125, 0.25, "SAXPY", "#ff7f0e"),
        (0.6, 1.2, "Softmax\n(N=1024)", "#2ca02c"),
        (15, 30, "Attention", "#d62728"),
        (50, 50, "Conv 3x3\nFP16", "#9467bd"),
        (20000, 67, "GEMM\nFP32", "#8c564b"),
    ]
    for ai, flops, name, color in kernels:
        ax.plot(ai, flops, "o", markersize=9, color=color, zorder=5)
        ox, oy = 1.6, 1.6
        if ai > 100:
            ox, oy = 0.5, 0.7
        elif ai < 0.2:
            ox = 1.8
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
    fig.savefig(os.path.join(OUTPUT_DIR, "roofline.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] roofline.svg")


# ============================================================
# 2. Memory Hierarchy Pyramid
# ============================================================
def memory_pyramid():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal")

    levels = [
        (1.0,  "Register",            "~256 B/thread\n~1 cycle",       "#7b1fa2"),
        (0.8,  "Shared Mem / L1",     "~128 KB/SM\n~5-10 cycles",     "#283593"),
        (0.6,  "L2 Cache",            "~4-40 MB\n~100 cycles",         "#1b5e20"),
        (0.4,  "Global Memory (HBM)",  "~80 GB\n~2 TB/s\n~400 cycles", "#e65100"),
        (0.2,  "Host CPU DRAM",       "~512 GB\n~50 GB/s\nPCIe 4.0",  "#b71c1c"),
    ]

    for i, (width, label, detail, color) in enumerate(levels):
        y = i
        rect = plt.Rectangle((-width/2, y), width, 0.9, facecolor=color,
                             alpha=0.85, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(0, y + 0.45, label, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(0, y + 0.1, detail, ha="center", va="center", fontsize=8,
                color="white", alpha=0.9)

    ax.annotate("", xy=(0, -0.1), xytext=(0, 5.2),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="gray"))
    ax.text(-0.7, 2.5, "Capacity \u2191\nSpeed \u2193\nLatency \u2191",
            fontsize=9, color="gray", va="center", ha="center", rotation=90)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_title("GPU Memory Hierarchy", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "memory_pyramid.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] memory_pyramid.svg")


# ============================================================
# 3. Tensor Core vs CUDA Core Throughput
# ============================================================
def throughput_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    precisions = ["FP64", "FP32", "TF32", "FP16", "BF16", "FP8", "INT8"]
    tensor_tflops = [67, 67, 989, 989, 989, 1979, 1979]
    cuda_tflops   = [67, 67, 0, 0, 0, 0, 0]

    x = np.arange(len(precisions))
    width = 0.35
    bars1 = ax.bar(x - width/2, tensor_tflops, width, label="Tensor Core",
                   color="#e65100", alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cuda_tflops, width, label="CUDA Core",
                   color="#1565c0", alpha=0.9, edgecolor="white", linewidth=0.5)

    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{int(bar.get_height())}", ha="center", fontsize=9,
                    fontweight="bold", color="#e65100")
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{int(bar.get_height())}", ha="center", fontsize=9,
                    fontweight="bold", color="#1565c0")

    ax.set_xticks(x)
    ax.set_xticklabels(precisions, fontsize=11)
    ax.set_ylabel("Peak Throughput (TFLOPS)", fontsize=11)
    ax.set_title("H100 SXM: Tensor Core vs CUDA Core Throughput", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 2300)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tensor_vs_cuda_throughput.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] tensor_vs_cuda_throughput.svg")


# ============================================================
# 4. NVLink Evolution
# ============================================================
def nvlink_evolution():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gens = ["NVLink 1.0\nPascal", "NVLink 2.0\nVolta", "NVLink 3.0\nAmpere",
            "NVLink 4.0\nHopper", "NVLink 5.0\nBlackwell"]
    per_link = [20, 25, 50, 50, 100]
    aggr = [160, 300, 600, 900, 1800]

    x = np.arange(len(gens))
    width = 0.3
    ax.bar(x - width/2, per_link, width, label="Per-link BW",
           color="#e65100", alpha=0.9, edgecolor="white")
    ax.bar(x + width/2, aggr, width, label="8 GPU aggregate BW",
           color="#1565c0", alpha=0.9, edgecolor="white")

    for i, v in enumerate(per_link):
        ax.text(i - width/2, v + 20, f"{v} GB/s", ha="center", fontsize=9,
                fontweight="bold", color="#e65100")
    for i, v in enumerate(aggr):
        ax.text(i + width/2, v + 30, f"{v} GB/s", ha="center", fontsize=9,
                fontweight="bold", color="#1565c0")

    ax.set_xticks(x)
    ax.set_xticklabels(gens, fontsize=10)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax.set_title("NVLink Evolution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 2200)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "nvlink_evolution.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] nvlink_evolution.svg")


# ============================================================
# 5. Kernel Arithmetic Intensity Distribution
# ============================================================
def kernel_arithmetic_intensity():
    fig, ax = plt.subplots(figsize=(6, 4))
    kernels = ["Vector Add", "SAXPY", "Softmax\nN=1024", "Attention\nN=4096",
               "Conv 3x3\nFP16", "GEMM\nFP32"]
    ai = [0.08, 0.125, 0.6, 15, 50, 20000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    bars = ax.barh(kernels, ai, color=colors, alpha=0.85, edgecolor="white", height=0.6)
    for bar, v in zip(bars, ai):
        ax.text(v * 1.1, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}" if v < 100 else f"{int(v)}",
                va="center", fontsize=10, fontweight="bold")

    ax.axvline(33.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Ridge Point (H100)")
    ax.text(33.5, len(kernels) - 0.5, "Ridge\n33.5", color="red", fontsize=8,
            ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte) \u2014 log scale", fontsize=11)
    ax.set_title("Kernel Arithmetic Intensity Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kernel_arithmetic_intensity.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] kernel_arithmetic_intensity.svg")


# ============================================================
# 6. GPU Chip Area Allocation (Pie Chart)
# ============================================================
def chip_area_allocation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # GPU chip area
    gpu_labels = ["SM / CUDA Core\n(Computation)", "SRAM\n(L2 + Register)",
                   "HBM PHY + I/O\n(Memory IF)", "NVLink / PCIe PHY\n(Interconnect)",
                   "Fixed Function\n(Raster/Video)", "Other\n(Clock/Scan)"]
    gpu_sizes = [45, 20, 15, 8, 7, 5]
    gpu_colors = ["#e65100", "#1565c0", "#2e7d32", "#7b1fa2", "#6d4c41", "#9e9e9e"]

    wedges1, texts1, autotexts1 = ax1.pie(gpu_sizes, labels=None, autopct="%d%%",
        colors=gpu_colors, startangle=90, explode=(0.05, 0, 0, 0, 0, 0),
        textprops={"fontsize": 10, "fontweight": "bold"})
    for at in autotexts1:
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("Ampere GA100 Chip Area", fontsize=12, fontweight="bold", pad=15)

    # CPU chip area for comparison
    cpu_labels = ["Cache\n(L1+L2+L3)", "Control Logic\n(Branch Predict/OOO)",
                   "Computation\n(ALU/FPU)", "Other\n(I/F/Clock)"]
    cpu_sizes = [50, 25, 20, 5]
    cpu_colors = ["#1565c0", "#6d4c41", "#e65100", "#9e9e9e"]

    wedges2, texts2, autotexts2 = ax2.pie(cpu_sizes, labels=None, autopct="%d%%",
        colors=cpu_colors, startangle=90, explode=(0.05, 0, 0, 0),
        textprops={"fontsize": 10, "fontweight": "bold"})
    for at in autotexts2:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("CPU Die Area (Typical)", fontsize=12, fontweight="bold", pad=15)

    # Shared legend
    fig.legend(wedges1, gpu_labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(OUTPUT_DIR, "chip_area_allocation.svg"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] chip_area_allocation.svg")


# ============================================================
# 7. Communication Hierarchy (Pyramid)
# ============================================================
def comm_hierarchy():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    levels = [
        (0.9, "Intra-Node\nNVLink / IF",  "600-900 GB/s",  "#1565c0"),
        (0.6, "Inter-Node\nNVSwitch / IF", "100-400 GB/s", "#e65100"),
        (0.3, "Inter-Rack\nIB / RoCE",     "25-200 GB/s",  "#6d4c41"),
    ]

    for i, (width, label, bw, color) in enumerate(levels):
        y = i
        rect = plt.Rectangle((-width/2, y), width, 1.0, facecolor=color,
                             alpha=0.85, edgecolor="white", linewidth=2.5)
        ax.add_patch(rect)
        ax.text(0, y + 0.5, label, ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")
        ax.text(0, y + 0.15, bw, ha="center", va="center", fontsize=9,
                color="white", alpha=0.9)

    ax.text(-0.55, 1.0, "Bandwidth\n\u2193", fontsize=10, color="gray",
            ha="center", va="center")
    ax.text(0.55, 1.0, "Latency\n\u2191", fontsize=10, color="gray",
            ha="center", va="center")

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.3, 3.3)
    ax.axis("off")
    ax.set_title("GPU Communication Hierarchy", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "comm_hierarchy.svg"), dpi=150)
    plt.close(fig)
    print("  [OK] comm_hierarchy.svg")


# ============================================================
# 8. AllReduce Ring Steps
# ============================================================
def ring_allreduce():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    steps = [
        ("Step 1", [(0, 1, "C0"), (1, 2, "C1"), (2, 3, "C2"), (3, 0, "C0")]),
        ("Step 2", [(0, 1, "C1"), (1, 2, "C2"), (2, 3, "C0"), (3, 0, "C1")]),
        ("Step 3", [(0, 1, "C2"), (1, 2, "C0"), (2, 3, "C1"), (3, 0, "C2")]),
    ]
    colors = {"C0": "#e65100", "C1": "#1565c0", "C2": "#2e7d32"}

    for idx, (title, transfers) in enumerate(steps):
        ax = axes[idx]
        # Position 4 GPUs in a ring
        pos = [(0, 0.5), (-0.45, -0.15), (0, -0.5), (0.45, 0.15)]
        gpu_pos = {}
        for i, (x, y) in enumerate(pos):
            circle = plt.Circle((x, y), 0.2, facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"GPU{i}", ha="center", va="center", fontsize=9, fontweight="bold", color="#1565c0")
            gpu_pos[i] = (x, y)

        # Draw transfers
        for src, dst, chunk in transfers:
            sx, sy = gpu_pos[src]
            dx, dy = gpu_pos[dst]
            color = colors[chunk]
            ax.annotate("", xy=(dx * 0.7 + sx * 0.3, dy * 0.7 + sy * 0.3),
                        xytext=(sx * 0.7 + dx * 0.3, sy * 0.7 + dy * 0.3),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2, connectionstyle="arc3,rad=0.2"))
            midx, midy = (sx + dx) / 2, (sy + dy) / 2 - 0.12
            ax.text(midx, midy, chunk, ha="center", va="center", fontsize=8,
                    fontweight="bold", color=color)

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"ReduceScatter {title}", fontsize=10, fontweight="bold")

    fig.suptitle("Ring AllReduce (4 GPUs, 3 steps)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ring_allreduce.svg"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ring_allreduce.svg")


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
    print("=" * 50)
    print("All done")
    print("=" * 50)
