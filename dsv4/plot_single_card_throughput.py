import os
import re

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


CSV_PATH = "./data/summary_best.csv"
OUTPUT_DIR = "./single_card_throughput_plots"


def setup_font():
    for font_path in ["/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [
        next((font for font in font_candidates if font in available_fonts), "DejaVu Sans")
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13


def format_seq_len(length):
    length = int(length)
    if length >= 1024 * 1024:
        return f"{length // (1024 * 1024)}M"
    if length >= 1024:
        return f"{length // 1024}K"
    return str(length)


def trailing_p_size(value):
    match = re.search(r"_(\d+)P$", str(value))
    return int(match.group(1)) if match else None


def hardware_type(value):
    return re.sub(r"_\d+P$", "", str(value))


def safe_name(value):
    return (
        str(value)
        .replace(" ", "_")
        .replace(",", "_")
        .replace("<=", "LE")
        .replace("/", "_")
    )


setup_font()
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
required_cols = [
    "model",
    "chip",
    "hardware_name",
    "seq_len",
    "latency_constraint_ms",
    "single_card_throughput_tps",
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"CSV缺少必要列: {missing_cols}")

df["supernode_size"] = df["chip"].map(trailing_p_size)
df["supernode_size"] = df["supernode_size"].fillna(df["hardware_name"].map(trailing_p_size))
df["hardware_type"] = df["hardware_name"].map(hardware_type)
df["single_card_throughput_tps"] = pd.to_numeric(
    df["single_card_throughput_tps"], errors="coerce"
)

df = df.dropna(
    subset=[
        "model",
        "seq_len",
        "latency_constraint_ms",
        "supernode_size",
        "hardware_type",
        "single_card_throughput_tps",
    ]
).copy()
df["supernode_size"] = df["supernode_size"].astype(int)

group_cols = ["model", "latency_constraint_ms", "seq_len"]
hardware_types = sorted(df["hardware_type"].unique())
color_map = dict(zip(hardware_types, plt.cm.tab10.colors))

for (model, tpot, seq_len), group in df.groupby(group_cols):
    model_name = str(model)[0].upper() + str(model)[1:]
    seq_str = format_seq_len(seq_len)
    title = f"{model_name}@KV={seq_str}, TPOT<={tpot}ms"

    fig, ax = plt.subplots(figsize=(9, 6))

    for hw_type in hardware_types:
        line_data = group[group["hardware_type"] == hw_type]
        if line_data.empty:
            continue

        line_data = (
            line_data.groupby("supernode_size", as_index=False)["single_card_throughput_tps"]
            .max()
            .sort_values("supernode_size")
        )
        ax.plot(
            line_data["supernode_size"],
            line_data["single_card_throughput_tps"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color_map[hw_type],
            label=hw_type,
        )

    sizes = sorted(group["supernode_size"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size}P" for size in sizes])
    ax.set_xlabel("超节点规模", fontsize=14)
    ax.set_ylabel("Single Card 吞吐 (tps)", fontsize=14)
    ax.set_title(title, fontsize=16, pad=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(title="硬件选型", fontsize=11, title_fontsize=12)
    ax.set_axisbelow(True)

    plt.tight_layout()
    file_name = (
        f"single_card_throughput_{safe_name(model)}_KV{seq_str}_TPOT_LE{safe_name(tpot)}ms.png"
    )
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"已生成 {df.groupby(group_cols).ngroups} 张 single card 吞吐折线图，目录: {OUTPUT_DIR}")
