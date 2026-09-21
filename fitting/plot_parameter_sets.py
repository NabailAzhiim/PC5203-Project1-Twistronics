"""
plot_parameter_sets.py

Final plotting script for the directory structure:

m_variation/
    dl=0.1_L=0.1/
        m=2_fitting_dl=0.1_eon=0.0_t0=1.0_L=0.1.txt
        m=3_fitting_dl=0.1_eon=0.0_t0=1.0_L=0.1.txt
        ...
    dl=0.1_L=0.3/
        ...
    dl=0.3_L=0.5/
        ...

Each fitting logfile is assumed to contain, near its beginning:

    m=3
    eon=0.0
    t0=1.0
    l_decay=0.1
    dl=0.1

    eps0 = ...
    t1   = ...
    t2   = ...

    RMSE=...

Only these summary values are parsed. The long k-point table later in the
logfile is deliberately ignored.

The script produces three sets of plots:

SET 1: variation with m
    epsilon_0(m), |t1|(m), |t2|(m), RMSE(m)
    at fixed dl = 0.1
    one curve for every available L

SET 2: variation with L
    epsilon_0(L), |t1|(L), |t2|(L), RMSE(L)
    at fixed dl = 0.1
    curves for selected m values (default: m=4,7)

SET 3: variation with layer distance dl
    epsilon_0(dl), |t1|(dl), |t2|(dl), RMSE(dl)
    at fixed L = 0.5
    curves for selected m values (default: m=4,7)

When new data points are added to m_variation/, the x-axis points and, for
SET 1, new L curves are detected automatically.

Dependencies:
    numpy
    pandas
    matplotlib
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# USER SETTINGS
# ======================================================================

DATA_ROOT = Path("m_variation")
OUTPUT_ROOT = Path("parameter_plots")

DPI = 200
FIGSIZE = (7.2, 5.0)
MARKER = "o"
LINEWIDTH = 1.8
MARKERSIZE = 6
SHOW_GRID = True

# Numerical tolerance for matching floating-point parameter values.
ATOL = 1e-10


# ----------------------------------------------------------------------
# SET 1: m variation
# ----------------------------------------------------------------------
SET1_FIXED_DL = 0.1

# None means: use every L found at SET1_FIXED_DL.
# Example to restrict:
SET1_L_VALUES = [0.3, 0.5, 0.75, 1.0, 2.0]
# SET1_L_VALUES = None


# ----------------------------------------------------------------------
# SET 2: L variation
# ----------------------------------------------------------------------
SET2_FIXED_DL = 0.1

# None means: use every available m.
# SET2_M_VALUES = [4, 7]
SET2_M_VALUES = None


# ----------------------------------------------------------------------
# SET 3: dl variation
# ----------------------------------------------------------------------
SET3_FIXED_L = 0.5

# None means: use every available m.
# SET3_M_VALUES = [4, 7]
SET3_M_VALUES = None


# ======================================================================
# LOGFILE PARSER
# ======================================================================

NUMBER = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"


def read_value(text, key):
    """Read `key = number` or `key=number` from the logfile summary."""
    match = re.search(
        rf"^\s*{re.escape(key)}\s*=\s*({NUMBER})",
        text,
        flags=re.MULTILINE,
    )
    if match:
        return float(match.group(1))
    return np.nan


def parse_fitting_log(path):
    """
    Parse only the summary part of one fitting logfile.

    We split before '# k_index' so the large per-k table is never parsed.
    """
    text = path.read_text(errors="replace")
    summary = text.split("# k_index", 1)[0]

    record = {
        "m": read_value(summary, "m"),
        "eon": read_value(summary, "eon"),
        "t0": read_value(summary, "t0"),
        "L": read_value(summary, "l_decay"),
        "dl": read_value(summary, "dl"),
        "e0": read_value(summary, "eps0"),
        "t1": read_value(summary, "t1"),
        "t2": read_value(summary, "t2"),
        "RMSE": read_value(summary, "RMSE"),
        "source_file": str(path),
    }

    required = ("m", "L", "dl", "e0", "t1", "t2", "RMSE")
    missing = [key for key in required if np.isnan(record[key])]

    if missing:
        raise ValueError(
            f"Missing {missing} in fitting logfile:\n  {path}"
        )

    record["m"] = int(round(record["m"]))
    return record


def load_all_fitting_logs():
    """
    Read exactly the fitting summary files, not the PNG files.

    Expected filename example:
        m=3_fitting_dl=0.1_eon=0.0_t0=1.0_L=0.1.txt
    """
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Cannot find directory: {DATA_ROOT.resolve()}"
        )

    files = sorted(DATA_ROOT.glob("dl=*_L=*/*_fitting_*.txt"))

    if not files:
        raise RuntimeError(
            "No fitting logfiles found.\n"
            "Expected files like:\n"
            "m_variation/dl=0.1_L=0.5/"
            "m=4_fitting_dl=0.1_eon=0.0_t0=1.0_L=0.5.txt"
        )

    records = []
    failures = []

    for path in files:
        try:
            records.append(parse_fitting_log(path))
        except Exception as exc:
            failures.append((path, exc))

    if failures:
        print("\nWARNING: some fitting logs could not be parsed:")
        for path, exc in failures:
            print(f"  {path}: {exc}")

    if not records:
        raise RuntimeError("No fitting logfiles could be parsed.")

    df = pd.DataFrame(records)

    # Retain signed hoppings and add magnitudes for plotting.
    df["abs_t1"] = df["t1"].abs()
    df["abs_t2"] = df["t2"].abs()

    # A few useful derived quantities for later analysis.
    df["abs_t2_over_t1"] = np.where(
        df["abs_t1"] > 0.0,
        df["abs_t2"] / df["abs_t1"],
        np.nan,
    )

    df["aM_over_a"] = np.sqrt(df["m"].astype(float)**2 + 1.0)
    df["aM_over_L"] = df["aM_over_a"] / df["L"]

    df = df.sort_values(["dl", "L", "m"]).reset_index(drop=True)
    return df


# ======================================================================
# SELECTION HELPERS
# ======================================================================

def close_to(series, value):
    return np.isclose(
        series.astype(float),
        float(value),
        rtol=0.0,
        atol=ATOL,
    )


def select_values(data, column, values):
    """
    If values is None, return all data.
    Otherwise retain rows whose column matches any requested value.
    """
    if values is None:
        return data

    mask = np.zeros(len(data), dtype=bool)

    for value in values:
        if np.issubdtype(data[column].dtype, np.number):
            mask |= close_to(data[column], value)
        else:
            mask |= data[column].eq(value).to_numpy()

    return data.loc[mask]


# ======================================================================
# GENERIC PLOT FUNCTION
# ======================================================================

def plot_grouped(
    data,
    x,
    y,
    group,
    xlabel,
    ylabel,
    title,
    output_path,
):
    if data.empty:
        print(f"WARNING: no data for {output_path}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for group_value, curve in data.groupby(group, sort=True):
        curve = curve.sort_values(x)

        ax.plot(
            curve[x],
            curve[y],
            marker=MARKER,
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            label=f"{group}={group_value:g}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if SHOW_GRID:
        ax.grid(True, alpha=0.25)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)

    print(f"Wrote {output_path}")


# Quantity, y-axis label, output filename stem
PLOT_QUANTITIES = [
    ("e0", r"$\epsilon_0$", "e0"),
    ("abs_t1", r"$|t_1|$", "abs_t1"),
    ("abs_t2", r"$|t_2|$", "abs_t2"),
    ("RMSE", "RMSE", "RMSE"),
]


# ======================================================================
# SET 1: epsilon0(m), |t1|(m), |t2|(m), RMSE(m)
# ======================================================================

def plot_m_variation(df):
    outdir = OUTPUT_ROOT / "set1_m_variation"
    outdir.mkdir(parents=True, exist_ok=True)

    data = df.loc[close_to(df["dl"], SET1_FIXED_DL)].copy()
    data = select_values(data, "L", SET1_L_VALUES)

    if data.empty:
        print("\nSET 1: no matching data.")
        return

    print("\nSET 1 -- m variation")
    print(f"fixed dl = {SET1_FIXED_DL}")
    print("m values =", sorted(data["m"].unique()))
    print("L curves =", sorted(data["L"].unique()))

    for y, ylabel, stem in PLOT_QUANTITIES:
        plot_grouped(
            data=data,
            x="m",
            y=y,
            group="L",
            xlabel=r"$m$",
            ylabel=ylabel,
            title=rf"{ylabel} vs $m$ ($d_l={SET1_FIXED_DL}$)",
            output_path=outdir / f"{stem}_vs_m.png",
        )


# ======================================================================
# SET 2: epsilon0(L), |t1|(L), |t2|(L), RMSE(L)
# ======================================================================

def plot_L_variation(df):
    outdir = OUTPUT_ROOT / "set2_L_variation"
    outdir.mkdir(parents=True, exist_ok=True)

    data = df.loc[close_to(df["dl"], SET2_FIXED_DL)].copy()
    data = select_values(data, "m", SET2_M_VALUES)

    if data.empty:
        print("\nSET 2: no matching data.")
        return

    print("\nSET 2 -- L variation")
    print(f"fixed dl = {SET2_FIXED_DL}")
    print("m curves =", sorted(data["m"].unique()))
    print("L values =", sorted(data["L"].unique()))

    for y, ylabel, stem in PLOT_QUANTITIES:
        plot_grouped(
            data=data,
            x="L",
            y=y,
            group="m",
            xlabel=r"$L$",
            ylabel=ylabel,
            title=rf"{ylabel} vs $L$ ($d_l={SET2_FIXED_DL}$)",
            output_path=outdir / f"{stem}_vs_L.png",
        )


# ======================================================================
# SET 3: epsilon0(dl), |t1|(dl), |t2|(dl), RMSE(dl)
# ======================================================================

def plot_dl_variation(df):
    outdir = OUTPUT_ROOT / "set3_dl_variation"
    outdir.mkdir(parents=True, exist_ok=True)

    data = df.loc[close_to(df["L"], SET3_FIXED_L)].copy()
    data = select_values(data, "m", SET3_M_VALUES)

    if data.empty:
        print("\nSET 3: no matching data.")
        return

    print("\nSET 3 -- layer-distance variation")
    print(f"fixed L = {SET3_FIXED_L}")
    print("m curves =", sorted(data["m"].unique()))
    print("dl values =", sorted(data["dl"].unique()))

    for y, ylabel, stem in PLOT_QUANTITIES:
        plot_grouped(
            data=data,
            x="dl",
            y=y,
            group="m",
            xlabel=r"$d_l$",
            ylabel=ylabel,
            title=rf"{ylabel} vs $d_l$ ($L={SET3_FIXED_L}$)",
            output_path=outdir / f"{stem}_vs_dl.png",
        )


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_all_fitting_logs()

    # Keep a convenient master table of everything that was parsed.
    csv_path = OUTPUT_ROOT / "all_fitting_parameters.csv"
    df.to_csv(csv_path, index=False)

    print(f"Read {len(df)} fitting logfiles.")
    print(f"Wrote {csv_path}")

    print("\nAll data available:")
    print("m  =", sorted(df["m"].unique()))
    print("L  =", sorted(df["L"].unique()))
    print("dl =", sorted(df["dl"].unique()))

    plot_m_variation(df)
    plot_L_variation(df)
    plot_dl_variation(df)


if __name__ == "__main__":
    main()
