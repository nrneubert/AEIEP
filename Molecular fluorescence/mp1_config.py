import matplotlib as mpl

# Define font size
fs = 14

# Update rcParams
mpl.rcParams.update({
    "text.usetex": True,                   # Use LaTeX for all text
    "font.family": "serif",                # Use a serif font
    "font.serif": ["Times", "Computer Modern Roman"],  # Common physics fonts
    "axes.labelsize": fs,                   # Axis labels size
    "axes.titlesize": fs,                   # Title size
    "font.size": fs,                        # General font size
    "legend.fontsize": fs-2,                # Legend font size
    "xtick.labelsize": fs-2,                # X-axis tick size
    "ytick.labelsize": fs-2,                # Y-axis tick size
    "axes.labelpad": 8,
    "axes.titlepad": 8,
    "axes.linewidth": 1.0,                   # Thickness of axes
    "legend.frameon": False,                 # No legend frame
    "xtick.direction": "in",                  # Ticks inside the plot
    "ytick.direction": "in",
    "xtick.major.size": 5,                    # Major tick size
    "ytick.major.size": 5,
    "xtick.minor.size": 3,                    # Minor tick size
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,              # Show minor ticks
    "ytick.minor.visible": True,
    "grid.linestyle": "--",                   # Dashed grid lines
    "grid.alpha": 0.6,                        # Grid transparency
    "grid.color": "gray",                     # Grid color
})
