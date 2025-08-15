import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

model_OPT = ['OPT-175B']
OPT_linear_percentage    = [0.986, 0.98, 0.95, 0.90, 0.82, 0.69]
OPT_attention_percentage = [0.014, 0.02, 0.05, 0.10, 0.18, 0.31]

model_LLAMA = ['LLaMA-3.1-405B']
LLAMA3_linear_percentage    = [0.99, 0.98, 0.97, 0.93, 0.87, 0.78]
LLAMA3_attention_percentage = [0.01, 0.02, 0.03, 0.07, 0.13, 0.22]
seq_len = [1024, 2048, 4096, 8192, 16384, 32768]
seq_len = ['1k', '2k', '4k', '8k', '16k', '32k']

# Convert data to numpy arrays for plotting
opt_linear_np = np.array(OPT_linear_percentage)
opt_attention_np = np.array(OPT_attention_percentage)
llama_linear_np = np.array(LLAMA3_linear_percentage)
llama_attention_np = np.array(LLAMA3_attention_percentage)

seq_len_labels = [str(s) for s in seq_len]  # X-axis tick labels

# Define number of sequence lengths
num_seq_lengths = len(seq_len_labels)

bar_width = 0.8  # Width of individual bars, consistent with original intent

# Create figure with a specific layout using GridSpec
fig = plt.figure(figsize=(30, 11.2), constrained_layout=True) # Added constrained_layout=True

# Define GridSpec: 2 rows (legend, plots), 2 cols for plots. height_ratios gives more space to plots.
# hspace is vertical space between the legend row and the plots row.
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 7], hspace=0.04, figure=fig)

ax_legend = fig.add_subplot(gs[0, :])  # Legend subplot: spans both columns in the first row
ax0 = fig.add_subplot(gs[1, 0])        # Subplot for OPT model (bottom-left)
ax1 = fig.add_subplot(gs[1, 1], sharey=ax0) # Subplot for LLaMA model (bottom-right), shares Y-axis with ax0

fontsize = 62
legend_fontsize = fontsize - 6

# --- Subplot 1: OPT Model (ax0) ---
x_indices_subplot = np.arange(num_seq_lengths)

ax0.bar(x_indices_subplot, opt_attention_np, bar_width, bottom=opt_linear_np, label="Attention", color='#A2C6EB', hatch='\\', zorder=2)
ax0.bar(x_indices_subplot, opt_linear_np, bar_width, label="Linear", color='#236AB3', hatch='/', zorder=2)

ax0.set_xticks(x_indices_subplot)
ax0.set_xticklabels(seq_len_labels, fontsize=fontsize)
ax0.set_ylabel('Relative OPs Percentage', fontsize=fontsize, labelpad=10)
ax0.tick_params(axis='y', labelsize=fontsize)
ax0.set_xlabel('Sequence Length', fontsize=fontsize, labelpad=10)
ax0.set_title(model_OPT[0], fontsize=fontsize, pad=20)
ax0.set_ylim(0, 1.0)
ax0.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)

# --- Subplot 2: LLaMA Model (ax1) ---
ax1.bar(x_indices_subplot, llama_attention_np, bar_width, bottom=llama_linear_np, label="Attention", color='#A2C6EB', hatch='\\', zorder=2)
ax1.bar(x_indices_subplot, llama_linear_np, bar_width, label="Linear", color='#236AB3', hatch='/', zorder=2)

ax1.set_xticks(x_indices_subplot)
ax1.set_xticklabels(seq_len_labels, fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize)
ax1.set_xlabel('Sequence Length', fontsize=fontsize, labelpad=10)
ax1.set_title(model_LLAMA[0], fontsize=fontsize, pad=20)
ax1.set_ylim(0, 1.0)
ax1.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)

# --- Create Common Legend in ax_legend ---
# Get handles and labels from one of the subplots (e.g., ax0)
handles, labels = ax0.get_legend_handles_labels()
# Use OrderedDict to ensure unique labels and maintain order
unique_legend_items = OrderedDict(zip(labels, handles))

ax_legend.legend(unique_legend_items.values(), unique_legend_items.keys(),
                #  bbox_to_anchor=(0.5, 0.8),
                 loc='center', ncol=2, fancybox=True, shadow=False, frameon=True,
                 edgecolor='black', fontsize=legend_fontsize)
ax_legend.axis('off') # Hide the axis lines, ticks, and background for the legend subplot

plt.show()

# To save the figure, uncomment the line below:
plt.savefig('figure2.pdf', dpi=300) # Updated filename