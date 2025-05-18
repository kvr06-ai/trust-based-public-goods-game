import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

plt.style.use('seaborn-v0_8-pastel')

# Create figure with more vertical space
fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
ax.axis('off')

# Refined color palette
bg_color = "#FFFFFF"
text_color = "#2C3E50"
box_border = "#78909C"
arrow_color = "#546E7A"

component_colors = {
    "input": "#FFECB3",
    "process": "#B3E5FC",
    "utility": "#C8E6C9",
    "rlhf": "#D1C4E9",
    "output": "#CFD8DC",
    "parameters": "#F5F5F5"
}

fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Box parameters
box_w = 2.8
box_h = 0.8
layer_y = [11.2, 8.5, 6.3, 4.1, 1.9]
center_x = 5.0
input_x = [2.3, 5.0, 7.7] # Used for inputs and process x positions

# --- Title ---
ax.text(center_x, 13.5, "Behavioral Agent Architecture", ha='center', fontsize=18,
        fontweight='bold', color=text_color)

# --- Layer 1: Inputs ---
input_labels = ["Endowment (e)", "Others' Contributions\n(c(j))", "Observed Payoffs\n(p(j))"]
input_nodes = {}

ax.text(center_x, layer_y[0] + box_h + 0.5,
        "Environment & Agent Inputs\n(Public Goods Multiplier, MPCR = 1.5)", ha='center',
        fontsize=12, color=text_color, fontweight='normal', va='bottom', linespacing=1.4)

for i, label in enumerate(input_labels):
    rect = patches.FancyBboxPatch((input_x[i] - box_w/2, layer_y[0] - box_h/2),
                                 box_w, box_h, boxstyle="round,pad=0.3",
                                 fc=component_colors["input"], ec=box_border, lw=1.2)
    ax.add_patch(rect)
    ax.text(input_x[i], layer_y[0], label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=text_color)
    input_nodes[label] = (input_x[i], layer_y[0] - box_h/2)


# --- Layer 2: Processing ---
process_labels = ["Trust\nMechanism", "Fairness\nPerception", "Social\nPressure"]
process_x = input_x.copy() # process_x is the same as input_x for this layer
process_nodes = {}

# Layer Title for Processing - MODIFIED for better placement
# Calculate x position to be between the first and second column of processing boxes
title_process_x = (process_x[0] + process_x[1]) / 2
ax.text(title_process_x, layer_y[1] + box_h + 0.4, "Behavioral\nProcessing Layer",
        ha='center', # Center the multi-line text block at the new x
        fontsize=11, # Slightly reduced font size for better fit if needed
        color=text_color, fontweight='normal', va='bottom', linespacing=1.3)

for i, label in enumerate(process_labels):
    rect = patches.FancyBboxPatch((process_x[i] - box_w/2, layer_y[1] - box_h/2),
                                 box_w, box_h, boxstyle="round,pad=0.3",
                                 fc=component_colors["process"], ec=box_border, lw=1.2)
    ax.add_patch(rect)
    ax.text(process_x[i], layer_y[1], label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=text_color)
    process_nodes[label] = (process_x[i], layer_y[1] - box_h/2)

# --- Layer 3: Integrated Utility ---
utility_box_w = box_w * 1.1
rect = patches.FancyBboxPatch((center_x - utility_box_w/2, layer_y[2] - box_h/2),
                             utility_box_w, box_h, boxstyle="round,pad=0.3",
                             fc=component_colors["utility"], ec=box_border, lw=1.2)
ax.add_patch(rect)
ax.text(center_x, layer_y[2], "Behavioral Utility Function", ha='center', va='center',
        fontsize=10, fontweight='bold', color=text_color)

# --- Layer 4: RLHF ---
rlhf_desc = "RL + Human Feedback (RLHF)\n(Self-play & human feedback every 50 rounds)"
rect = patches.FancyBboxPatch((center_x - utility_box_w/2, layer_y[3] - box_h/2),
                             utility_box_w, box_h*1.2, boxstyle="round,pad=0.3",
                             fc=component_colors["rlhf"], ec=box_border, lw=1.2)
ax.add_patch(rect)
ax.text(center_x, layer_y[3], rlhf_desc, ha='center', va='center',
        fontsize=9, fontweight='bold', color=text_color, linespacing=1.4)


# --- Layer 5: Output ---
rect = patches.FancyBboxPatch((center_x - utility_box_w/2, layer_y[4] - box_h/2),
                             utility_box_w, box_h, boxstyle="round,pad=0.3",
                             fc=component_colors["output"], ec=box_border, lw=1.2)
ax.add_patch(rect)
ax.text(center_x, layer_y[4], "Contribution (c(i))", ha='center', va='center',
        fontsize=10, fontweight='bold', color=text_color)

# --- Arrows ---
arrow_props = dict(arrowstyle="->", color=arrow_color, lw=1.5, shrinkA=5, shrinkB=5)

# Inputs to Processing
for i in range(3):
    ax.annotate("", xy=(process_x[i], layer_y[1] + box_h/2), # Removed +0.05, should be fine now
                xytext=(input_x[i], layer_y[0] - box_h/2),
                arrowprops=arrow_props)

# Processing to Utility
for i in range(3):
    ax.annotate("", xy=(center_x, layer_y[2] + box_h/2),
                xytext=(process_x[i], layer_y[1] - box_h/2),
                arrowprops=arrow_props)

# Utility to RLHF
ax.annotate("", xy=(center_x, layer_y[3] + box_h*1.2/2),
            xytext=(center_x, layer_y[2] - box_h/2),
            arrowprops=arrow_props)

# RLHF to Output
ax.annotate("", xy=(center_x, layer_y[4] + box_h/2),
            xytext=(center_x, layer_y[3] - box_h*1.2/2),
            arrowprops=arrow_props)

# --- Feedback Loop ---
feedback_props = dict(arrowstyle="->", color=arrow_color, lw=1.5, linestyle='--',
                      connectionstyle="arc3,rad=-0.3", shrinkA=8, shrinkB=8)
ax.annotate("Feedback to Environment",
            xy=(input_x[0] - box_w/2 - 0.35 , layer_y[0]),
            xytext=(center_x - utility_box_w/2 - 0.35, layer_y[4]),
            arrowprops=feedback_props,
            ha='right', va='center', fontsize=9, fontweight='normal', color=text_color,
            xycoords='data', textcoords='data')


# --- Key Parameters Box ---
param_box_h = 0.7
param_box_y = 0.6
param_box_w_custom = 7.8
param_box = patches.FancyBboxPatch((center_x - param_box_w_custom/2, param_box_y - param_box_h/2),
                             param_box_w_custom, param_box_h, boxstyle="round,pad=0.3",
                             fc=component_colors["parameters"], ec=box_border, lw=1.2)
ax.add_patch(param_box)
param_text = (r"Key Parameters: $\lambda=0.8$ (Trust)  |  "
              r"$\alpha=0.5, \beta=0.2$ (Fairness)  |  "
              r"$\gamma=0.3$ (Social Pressure)")
ax.text(center_x, param_box_y, param_text, ha='center', va='center', fontsize=8.5,
        fontweight='normal', color=text_color)

# Set limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

plt.tight_layout(pad=0.5)
plt.savefig('behavioral_agent_architecture_minimal_v3.png', dpi=300, bbox_inches='tight')
plt.close()

print("Minimalist diagram v3 created and saved as 'behavioral_agent_architecture_minimal_v3.png'")