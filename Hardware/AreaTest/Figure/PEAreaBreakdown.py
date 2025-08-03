import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch



def pe_area_breakdown_plot(dc_result):

    shared_adder = {
        'A_FP16': 113.7,
        'A_BF16': 95.0,
        'A_FP32': 250.5
    }

    AxCore_Real_PE = [
        dc_result['W4']['A_FP16']['PE Total (NoAdd)'] + shared_adder['A_FP16'],    # W4-FP16
        dc_result['W4']['A_BF16']['PE Total (NoAdd)'] + shared_adder['A_BF16'],    # W4-BF16
        dc_result['W4']['A_FP32']['PE Total (NoAdd)'] + shared_adder['A_FP32'],    # W4-FP32
        dc_result['W8']['A_FP16']['PE Total (NoAdd)'] + shared_adder['A_FP16'],    # W8-FP16
        dc_result['W8']['A_BF16']['PE Total (NoAdd)'] + shared_adder['A_BF16'],    # W8-BF16
        dc_result['W8']['A_FP32']['PE Total (NoAdd)'] + shared_adder['A_FP32'],    # W8-FP32
    ]


    breakdowns_AxCore_PE = [
        
        [   # W4-FP16
            dc_result['W4']['A_FP16']['Mult'] / AxCore_Real_PE[0],                                                                                        # Approx Mult
            shared_adder['A_FP16']            / AxCore_Real_PE[0],                                                                                        # NoNorm FP Add
            dc_result['W4']['A_FP16']['SNC']  / AxCore_Real_PE[0],                                                                                        # SNC
            (dc_result['W4']['A_FP16']['PE Total (NoAdd)'] - dc_result['W4']['A_FP16']['Mult'] - dc_result['W4']['A_FP16']['SNC']) / AxCore_Real_PE[0]    # Others
        ],
        [   # W4-BF16
            dc_result['W4']['A_BF16']['Mult'] / AxCore_Real_PE[1],                                                                                        # Approx Mult
            shared_adder['A_BF16']            / AxCore_Real_PE[1],                                                                                        # NoNorm FP Add
            dc_result['W4']['A_BF16']['SNC']  / AxCore_Real_PE[1],                                                                                        # SNC
            (dc_result['W4']['A_BF16']['PE Total (NoAdd)'] - dc_result['W4']['A_BF16']['Mult'] - dc_result['W4']['A_BF16']['SNC']) / AxCore_Real_PE[1]    # Others
        ],
        [   # W4-FP32
            dc_result['W4']['A_FP32']['Mult'] / AxCore_Real_PE[2],                                                                                        # Approx Mult
            shared_adder['A_FP32']            / AxCore_Real_PE[2],                                                                                        # NoNorm FP Add
            dc_result['W4']['A_FP32']['SNC']  / AxCore_Real_PE[2],                                                                                        # SNC
            (dc_result['W4']['A_FP32']['PE Total (NoAdd)'] - dc_result['W4']['A_FP32']['Mult'] - dc_result['W4']['A_FP32']['SNC']) / AxCore_Real_PE[2]    # Others
        ],
        [   # W8-FP16
            dc_result['W8']['A_FP16']['Mult'] / AxCore_Real_PE[3],                                                                                        # Approx Mult
            shared_adder['A_FP16']            / AxCore_Real_PE[3],                                                                                        # NoNorm FP Add
            dc_result['W8']['A_FP16']['SNC']  / AxCore_Real_PE[3],                                                                                        # SNC
            (dc_result['W8']['A_FP16']['PE Total (NoAdd)'] - dc_result['W8']['A_FP16']['Mult'] - dc_result['W8']['A_FP16']['SNC']) / AxCore_Real_PE[3]    # Others
        ],
        [   # W8-BF16
            dc_result['W8']['A_BF16']['Mult'] / AxCore_Real_PE[4],                                                                                        # Approx Mult
            shared_adder['A_BF16']            / AxCore_Real_PE[4],                                                                                        # NoNorm FP Add
            dc_result['W8']['A_BF16']['SNC']  / AxCore_Real_PE[4],                                                                                        # SNC
            (dc_result['W8']['A_BF16']['PE Total (NoAdd)'] - dc_result['W8']['A_BF16']['Mult'] - dc_result['W8']['A_BF16']['SNC']) / AxCore_Real_PE[4]    # Others
        ],
        [   # W8-FP32
            dc_result['W8']['A_FP32']['Mult'] / AxCore_Real_PE[5],                                                                                        # Approx Mult
            shared_adder['A_FP32']            / AxCore_Real_PE[5],                                                                                        # NoNorm FP Add
            dc_result['W8']['A_FP32']['SNC']  / AxCore_Real_PE[5],                                                                                        # SNC
            (dc_result['W8']['A_FP32']['PE Total (NoAdd)'] - dc_result['W8']['A_FP32']['Mult'] - dc_result['W8']['A_FP32']['SNC']) / AxCore_Real_PE[5]    # Others
        ],
    ]


    # Baseline PE
    bl_pe = np.array([
        # FPC   , FPMA , FIGNA , FIGLUT
        [1029 , 412  , 247   , 225   ],
        [800  , 400  , 216   , 181   ],
        [3818 , 1108 , 420   , 461   ],
        [1022 , 412  , 448   , 212   ],
        [830  , 400  , 365   , 164   ],
        [3075 , 1108 , 703   , 456   ],
    ], dtype=object)


    # Font configuration dictionary
    font_config = {
        'title'      : 24,
        'axis_label' : 27,
        'tick_label' : 22,
        'bar_label'  : 23,
    }

    # Original data (baseline values for each main category)
    groups = ["W4-\nFP16", "W4-\nBF16", "W4-\nFP32", "W8-\nFP16", "W8-\nBF16", "W8-\nFP32"]
    categories = ['FPC', 'FPMA', 'FIGNA', 'FIGLUT', 'AxCore']

    # Sub-value percentage configuration (the 5 sub-values for each sub-category as a percentage of the total bar height)
    percentages = np.array([
        # MARK: FP16-W4
        [   #  x ,  +  ,  SNC,   O
            [0.22, 0.63, 0.00, 0.15],  # FPC
            [0.18, 0.69, 0.00, 0.13],  # FPMA
            [0.47, 0.34, 0.00, 0.19],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[0]    # AxCore
        ],
        # MARK: BF16-W4
        [   #  x ,  +  ,  SNC,   O
            [0.15, 0.71, 0.00, 0.14],  # FPC
            [0.19, 0.69, 0.00, 0.12],  # FPMA
            [0.46, 0.34, 0.00, 0.20],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[1]    # AxCore
        ],
        # MARK: FP32-W4
        [   #  x ,  +  ,  SNC,   O
            [0.42, 0.51, 0.00, 0.07],  # FPC
            [0.16, 0.75, 0.00, 0.09],  # FPMA
            [0.70, 0.30, 0.00, 0.00],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[2]    # AxCore
        ],
        # MARK: FP16-W8
        [   #  x ,  +  ,  SNC,   O
            [0.24, 0.64, 0.00, 0.12],  # FPC
            [0.18, 0.69, 0.00, 0.13],  # FPMA
            [0.55, 0.23, 0.00, 0.22],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[3]    # AxCore
        ],
        # MARK: BF16-W8
        [   #  x ,  +  ,  SNC,   O
            [0.15, 0.71, 0.00, 0.14],  # FPC
            [0.19, 0.69, 0.00, 0.12],  # FPMA
            [0.56, 0.31, 0.00, 0.13],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[4]    # AxCore
        ],
        # MARK: FP32-W8
        [   #  x ,  +  ,  SNC,   O
            [0.42, 0.51, 0.00, 0.07],  # FPC
            [0.16, 0.75, 0.00, 0.09],  # FPMA
            [0.84, 0.16, 0.00, 0.00],  # FIGNA
            [0.00, 0.00, 0.00, 1.00],  # FIGLUT
            breakdowns_AxCore_PE[5]    # AxCore
        ],
    ], dtype=object)


    # Comparing with Baseline values
    pe_comparing = np.round(
        np.array([
            [1, bl_pe[0][1]/bl_pe[0][0], bl_pe[0][2]/bl_pe[0][0], bl_pe[0][3]/bl_pe[0][0], AxCore_Real_PE[0]/bl_pe[0][0]],
            [1, bl_pe[1][1]/bl_pe[1][0], bl_pe[1][2]/bl_pe[1][0], bl_pe[1][3]/bl_pe[1][0], AxCore_Real_PE[1]/bl_pe[1][0]],
            [1, bl_pe[2][1]/bl_pe[2][0], bl_pe[2][2]/bl_pe[2][0], bl_pe[2][3]/bl_pe[2][0], AxCore_Real_PE[2]/bl_pe[2][0]],
            [1, bl_pe[3][1]/bl_pe[3][0], bl_pe[3][2]/bl_pe[3][0], bl_pe[3][3]/bl_pe[3][0], AxCore_Real_PE[3]/bl_pe[3][0]],
            [1, bl_pe[4][1]/bl_pe[4][0], bl_pe[4][2]/bl_pe[4][0], bl_pe[4][3]/bl_pe[4][0], AxCore_Real_PE[4]/bl_pe[4][0]],
            [1, bl_pe[5][1]/bl_pe[5][0], bl_pe[5][2]/bl_pe[5][0], bl_pe[5][3]/bl_pe[5][0], AxCore_Real_PE[5]/bl_pe[5][0]],
        ]),
        2
    )
    

    # Generate actual values (percentage * baseline value)
    values = []
    # for group_idx, group in enumerate(pe_compare):
    for group_idx, group in enumerate(pe_comparing):
        new_group = []
        for cat_idx, base_val in enumerate(group):
            # Get percentage configuration
            pct = percentages[group_idx][cat_idx]
            # Calculate actual values
            real_vals = [base_val * p for p in pct]
            new_group.append(real_vals)
        values.append(new_group)
    values = np.array(values, dtype=object)

    # Flatten the data (keep main category labels unchanged)
    flat_labels = []
    flat_values = []
    for group, group_vals in zip(groups, values):
        for cat, cat_vals in zip(categories, group_vals):
            flat_labels.append(f"{group}-{cat}")
            flat_values.append(cat_vals)


    sub_textures = {
        'Mul'     : {'color': '#236AB3', 'hatch': '\\\\'},
        'Add'     : {'color': '#78A7D8', 'hatch': '//'},
        'SNC'     : {'color': '#A2C6EB', 'hatch': ''},
        'Others'  : {'color': '#CCE5FE', 'hatch': 'xx'},
    }


    # Plotting configuration
    style_config = {
        'figsize'          : (11, 6),
        'bar_width'        : 0.6,
        'bar_spacing'      : 0.2,
        'label_offset'     : 0.05,
        'group_label_xbias': 0.72,      # x-axis offset for group labels
        'group_label_ybias': 0.07,      # y-axis offset for group labels
        'margin_adjust'    : 0.02
    }

    def create_stacked_subplot(ax, labels, stacked_values, ylabel, 
                            bar_width=0.6, bar_spacing=0.8, 
                            label_offset=0.05, group_label_xbias=0.42, group_label_ybias=0.15, margin=0.02):
        """Create a stacked subplot"""
        x_positions = np.arange(len(labels)) * (bar_width + bar_spacing)
        
        for x, sub_vals, label in zip(x_positions, stacked_values, labels):
            bottom = 0
            # Reverse the iteration order to maintain the sub-category correspondence
            for i in reversed(range(len(sub_vals))):  # Iterate in reverse
                sub_val = sub_vals[i]
                sub_cat = list(sub_textures.keys())[i]  # Use the newly defined names
                texture = sub_textures[sub_cat]
                
                ax.bar(x, sub_val, width=bar_width, 
                    bottom=bottom,
                    color=texture['color'],
                    hatch=texture['hatch'],
                    edgecolor='black',
                    linewidth=0.8)
                bottom += sub_val  # The bottom position is still correctly accumulated
            
            # Add sum labels
            total = sum(sub_vals)
            if total == 1:
                pass
                # ax.text(x, total + label_offset - 0.04, f"{total:.0f}",
                #         ha='center', va='bottom',
                #         fontsize=font_config['bar_label'],
                #         rotation=0)
            elif total > 0:
                ax.text(x, total + label_offset, f"{total:.2f}",
                        ha='center', va='bottom',
                        fontsize=font_config['bar_label'],
                        rotation=90)

        # Set the x-axis labels at the bottom (main category labels)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label.split('-')[-1] for label in labels],
                        fontsize=font_config['tick_label'],
                        rotation=90)
        ax.tick_params(axis='x', length=0, pad=15)
        ax.xaxis.set_ticks_position('bottom')
        
        # Add group labels at the top
        group_centers = []
        for i in range(len(groups)):
            start = i * len(categories)
            end = start + len(categories) - 1
            center = (x_positions[start] + x_positions[end]) / 2
            group_centers.append(center)
        
        for center, label in zip(group_centers, groups):
            ax.text(center+group_label_xbias, 0.73 + group_label_ybias, label,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom',
                    fontsize=font_config['tick_label'])
        
        # Add separator lines between groups
        for i in range(len(groups)-1):
            boundary = (x_positions[(i+1)*len(categories)-1] + x_positions[(i+1)*len(categories)])/2
            ax.axvline(x=boundary, color='black', linewidth=1.2,
                    ymin=0, ymax=1, clip_on=False)
        
        # # Add separator lines between categories (separation within each group of 4)
        # for i in range(len(groups)):
        #     for j in range(1, len(categories)):  # Start from 1, as no separator is needed before the first category
        #         pos_idx = i * len(categories) + j
        #         boundary = (x_positions[pos_idx-1] + x_positions[pos_idx]) / 2
        #         ax.axvline(x=boundary, color='gray', linewidth=0.8, linestyle='--',
        #                   ymin=0, ymax=1, clip_on=False, alpha=0.7)
        
        # Set border style
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        # Adjust left and right margins
        ax.set_xlim(
            x_positions[0] - bar_width/2 - margin*(x_positions[-1]-x_positions[0]),
            x_positions[-1] + bar_width/2 + margin*(x_positions[-1]-x_positions[0])
        )
        
        # Set y-axis
        ax.set_ylabel(ylabel, fontsize=font_config['axis_label'])
        ax.tick_params(axis='y', labelsize=font_config['tick_label'])
        ax.set_ylim(0, 1) 
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Create two subplots: one for the main chart, one for the legend
    fig = plt.figure(figsize=style_config['figsize'])
    gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 0.85])  # Allocate space, small area on top for legend, large area below for main chart
    legend_ax = fig.add_subplot(gs[0])  # Dedicated area for the legend
    main_ax = fig.add_subplot(gs[1])    # Main chart area

    # Hide the axes of the legend area
    legend_ax.axis('off')

    # Generate the main chart
    create_stacked_subplot(
        main_ax,
        labels=flat_labels,
        stacked_values=flat_values,
        ylabel='Normalized Area',
        bar_width=style_config['bar_width'],
        bar_spacing=style_config['bar_spacing'],
        label_offset=style_config['label_offset'],
        group_label_xbias=style_config['group_label_xbias'],
        group_label_ybias=style_config['group_label_ybias'],
        margin=style_config['margin_adjust']
    )

    # Create legend handles
    legend_elements = []
    for sub_cat, texture in sub_textures.items():
        # Create a patch with the correct texture
        patch = Patch(
            facecolor=texture['color'],
            hatch=texture['hatch'],
            edgecolor='black',
            linewidth=0.8,
            label=sub_cat
        )
        legend_elements.append(patch)

    # Create the legend in the dedicated legend area
    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        frameon=True,
        # fancybox=True,
        # shadow=False,
        fontsize=font_config['tick_label'] - 1,
        title_fontsize=font_config['tick_label'],
        # framealpha=1.0,
        labelspacing=0.5,
        handlelength=1.5,
        handletextpad=0.5,
        edgecolor='black',
    )

    # Adjust the placement
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig('/hpc/home/connect.ychen433/CodeArea/TestAE/AreaTest/Figure/pe_area_breakdown.pdf', dpi=300, bbox_inches='tight')

    print("TestAE/AreaTest/Figure/pe_area_breakdown.pdf has been generated.")
