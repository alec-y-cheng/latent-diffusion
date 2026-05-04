import sys

def modify_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Update plot_domain_panel labels and styling
    content = content.replace('ax.set_title("Input Domain")', 
        'ax.set_title("Domain Setup", pad=36, fontsize=15, fontweight="bold",\n                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))')
    
    content = content.replace('for sp in ax.spines.values():\n        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1)',
        'for sp in ax.spines.values():\n        sp.set_visible(False)')
        
    content = content.replace('fontsize=7', 'fontsize=15')
    content = content.replace('fontsize=6', 'fontsize=9')
    
    # 2. Fully rewrite save_standardized_plot to match WindTransformer horizontal layout
    
    start_idx = content.find('def save_standardized_plot')
    if start_idx == -1:
        print("save_standardized_plot not found")
        return
        
    end_idx = content.find('\n\n', start_idx + 1000) # approximate
    # Wait, the ending is actually around line 351 where `plt.close('all')` is.
    # Let's cleanly replace the entire function string.

    return content

if __name__ == '__main__':
    pass
