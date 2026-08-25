import json
import numpy as np
import os
import copy
import argparse
from matplotlib import pyplot as plt
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str)
parser.add_argument("--cn_file_path",
                    type=str,
                    help="Flie to be plotted.")
parser.add_argument("--num_layers",
                    type=int,
                    help="Number of layers in the model.")
args = parser.parse_args()

fig_dir = 'plot_results/figs/cn_distribution'


model_color_dict={
    "gemma-2b": "#fed070",
    "lama2-7b": "#e58b7b",
    "lama2-13b": "#97af19",
    "llama3-8b": "#386795",
}

y_points = []
tot_bag_num = 0
tot_rel_num = 0
tot_cneurons = 0
cn_bag_counter = Counter()

with open(args.cn_file_path, 'r') as f:
    cn_bag_list = json.load(f)
    for cn_bag in cn_bag_list:
        for cn in cn_bag:
            cn_bag_counter.update([cn[0]])
            y_points.append(cn[0])
    # tot_num = len(cn_bag_list)

cn_bag_counter_ori = copy.deepcopy(cn_bag_counter)

for k, v in cn_bag_counter.items():
    tot_cneurons += cn_bag_counter[k]
for k, v in cn_bag_counter.items():
    cn_bag_counter[k] /= tot_cneurons

# average # Cneurons
print('total # neurons:', tot_cneurons)

plt.figure(figsize=(8.5, 3))

x = np.array([i + 1 for i in range(args.num_layers)])
y = np.array([cn_bag_counter[i] for i in range(args.num_layers)])
# plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i for i in range(4, args.num_layers+1, 4)], labels=[i for i in range(4, args.num_layers+1, 4)], fontsize=20)
plt.yticks(np.arange(0, 0.3, 0.1), labels=[f'{np.abs(i)}%' for i in range(0, 30, 10)], fontsize=10)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=20)

plt.ylim(y.min() - 0.006, y.max() + 0.02)

plt.xlim(0.3, args.num_layers+0.7)
plt.bar(x, y, width=1.02, color=model_color_dict[args.model_name])

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
plt.savefig(os.path.join(fig_dir, f'{args.model_name}_cneurons_distribution.pdf'), dpi=100)
plt.close()