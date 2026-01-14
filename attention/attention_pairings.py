import RNA  # Requires ViennaRNA Python bindings
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tqdm
import matplotlib
matplotlib.use("Agg")

# ---------- Secondary Structure Utilities ----------

def get_base_pairs(dot_bracket: str):
    stack = []
    pairs = []
    for i, char in enumerate(dot_bracket):
        if char == "(":
            stack.append(i)
        elif char == ")":
            j = stack.pop()
            pairs.append((j, i))
    return set(pairs)

def add_secondary_structure_to_items(items):
    for item in items:
        seq = item['primary']
        structure, _ = RNA.fold(seq)
        item['structure'] = structure
        item['base_pairs'] = get_base_pairs(structure)
    return items

# ---------- Feature Class for RNA Pairing ----------

class BasePairFeature:
    def get_values(self, item, from_index, to_index):
        # Offset for [CLS] token
        from_index -= 1
        to_index -= 1
        paired = (from_index, to_index) in item.get('base_pairs', set()) or \
                 (to_index, from_index) in item.get('base_pairs', set())
        return {'paired': float(paired)}

# ---------- Attention Computation Logic ----------

def compute_mean_attention(model, tokenizer, items, features, cuda=True, min_attn=0):
    model.eval()
    feature_to_weighted_sum = defaultdict(lambda: torch.zeros((model.config.num_hidden_layers,
                                                               model.config.num_attention_heads), dtype=torch.double))
    weight_total = torch.zeros((model.config.num_hidden_layers,
                                model.config.num_attention_heads), dtype=torch.double)

    with torch.no_grad():
        for item in tqdm.tqdm(items):
            tokens = tokenizer(item['primary'], return_tensors="pt")
            inputs = {k: v.to("cuda") if cuda else v for k, v in tokens.items()}

            outputs = model(**inputs, output_attentions=True, return_dict=True)
            attns = torch.stack(outputs["attentions"])  # shape: [layers, batch, heads, seq, seq]
            mask = (attns >= min_attn).squeeze(1).cpu()  # [layers, heads, seq, seq]

            weight_total += mask.long().sum((-2, -1))

            seq_len = attns.size(-1)
            for to_index in range(1, seq_len - 1):
                for from_index in range(1, seq_len - 1):
                    for feature in features:
                        feature_dict = feature.get_values(item, from_index, to_index)
                        for feature_name, value in feature_dict.items():
                            attn_scores = attns[:, 0, :, from_index, to_index].cpu()  # [layers, heads]
                            feature_to_weighted_sum[feature_name] += attn_scores
    return feature_to_weighted_sum, weight_total

# ---------- Visualization ----------

def plot_attention_map(data_tensor, title="Base Pair Attention", output_path="pair_attention_map.png"):
    # Style settings
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    data = data_tensor.numpy()
    data_normalized = data / (data.max() + 1e-9)
    row_max = data_normalized.max(axis=1)

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=(14, 1.8), wspace=0.3)

    # Main heatmap
    ax = fig.add_subplot(gs[0])
    sns.heatmap(
        data_normalized,
        ax=ax,
        cmap="YlGnBu",
        cbar_kws={"label": "% Attention"},
        xticklabels=[f"H{i+1}" for i in range(data.shape[1])],
        yticklabels=[f"L{i+1}" for i in range(data.shape[0])],
        linewidths=0.3,
        linecolor='white',
        square=True
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(title, pad=12)

    # Max bar (row-wise max values)
    #ax_bar = fig.add_subplot(gs[1])
    #ax_bar.barh(
    #    y=np.arange(data.shape[0]),
    #    width=row_max,
    #    color="#0a3c66",
    #    edgecolor="none"
    #)
    #ax_bar.set_xlim(0, 1)
    #ax_bar.set_xticks([0, 0.5, 1.0])
    #ax_bar.set_xticklabels(['0%', '50%', '100%'])
    #ax_bar.set_title("Max", fontsize=11)
    #ax_bar.invert_yaxis()
    #ax_bar.tick_params(left=False, labelleft=False, bottom=False)

    # Style cleanup
    #for spine in ax_bar.spines.values():
    #    spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved beautiful attention map to: {output_path}")

# ---------- Main Execution ----------

if __name__ == "__main__":
    from omnigenome import ModelHub, OmniSingleNucleotideTokenizer

    model_ckpt = "saved_models/"
    cuda = torch.cuda.is_available()

    # Load model and tokenizer
    model = ModelHub.load(model_ckpt)
    model.model.config.output_attentions = True
    if cuda:
        model.to("cuda")

    tokenizer = OmniSingleNucleotideTokenizer.from_pretrained(
        "../PK_Models/OmniGenome-52M/",
        trust_remote_code=True
    )

    rnas = [
        "AUGCGUAGCGCUAAGCUACGUACGUACGUA",
        "AUGCGACGUAAGCGGCUAUCGCGAUGCUG",
    ]

    items = [{'primary': rna} for rna in rnas]
    items = add_secondary_structure_to_items(items)
    features = [BasePairFeature()]

    feature_to_weighted_sum, weight_total = compute_mean_attention(
        model.model,
        tokenizer,
        items,
        features,
        cuda=cuda,
        min_attn=0.0
    )

    # Visualize attention for base-paired positions
    plot_attention_map(
        feature_to_weighted_sum["paired"] / (weight_total + 1e-9),
        title="Attention on Base-Paired Positions",
        output_path="paired_attention_heatmap.png"
    )