import RNA  # Requires ViennaRNA Python bindings
import torch
import torch.nn.functional as F
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tqdm
import matplotlib
from bprna_utils import vienna_fold, create_vienna_dbn_file, get_bpRNA_indexes
import matplotlib.patches as mpatches
matplotlib.use("Agg")


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


def plot_position_based_attention(model, tokenizer, items, visualise_base_pairs=True,
                                  visualise_mutations=False, visualise_motifs=True,
                                  output_path="attention_by_position.png", cuda=True):
    model.eval()

    all_layers_attn = []  # Will be [layers, positions]

    with torch.no_grad():
        for item in tqdm.tqdm(items):
            tokens = tokenizer(item['primary'], return_tensors="pt")
            inputs = {k: v.to("cuda") if cuda else v for k, v in tokens.items()}
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            attns = torch.stack(outputs.attentions)  # [layers, batch, heads, seq, seq]

            # Remove CLS and SEP (assume 1 at each end)
            attn_trimmed = attns[:, 0, :, 1:-1, 1:-1]  # [layers, heads, seq, seq]

            mean_attn = attn_trimmed.mean(dim=1).mean(dim=0)  # average over heads and layers -> [seq, seq]
            # mean_attn = attn_trimmed.mean(dim=1)  # average over heads → [layers, seq, seq]

            seq_len = mean_attn.shape[-1]
            all_layers_attn.append(mean_attn)


    # Average over all sequences
    max_len1 = max(a.shape[0] for a in all_layers_attn)
    max_len2 = max(a.shape[1] for a in all_layers_attn)

    # Pad to (max_len) columns and (max_layers) rows
    padded = [
        F.pad(a, (0, max_len2 - a.shape[1], 0, max_len1 - a.shape[0]))
        for a in all_layers_attn
    ]
    # Now stack safely
    stacked = torch.stack(padded)

    # stacked = torch.stack(all_layers_attn)  # [num_seqs, layers, positions]
    mean_attn_by_layer_and_pos = stacked.mean(dim=0).cpu().numpy()  # [layers, positions]

    structure, _ = RNA.fold(items[0]['primary'])
    base_pairs = get_base_pairs(structure)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        mean_attn_by_layer_and_pos,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Mean Attention"},
        xticklabels=10,
        yticklabels=10,
    )
    ax.set_xlabel("Nucleotide Position")
    ax.set_ylabel("Nucleotide Position")
    ax.set_title("Averaged Attention for Nucleotide Positions",
                 pad=12)
    ax.set_aspect("equal")

    if visualise_base_pairs:
        for (i, j) in base_pairs:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=1.5))
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor="red", lw=1.5))

    if visualise_motifs:
        vienna_struct, _ = vienna_fold(items[0]['primary'])
        create_vienna_dbn_file("temp.dbn", items[0]['primary'], vienna_struct)
        vienna_annotation = get_bpRNA_indexes("temp.dbn")
        os.remove("temp.dbn")
        os.remove("temp.st")

        motif_color_map = {
            # "helix_stems": "red",
            "hairpin_loops": "blue",
            "bulge_loops": "green",
            "internal_loops": "orange",
            "multi_loops": "purple",
            "external_loops": "gray",
            "dangling_ends": "cyan",
            # "segments": "brown"
        }
        if len(items[0]["primary"]) > 100:
            line_width = .12
        elif len(items[0]['primary']) > 50:
            line_width = .25
        else:
            line_width = 1.5
        for motif, indexes in vienna_annotation.items():
            if motif == "helix_stems" or motif == "segments":
                continue # Skip Helix Stems and Segments
            for i, index in enumerate(indexes):
                for j, index2 in enumerate(indexes):
                    ax.add_patch(plt.Rectangle((indexes[i], indexes[j]), 1, 1, fill=False,
                                               edgecolor=motif_color_map[motif], lw=line_width))
        legend_handles = [
            mpatches.Patch(color=color, label=motif)
            for motif, color in motif_color_map.items()
            if len(vienna_annotation[motif]) > 0
        ]

        # Add the legend to the axes
        ax.legend(handles=legend_handles, title="Motifs", loc="center left", fontsize='small',
                  bbox_to_anchor=(1.25,1))

    if visualise_mutations:
        for mutation_index in visualise_mutations:
            ax.add_patch(plt.Rectangle((mutation_index, mutation_index), 1, 1, fill=False, edgecolor="red", lw=1.5))

    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"📈 Saved position-based attention plot to: {output_path}")


if __name__ == "__main__":
    from omnigenome import ModelHub, OmniSingleNucleotideTokenizer

    model_name = "OmniGenome-52M"

    # model_ckpt = "OmniGenome-1.5_bpRNA_FineTuned/"
    model_ckpt = "saved_models/"

    cuda = torch.cuda.is_available()

    # Load Fine-Tuned model and tokenizer
    model = ModelHub.load(model_ckpt)
    model.model.config.output_attentions = True
    if cuda:
        model.to("cuda")

    tokenizer = OmniSingleNucleotideTokenizer.from_pretrained(
        "saved_models/",
        trust_remote_code=True
    )

    rnas = [
        "ACCUACACCCCAUGCGCGCUGACCUCCGUCAGCACCAUGCCCAGGCAGCUUCGGUCUUCAAAGCUGCGGCGCGGUUCUUCCGCGUUGACCGAUCCAGUCCGCUUCGCAUCGCUCCUGCGGACGAUCGCCAUGUCCUGGCGACGAUGC",
        "GCAGCAUUGAUUAAUCUCAAUUUGUAAAUGUGAGCGAUUUUAAAGUAUUUGACGCACUCACUUUGCAAUUGGAGAUUGCUCGAGAUUGC",
        "AUGCGUAGCGCUAAGCUACGUACGUACGUA",
    ]

    dnas = [
        "ATGCCCTAGGTCGAACTGGATGCTAGCTAGGTCAGGCTAG",  # Non-Mutated
        "ATGCCCTAGGTCGAAAAAGATGCTAGCTAGGTCAGGCTAG",  # SNP Mutated
        # "ATGCCCTAGGTCGAGGATGCTAGCTAGGTCAGGCTAG",  # Deletion Mutated
        # "ATGCCCTAGGTCTGAGAACTGGATGCTAGCTAGGTCAGGCTAG",  # Insertion Mutated
    ]

    items = [{'primary': rna} for rna in rnas]
    # items = [{'primary': dna} for dna in dnas]
    # [4] represents the mutated, [13,14,15] represents the deleted, [10,11,12] represents the inserted
    # mutations = [[15,16,17], [15,16,17]]

    for i, item in enumerate(items):
        plot_position_based_attention(
            model.model,
            tokenizer,
            [item],
            visualise_base_pairs=True,
            visualise_motifs=False,
            # visualise_mutations=mutations[i],
            output_path=f"rna_attention_by_position{i}_{model_name}.png",
            cuda=cuda
        )
