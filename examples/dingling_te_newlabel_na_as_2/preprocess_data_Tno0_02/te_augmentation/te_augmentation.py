from omnigenbench import OmniModelForAugmentation

# Initialize augmentation model (uses a masked language model under the hood)
augmentor = OmniModelForAugmentation(
    config_or_model="yangheng/OmniGenome-186M",
    noise_ratio=0.15,     # Proportion of tokens to mask per sequence
    instance_num=3,       # Number of augmented variants per input sequence
    batch_size=32         # Batched decoding for speed
)

# Augment a single sequence
original_seq = "AUGCGAUCUCGAGCUACGUCGAUG"
augmented_sequences = augmentor.augment(seq=original_seq, k=5)

for i, aug_seq in enumerate(augmented_sequences, 1):
    print(f"Augmentation {i}: {aug_seq}")


# Augment sequences from input file and save to output file
augmentor.augment_from_file(
    input_file="train_filtered.csv",
    output_file="train_filtered_augmented.csv"
)