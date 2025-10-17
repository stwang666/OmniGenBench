import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omnigenbench import ModelHub

# --- 步骤 1: 加载训练好的模型和分词器 ---

print("正在加载训练好的模型...")
# 注意：这里我们加载一个已经微调好的模型。
# 您也可以替换为您自己训练并保存的本地模型路径，例如 "ogb_te_finetuned"
model_path = "yangheng/ogb_te_finetuned"
# 使用 ModelHub 加载模型，它会自动处理分词器等组件
inference_model = ModelHub.load(model_path)

# 确定运行设备 (GPU或CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model.to(device)
inference_model.eval()  # 设置为评估模式
print(f"模型加载成功，并已移动到 {device} 设备。")


# --- 步骤 2: 定义辅助函数，用于获取特定类别的预测概率 ---

def get_prediction_prob(model, sequence, target_class=1):
    """
    获取模型对于给定序列属于 target_class 的概率。
    target_class=1 代表 "高翻译效率"
    """
    with torch.no_grad():
        inputs = model.tokenizer(sequence, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # 从原始分数 (logits) 计算概率
        probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
        # 返回目标类别的概率
        return probabilities[0, target_class].item()


# --- 步骤 3: 实现“计算机内定点突变”核心逻辑 ---

def in_silico_mutagenesis(model, sequence, target_class=1):
    """
    对序列进行定点突变，并计算每个突变对模型预测概率的影响。
    """
    print(f"\n正在对序列进行计算机内定点突变分析...")

    # 定义所有可能的核苷酸 (RNA序列)
    NUCLEOTIDES = ['A', 'U', 'C', 'G']
    seq_len = len(sequence)

    # 初始化一个矩阵来存储重要性得分
    # 形状: (4, seq_len)，4代表4种核苷酸
    importance_scores = np.zeros((len(NUCLEOTIDES), seq_len))

    # 1. 获取原始序列的基准分数
    baseline_score = get_prediction_prob(model, sequence, target_class)
    print(f"基准序列预测为'高TE'的概率: {baseline_score:.4f}")

    # 2. 遍历序列中的每一个位置
    for i in range(seq_len):
        original_nucleotide = sequence[i]

        # 3. 在当前位置尝试突变为所有可能的核苷酸
        for j, mutated_nucleotide in enumerate(NUCLEOTIDES):
            # 如果突变后和原来一样，则影响为0，跳过
            if original_nucleotide == mutated_nucleotide:
                continue

            # 创建突变后的序列
            mutated_sequence = list(sequence)
            mutated_sequence[i] = mutated_nucleotide
            mutated_sequence = "".join(mutated_sequence)

            # 4. 获取突变后的分数
            mutated_score = get_prediction_prob(model, mutated_sequence, target_class)

            # 5. 计算分数变化，并存入矩阵
            # 这个值代表了该突变对预测的“破坏”程度
            score_change = baseline_score - mutated_score
            importance_scores[j, i] = score_change

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{seq_len} 个位置...")

    print("定点突变分析完成！")
    return importance_scores, baseline_score


# --- 步骤 4: 执行实验并可视化结果 ---

# 从教程中选择一个基准序列进行分析
# 可以选择high TE的序列，这样更能说明问题
baseline_sequence = "AGAGAGAAGGGAGGAGGAGGGGAGGTCCGGGAGGAAGAAGAAAGGGAAGAGATGACATGTGGGGCCCACACGTCATTGGGCCCCCACAATTTTGTGTGTGCGAATGAGAAATGGGTCCCACATATATGTTTTTAATTGAAATGTCACATCAGCGCCACGTCAGCACACAGACTAGGTCAACACTGCCACGTCAGTGTAACTGCCCTCCAAAACCACCAAGGGAGTAAAATTGCACCGGTTTCAATAGTTTGGGGGTGAAGATATCCAATTTTGCGGTTTAGGGTCATGGATTAGATTCGGGCCACTTTTTAAGGGAGTAAAAGTGAACTTATTCCATAAATTTAACACTCCAAAGCTGAAACAGGCAGGCCCACGGATTGGAAGCCCAGGAGGCCAGGTGGGAGTACAATAGGGAATTCGCATGGAGGTAGGGTTGCAGGGTAGGGATGGGATTTCACATCCGAAGGGATCAGATCTGATCTGAGGAGTCAAATACTCCT"

# 执行实验
importance_scores, baseline_score = in_silico_mutagenesis(inference_model, baseline_sequence)

# 可视化结果
print("\n正在生成重要性热力图...")
plt.figure(figsize=(20, 5))
sns.heatmap(
    importance_scores,
    cmap="RdBu_r",  # 使用红-白-蓝的色谱，红色代表影响大
    center=0,  # 将颜色中心设为0
    yticklabels=['A', 'U', 'C', 'G'],
    linewidths=.5
)
plt.title(f"In-Silico Mutagenesis Saliency Map (Baseline Score={baseline_score:.2f})")
plt.xlabel("Sequence Position")
plt.ylabel("Mutated To")
plt.show()