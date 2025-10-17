import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omnigenbench import ModelHub

# --- 步骤 1: 加载模型和定义辅助函数 ---

print("正在加载训练好的模型...")
model_path = "yangheng/ogb_te_finetuned"
#[cite_start]SS
inference_model = ModelHub.load(model_path)  # [cite: 18]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model.to(device)
inference_model.eval()
print(f"模型加载成功，并已移动到 {device} 设备。")


def get_prediction_prob(model, sequence, target_class=1):
    """获取模型对于给定序列属于 target_class 的概率。"""
    with torch.no_grad():
        inputs = model.tokenizer(sequence, return_tensors="pt").to(device)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
        return probabilities[0, target_class].item()


# --- 步骤 2: 计算每个位置的重要性得分 ---

def calculate_positional_importance(model, sequence, target_class=1):
    """
    通过扰动（突变）序列，计算每个位置对预测结果的重要性。
    """
    print(f"\n正在计算每个位置的重要性得分...")

    NUCLEOTIDES = ['A', 'U', 'C', 'G']
    seq_len = len(sequence)
    # 这次我们只需要一个一维数组来存储每个位置的重要性
    positional_scores = np.zeros(seq_len)

    # 获取基准分数
    baseline_score = get_prediction_prob(model, sequence, target_class)
    print(f"基准序列预测为'高TE'的概率: {baseline_score:.4f}")

    for i in range(seq_len):
        original_nucleotide = sequence[i]

        # 记录这个位置上所有突变导致的分数下降
        score_drops = []

        for mutated_nucleotide in NUCLEOTIDES:
            if original_nucleotide == mutated_nucleotide:
                continue

            mutated_sequence = list(sequence)
            mutated_sequence[i] = mutated_nucleotide
            mutated_sequence = "".join(mutated_sequence)

            mutated_score = get_prediction_prob(model, mutated_sequence, target_class)
            score_drops.append(baseline_score - mutated_score)

        # 将该位置上最大的分数下降值作为该位置的重要性得分
        if score_drops:
            positional_scores[i] = max(score_drops)

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{seq_len} 个位置...")

    print("重要性得分计算完成！")
    return positional_scores


# --- 步骤 3: 生成带标注的 HTML 可视化文件 ---

def generate_html_visualization(sequence, scores, filename="sequence_importance.html"):
    """
    根据重要性得分生成一个带颜色标注的HTML文件。
    """
    print(f"\n正在生成可视化文件: {filename}...")

    # 将得分归一化到 0-1 范围，以便映射到颜色
    # 添加一个极小值避免除以零
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)

    # 选择一个颜色映射，例如 'Reds'，分数越高颜色越红
    cmap = plt.get_cmap('Reds')

    html_content = """
    <html>
    <head>
    <style>
        body { font-family: monospace; font-size: 16px; word-wrap: break-word; }
        span { padding: 2px; border-radius: 3px; }
        .legend { margin-top: 20px; }
        .color-box { display: inline-block; width: 20px; height: 20px; border: 1px solid #ccc; vertical-align: middle; }
    </style>
    </head>
    <body>
    <h2>序列重要性标注</h2>
    <p>颜色越红的碱基，代表它对模型做出‘高翻译效率’这个预测的贡献越大。改变它会导致预测概率大幅下降。</p>
    <div>
    """

    # 为每个碱基生成一个带背景色的 <span> 标签
    for char, norm_score in zip(sequence, normalized_scores):
        # 获取颜色 (RGBA格式，A是透明度)
        rgba_color = cmap(norm_score)
        # 转换为 CSS 的 rgb 格式
        rgb_color = mcolors.to_hex(rgba_color)

        html_content += f'<span style="background-color: {rgb_color};">{char}</span>'

    html_content += """
    </div>
    <div class="legend">
        <b>图例:</b>
        <span style="background-color: #FFFFFF;">不重要</span>
        <span class="color-box" style="background-color: #fee0d2;"></span>
        <span class="color-box" style="background-color: #fc9272;"></span>
        <span class="color-box" style="background-color: #ef3b2c;"></span>
        <span class="color-box" style="background-color: #a50f1b;"></span>
        <span style="background-color: #a50f1b; color: white;">最重要</span>
    </div>
    </body>
    </html>
    """

    # 将HTML内容写入文件
    with open(filename, "w") as f:
        f.write(html_content)

    print(f"可视化文件已保存。请用浏览器打开 ./{filename}")


# --- 步骤 4: 执行 ---

# [cite_start]从教程中选择一个基准序列进行分析 [cite: 18]
# 截取前200个核苷酸以加快演示速度
baseline_sequence = "TAGGGCTATTACCTATCAAGGGGCTTGAACCAGTATAATTCTTGTCTTTTGGTTGCTTGATGTCGTACTACGTAGATCCTTGTACCAACGTACCCCAATACCCTCTATATCCGGTCTACGGGTATCACCCGTCGATACTCCTAATAATCTAGATTATAATAATCCTATGCTGAACCAAACAGGGCCTAAAAGAATCTATTGGTAAATTTTTTATATATATGTTTGTAGTGGCTCAAAAGCTAATAATAAAAAATATACGTTAAAAATATATTTAAATAACTTTAAAATCAAGTTCAAAAAGTTTAAATTTTGGTTATTATTTGACTTATTAGATCAACCGAGGAGGCTATTTACCACTCCCTGCCACTGCAGCACACCGACACGTGACACGTACACCCTCCCGTGACGCCGCCGCGCGTGTGCTACGGCCACACGCGCGCGCGTACACGCGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGGAGAAGCGGCG"

# 1. 计算重要性得分
positional_scores = calculate_positional_importance(inference_model, baseline_sequence)

# 2. 生成并保存HTML文件
generate_html_visualization(baseline_sequence, positional_scores)