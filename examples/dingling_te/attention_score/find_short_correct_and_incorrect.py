import os
import sys
import json
import math
import torch
import pandas as pd
import numpy as np

from omnigenbench import ModelHub

MODEL_PATH = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
VALID_CSV = "/home/sw1136/OmniGenBench/examples/dingling_te/valid.csv"
TEST_CSV = "/home/sw1136/OmniGenBench/examples/dingling_te/test.csv"

# 9 tissues
TISSUES = [
    'root', 'seedling', 'leaf', 'FMI', 'FOD',
    'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
]
LABEL_COLS = [f"{t}_TE_label" for t in TISSUES]
LABEL_NAMES = ['Low', 'Medium', 'High']
LABEL2IDX = {'Low': 0, 'Medium': 1, 'High': 2}

TOP_K = 10  # 每类输出前10条（可调整）


def load_df(path: str) -> pd.DataFrame:
    """加载CSV文件"""
    df = pd.read_csv(path)
    return df


def row_to_dict(row):
    """将pandas Series转换为字典，处理NaN值"""
    d = row.to_dict()
    # 将NaN转换为字符串'nan'，与训练时一致
    for k, v in d.items():
        if pd.isna(v):
            d[k] = 'nan'
    return d


def row_true_labels(row_dict) -> list:
    """从行字典中提取真实标签"""
    true = []
    for col in LABEL_COLS:
        v = row_dict.get(col, 'nan')
        if isinstance(v, str):
            if v in LABEL2IDX:
                true.append(LABEL2IDX[v])
            else:
                true.append(-100)  # 无效标签
        else:
            true.append(-100)
    return true


def judge_correct(true_labels: list, pred_labels: np.ndarray) -> tuple:
    """判断预测是否正确"""
    valid_idx = [i for i, y in enumerate(true_labels) if y != -100]
    if not valid_idx:
        return 0, 0
    correct = sum(1 for i in valid_idx if int(pred_labels[i]) == int(true_labels[i]))
    total = len(valid_idx)
    return correct, total


def infer_split(model, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """对单个split进行推理"""
    records = []
    total_rows = len(df)
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   处理进度: {idx}/{total_rows} ({idx*100/total_rows:.1f}%)")
            
            seq = row.get('seq', '')
            if not isinstance(seq, str) or len(seq) == 0:
                continue
            
            # 转换为字典格式（与训练时一致）
            row_dict = row_to_dict(row)
            true_labels = row_true_labels(row_dict)
            
            # 跳过没有有效标签的样本
            if all(y == -100 for y in true_labels):
                continue
            
            try:
                # 使用与训练时相同的推理方式：model.inference(sequence, **row)
                outputs = model.inference(seq, **row_dict)
                predictions = outputs['predictions'].cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
                confidence = outputs['confidence'].cpu().numpy()
            except Exception as e:
                print(f"   警告: ID={row_dict.get('ID', idx)} 推理失败: {e}")
                continue
            
            correct, total = judge_correct(true_labels, predictions)
            acc = correct / total if total > 0 else 0.0
            avg_conf = float(np.mean(confidence)) if hasattr(confidence, '__len__') else float(confidence)
            
            records.append({
                'ID': row_dict.get('ID', f'{split_name}_{idx}'),
                'split': split_name,
                'seq': seq,
                'seq_len': len(seq),
                'true_labels': true_labels,
                'pred_labels': predictions.tolist(),
                'correct': correct,
                'total': total,
                'acc': acc,
                'avg_conf': avg_conf,
            })
    
    return pd.DataFrame(records)


def pick_shortest(df: pd.DataFrame, correct: bool, top_k: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    if correct:
        sub = df[df['acc'] == 1.0].copy()
    else:
        sub = df[(df['total'] > 0) & (df['acc'] < 1.0)].copy()
    if len(sub) == 0:
        return sub
    sub = sub.sort_values(['seq_len', 'acc' if not correct else 'avg_conf'], ascending=[True, True])
    return sub.head(top_k)


def main():
    print("=" * 80)
    print("查找 valid/test 中相对较短的预测正确与预测错误样本（使用微调模型）")
    print("=" * 80)

    print("加载模型...")
    model = ModelHub.load(MODEL_PATH)
    model.eval()

    print("加载数据...")
    valid_df = load_df(VALID_CSV)
    test_df = load_df(TEST_CSV)

    print("对验证集推理...")
    valid_pred = infer_split(model, valid_df, 'valid')

    print("对测试集推理...")
    test_pred = infer_split(model, test_df, 'test')

    if len(valid_pred) == 0 and len(test_pred) == 0:
        print("警告：未得到任何预测结果，可能是模型加载/推理失败。")
        return

    all_pred = pd.concat([valid_pred, test_pred], ignore_index=True)

    result = {}
    for split in ['valid', 'test']:
        split_df = all_pred[all_pred['split'] == split] if 'split' in all_pred.columns else pd.DataFrame()
        shortest_correct = pick_shortest(split_df, correct=True, top_k=TOP_K)
        shortest_incorrect = pick_shortest(split_df, correct=False, top_k=TOP_K)
        result[split] = {
            'shortest_correct': shortest_correct.to_dict(orient='records') if len(shortest_correct) else [],
            'shortest_incorrect': shortest_incorrect.to_dict(orient='records') if len(shortest_incorrect) else [],
        }

        print(f"\n[{split}] 最短且预测完全正确（前{TOP_K}条，按长度升序）：")
        if len(shortest_correct) == 0:
            print("  无")
        else:
            for idx, r in enumerate(result[split]['shortest_correct'], 1):
                print(f"  {idx:2d}. ID={r['ID']} len={r['seq_len']} acc={r['acc']:.2f} conf={r['avg_conf']:.3f}")

        print(f"\n[{split}] 最短且预测错误（前{TOP_K}条，按长度升序）：")
        if len(shortest_incorrect) == 0:
            print("  无")
        else:
            for idx, r in enumerate(result[split]['shortest_incorrect'], 1):
                print(f"  {idx:2d}. ID={r['ID']} len={r['seq_len']} acc={r['acc']:.2f} correct={r['correct']}/{r['total']}")

    out_dir = "/home/sw1136/OmniGenBench/examples/dingling_te/attention_score"
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, 'short_samples_valid_test.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_json}")

    out_csv = os.path.join(out_dir, 'all_predictions_valid_test.csv')
    all_pred.to_csv(out_csv, index=False)
    print(f"已保存: {out_csv}")


if __name__ == '__main__':
    main()
