# -*- coding: utf-8 -*-
# file: advanced_perturbation_analysis.py
# time: 10:30 19/10/2025
# author: Advanced Sequence Perturbation Analysis
# Description: Comprehensive perturbation analysis with multiple visualization options

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import pandas as pd

from omnigenbench import (
    ModelHub,
    OmniTokenizer,
)

# 导入自定义的 Dataset 类
import sys
sys.path.append(os.path.dirname(__file__))
from triclass_te import TriClassTEDataset


class SequencePerturbationAnalyzer:
    """序列扰动分析器"""
    
    def __init__(self, model, tissue_names=None):
        """
        初始化分析器
        
        Args:
            model: 训练好的模型
            tissue_names: 组织名称列表
        """
        self.model = model
        self.model.eval()
        
        if tissue_names is None:
            self.tissue_names = [
                'root', 'seedling', 'leaf', 'FMI', 'FOD',
                'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
            ]
        else:
            self.tissue_names = tissue_names
        
        self.nucleotides = ['A', 'T', 'C', 'G']
        self.label_names = ['Low', 'Medium', 'High']
    
    def substitute_nucleotide(self, sequence, position, new_nucleotide=None):
        """
        在指定位置替换核苷酸
        
        Args:
            sequence: DNA序列
            position: 位置索引
            new_nucleotide: 新的核苷酸（如果为None则随机选择）
        
        Returns:
            扰动后的序列
        """
        seq_list = list(sequence)
        original_nuc = seq_list[position].upper()
        
        if new_nucleotide is None:
            # 随机选择一个不同的核苷酸
            alternative_nucs = [n for n in self.nucleotides if n != original_nuc]
            new_nucleotide = random.choice(alternative_nucs)
        
        seq_list[position] = new_nucleotide
        return ''.join(seq_list)
    
    def delete_nucleotide(self, sequence, position):
        """删除指定位置的核苷酸"""
        return sequence[:position] + sequence[position+1:]
    
    def insert_nucleotide(self, sequence, position, nucleotide=None):
        """在指定位置插入核苷酸"""
        if nucleotide is None:
            nucleotide = random.choice(self.nucleotides)
        return sequence[:position] + nucleotide + sequence[position:]
    
    def get_prediction(self, sample_data):
        """获取模型预测"""
        with torch.no_grad():
            outputs = self.model.inference(sample_data)
            predictions = outputs['predictions'].cpu().numpy()
            probabilities = outputs['probabilities'].cpu().numpy()
        return predictions, probabilities
    
    def calculate_prediction_change_score(self, original_pred, perturbed_pred, 
                                         original_probs, perturbed_probs):
        """
        计算预测变化分数（综合考虑类别变化和概率变化）
        
        Returns:
            change_score: 变化分数
            category_change: 类别变化数量
            prob_change: 概率变化幅度
        """
        # 类别变化
        category_change = (original_pred != perturbed_pred).sum()
        
        # 概率变化（KL散度）
        prob_change = 0
        for i in range(len(original_pred)):
            # 避免log(0)
            eps = 1e-10
            orig_p = original_probs[i] + eps
            pert_p = perturbed_probs[i] + eps
            # KL散度
            kl_div = np.sum(orig_p * np.log(orig_p / pert_p))
            prob_change += kl_div
        
        prob_change /= len(original_pred)
        
        # 综合分数（类别变化权重更高）
        change_score = 0.7 * (category_change / len(original_pred)) + 0.3 * prob_change
        
        return change_score, category_change, prob_change
    
    def analyze_position_importance(self, sequence, sample_data, 
                                   position, num_perturbations=10,
                                   perturbation_type='substitution'):
        """
        分析单个位置的重要性
        
        Args:
            sequence: 输入序列
            sample_data: 样本数据
            position: 要分析的位置
            num_perturbations: 扰动次数
            perturbation_type: 扰动类型 ('substitution', 'deletion', 'insertion')
        
        Returns:
            importance_metrics: 重要性指标字典
        """
        # 获取原始预测
        original_pred, original_probs = self.get_prediction(sample_data)
        
        change_scores = []
        category_changes = []
        prob_changes = []
        tissue_changes = np.zeros(len(self.tissue_names))
        
        for _ in range(num_perturbations):
            # 生成扰动序列
            if perturbation_type == 'substitution':
                perturbed_seq = self.substitute_nucleotide(sequence, position)
            elif perturbation_type == 'deletion':
                perturbed_seq = self.delete_nucleotide(sequence, position)
            elif perturbation_type == 'insertion':
                perturbed_seq = self.insert_nucleotide(sequence, position)
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
            # 创建扰动样本
            perturbed_data = sample_data.copy()
            perturbed_data['sequence'] = perturbed_seq
            
            # 获取扰动后的预测
            perturbed_pred, perturbed_probs = self.get_prediction(perturbed_data)
            
            # 计算变化
            change_score, cat_change, prob_change = self.calculate_prediction_change_score(
                original_pred, perturbed_pred, original_probs, perturbed_probs
            )
            
            change_scores.append(change_score)
            category_changes.append(cat_change)
            prob_changes.append(prob_change)
            
            # 记录每个组织的变化
            for tissue_idx in range(len(self.tissue_names)):
                if original_pred[tissue_idx] != perturbed_pred[tissue_idx]:
                    tissue_changes[tissue_idx] += 1
        
        importance_metrics = {
            'change_score': np.mean(change_scores),
            'change_score_std': np.std(change_scores),
            'category_change': np.mean(category_changes),
            'prob_change': np.mean(prob_changes),
            'tissue_changes': tissue_changes / num_perturbations,
        }
        
        return importance_metrics
    
    def analyze_full_sequence(self, sequence, sample_data, 
                             step_size=10, num_perturbations=5,
                             perturbation_type='substitution'):
        """
        分析整个序列的重要性
        
        Args:
            sequence: 输入序列
            sample_data: 样本数据
            step_size: 采样步长
            num_perturbations: 每个位置的扰动次数
            perturbation_type: 扰动类型
        
        Returns:
            results: 分析结果字典
        """
        seq_length = len(sequence)
        positions_to_test = range(0, seq_length, step_size)
        
        print(f"\n🔬 Analyzing sequence with {len(list(positions_to_test))} positions...")
        
        # 初始化结果数组
        importance_scores = np.zeros(seq_length)
        position_tissue_importance = np.zeros((seq_length, len(self.tissue_names)))
        
        for pos in tqdm(positions_to_test, desc="Analyzing positions"):
            metrics = self.analyze_position_importance(
                sequence, sample_data, pos, 
                num_perturbations=num_perturbations,
                perturbation_type=perturbation_type
            )
            
            # 填充到步长内的所有位置
            start_pos = pos
            end_pos = min(pos + step_size, seq_length)
            for p in range(start_pos, end_pos):
                importance_scores[p] = metrics['change_score']
                position_tissue_importance[p] = metrics['tissue_changes']
        
        results = {
            'importance_scores': importance_scores,
            'position_tissue_importance': position_tissue_importance,
            'sequence': sequence,
            'sample_data': sample_data,
        }
        
        return results
    
    def plot_comprehensive_analysis(self, results, sample_id="sample", 
                                   save_dir=None, show_top_k=20):
        """
        绘制综合分析图
        
        Args:
            results: 分析结果
            sample_id: 样本ID
            save_dir: 保存目录
            show_top_k: 显示前K个重要位置
        """
        importance_scores = results['importance_scores']
        position_tissue_importance = results['position_tissue_importance']
        sequence = results['sequence']
        
        seq_length = len(sequence)
        positions = np.arange(seq_length)
        
        # 创建大型综合图
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. 总体重要性条形图（带核苷酸颜色）
        ax1 = fig.add_subplot(gs[0, :])
        colors = []
        nuc_color_map = {
            'A': '#FF6B6B', 'T': '#4ECDC4', 
            'C': '#45B7D1', 'G': '#FFA07A'
        }
        for nuc in sequence:
            colors.append(nuc_color_map.get(nuc.upper(), '#CCCCCC'))
        
        ax1.bar(positions, importance_scores, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('Sequence Position', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Importance Score', fontsize=14, fontweight='bold')
        ax1.set_title(f'Overall Sequence Importance - {sample_id}', 
                     fontsize=16, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='A'),
            Patch(facecolor='#4ECDC4', label='T'),
            Patch(facecolor='#45B7D1', label='C'),
            Patch(facecolor='#FFA07A', label='G')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=12)
        
        # 2. 组织特异性重要性热力图
        ax2 = fig.add_subplot(gs[1, :])
        sns.heatmap(position_tissue_importance.T, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Prediction Change Frequency'},
                   xticklabels=False,
                   yticklabels=self.tissue_names,
                   ax=ax2,
                   vmin=0,
                   vmax=1)
        ax2.set_xlabel('Sequence Position', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Tissue Type', fontsize=14, fontweight='bold')
        ax2.set_title('Tissue-Specific Importance Heatmap', fontsize=16, fontweight='bold')
        
        # 3. Top K 重要位置详细信息
        ax3 = fig.add_subplot(gs[2, 0])
        top_k_indices = np.argsort(importance_scores)[-show_top_k:][::-1]
        top_k_scores = importance_scores[top_k_indices]
        top_k_nucs = [sequence[i] for i in top_k_indices]
        
        bars = ax3.barh(range(show_top_k), top_k_scores, 
                       color=[nuc_color_map.get(n.upper(), '#CCCCCC') for n in top_k_nucs])
        ax3.set_yticks(range(show_top_k))
        ax3.set_yticklabels([f"Pos {idx}: {nuc}" for idx, nuc in zip(top_k_indices, top_k_nucs)])
        ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax3.set_title(f'Top {show_top_k} Most Important Positions', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. 核苷酸类型重要性分布
        ax4 = fig.add_subplot(gs[2, 1])
        nuc_importance = {'A': [], 'T': [], 'C': [], 'G': []}
        for i, nuc in enumerate(sequence):
            if nuc.upper() in nuc_importance:
                nuc_importance[nuc.upper()].append(importance_scores[i])
        
        box_data = [nuc_importance[n] for n in ['A', 'T', 'C', 'G']]
        bp = ax4.boxplot(box_data, labels=['A', 'T', 'C', 'G'], patch_artist=True)
        for patch, nuc in zip(bp['boxes'], ['A', 'T', 'C', 'G']):
            patch.set_facecolor(nuc_color_map[nuc])
            patch.set_alpha(0.7)
        ax4.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Nucleotide Type', fontsize=12, fontweight='bold')
        ax4.set_title('Importance Distribution by Nucleotide Type', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. 滑动窗口平均重要性
        ax5 = fig.add_subplot(gs[3, 0])
        window_size = 50
        if seq_length >= window_size:
            smoothed = np.convolve(importance_scores, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
            ax5.plot(range(len(smoothed)), smoothed, linewidth=2, color='#E74C3C')
            ax5.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='#E74C3C')
        else:
            ax5.plot(positions, importance_scores, linewidth=2, color='#E74C3C')
        ax5.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Smoothed Importance', fontsize=12, fontweight='bold')
        ax5.set_title(f'Smoothed Importance (Window={window_size})', fontsize=14, fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # 6. 统计信息表格
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.axis('off')
        
        stats_data = [
            ['Mean Importance', f'{importance_scores.mean():.4f}'],
            ['Std Importance', f'{importance_scores.std():.4f}'],
            ['Max Importance', f'{importance_scores.max():.4f}'],
            ['Min Importance', f'{importance_scores.min():.4f}'],
            ['Sequence Length', f'{seq_length} bp'],
            ['A Count', f'{sequence.upper().count("A")}'],
            ['T Count', f'{sequence.upper().count("T")}'],
            ['C Count', f'{sequence.upper().count("C")}'],
            ['G Count', f'{sequence.upper().count("G")}'],
            ['GC Content', f'{(sequence.upper().count("G") + sequence.upper().count("C")) / seq_length * 100:.2f}%'],
        ]
        
        table = ax6.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#3498DB')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
        
        ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'Comprehensive Sequence Perturbation Analysis\nSample: {sample_id}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'comprehensive_analysis_{sample_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comprehensive analysis saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def export_results_to_csv(self, results, sample_id, save_dir=None):
        """导出结果到CSV文件"""
        importance_scores = results['importance_scores']
        position_tissue_importance = results['position_tissue_importance']
        sequence = results['sequence']
        
        # 创建DataFrame
        data = {
            'position': range(len(sequence)),
            'nucleotide': list(sequence),
            'importance_score': importance_scores,
        }
        
        # 添加每个组织的重要性
        for i, tissue in enumerate(self.tissue_names):
            data[f'{tissue}_importance'] = position_tissue_importance[:, i]
        
        df = pd.DataFrame(data)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f'importance_analysis_{sample_id}.csv')
            df.to_csv(csv_path, index=False)
            print(f"💾 Results exported to: {csv_path}")
        
        return df


def main():
    """主函数"""
    print("="*80)
    print("🧬 Advanced Sequence Perturbation Importance Analysis")
    print("="*80)
    
    # 加载模型
    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
    print(f"\n📦 Loading model from: {model_path}")
    model = ModelHub.load(model_path)
    
    # 创建分析器
    analyzer = SequencePerturbationAnalyzer(model)
    
    # 加载数据集
    print("\n📊 Loading dataset...")
    model_name_or_path = "yangheng/OmniGenome-52M"
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    datasets = TriClassTEDataset.from_hub(
        dataset_name_or_path=os.path.dirname(__file__),
        tokenizer=tokenizer,
        max_length=512,
        force_padding=False
    )
    
    # 选择测试样本
    num_samples = min(3, len(datasets['test'].examples))
    test_samples = datasets['test'].examples[:num_samples]
    
    # 创建保存目录
    save_dir = "perturbation_analysis_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 对每个样本进行分析
    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"🔬 Analyzing Sample {idx+1}/{num_samples}")
        print(f"   Sample ID: {sample['ID']}")
        print(f"   Sequence Length: {len(sample['sequence'])} bp")
        print(f"{'='*80}")
        
        # 进行全序列分析
        results = analyzer.analyze_full_sequence(
            sequence=sample['sequence'],
            sample_data=sample,
            step_size=10,  # 每10个碱基测试一次
            num_perturbations=5,  # 每个位置扰动5次
            perturbation_type='substitution'
        )
        
        # 绘制综合分析图
        analyzer.plot_comprehensive_analysis(
            results=results,
            sample_id=sample['ID'],
            save_dir=save_dir,
            show_top_k=20
        )
        
        # 导出结果到CSV
        df = analyzer.export_results_to_csv(
            results=results,
            sample_id=sample['ID'],
            save_dir=save_dir
        )
        
        print(f"\n✅ Sample {idx+1} analysis completed!")
        print(f"   Mean importance: {results['importance_scores'].mean():.4f}")
        print(f"   Max importance: {results['importance_scores'].max():.4f}")
        print(f"   Results saved to: {save_dir}/")
    
    print("\n" + "="*80)
    print("🎉 All analyses completed!")
    print(f"📁 Results saved to: {os.path.abspath(save_dir)}/")
    print("="*80)


if __name__ == "__main__":
    main()
