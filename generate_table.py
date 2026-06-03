import pandas as pd
import sys

dataset = sys.argv[1]

# 加载结果
results = pd.read_csv(f'results/{dataset}_ablation.csv', 
                     names=['Variant', 'MAE', 'RMSE'])

# 计算相对于完整模型的变化
full_mae = results[results['Variant'] == 'full']['MAE'].values[0]
full_rmse = results[results['Variant'] == 'full']['RMSE'].values[0]

results['MAE Change'] = results['MAE'].apply(
    lambda x: f"+{(x - full_mae)/full_mae*100:.1f}%" if x > full_mae else ""
)

results['RMSE Change'] = results['RMSE'].apply(
    lambda x: f"+{(x - full_rmse)/full_rmse*100:.1f}%" if x > full_rmse else ""
)

# 格式化表格
results['MAE'] = results['MAE'].apply(lambda x: f"{x:.3f}")
results['RMSE'] = results['RMSE'].apply(lambda x: f"{x:.3f}")

# 重命名变体
variant_names = {
    'full': 'Full Model',
    'no_direction': 'w/o Direction',
    'no_roadtype': 'w/o Road Type',
    'no_adaptive': 'w/o Adaptive Weights',
    'no_constraint': 'w/o Constraints'
}

results['Variant'] = results['Variant'].map(variant_names)

# 保存为Markdown表格
results.to_markdown(f'results/{dataset}_ablation_table.md', index=False)
print(f"Ablation study table saved to results/{dataset}_ablation_table.md")
