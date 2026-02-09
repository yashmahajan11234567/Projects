import pandas as pd

# Load product dimension, force both name and ID to string for clean matching
dim_products = pd.read_csv('export/dim_product.csv')
product_name_to_id = {str(name).strip(): int(pid) for name, pid in zip(dim_products['product_name'], dim_products['product_id'])}

# Utility for splitting names, stripping quotes/commas, mapping to IDs
def names_to_ids(cell):
    # Handles quotes and commas inside quoted text
    if pd.isnull(cell):
        return []
    names = []
    cell = str(cell)
    if cell.startswith('"') and cell.endswith('"'):
        # Remove surrounding quotes, then split by comma
        names = [n.strip() for n in cell[1:-1].split(',')]
    else:
        names = [n.strip() for n in cell.split(',')]
    # Map to IDs, skipping names not found
    ids = [product_name_to_id[n] for n in names if n in product_name_to_id]
    return ids

# Load and expand assoc_rules
arules = pd.read_csv('export/assoc_rules.csv')

assoc_rows = []
for idx, row in arules.iterrows():
    a_ids = names_to_ids(row['antecedents'])
    c_ids = names_to_ids(row['consequents'])
    for pid in a_ids:
        assoc_rows.append({
            'rule_idx': idx,
            'product_id': pid,
            'rule_type': 'antecedent',
            'confidence': row['confidence'],
            'lift': row['lift'],
            'support': row['support'],
            'other_part': row['consequents']
        })
    for pid in c_ids:
        assoc_rows.append({
            'rule_idx': idx,
            'product_id': pid,
            'rule_type': 'consequent',
            'confidence': row['confidence'],
            'lift': row['lift'],
            'support': row['support'],
            'other_part': row['antecedents']
        })

expanded_assoc = pd.DataFrame(assoc_rows)
expanded_assoc.to_csv('export/assoc_rules_exploded.csv', index=False)
print("export/assoc_rules_exploded.csv written!")

# -- Recommendations (unchanged, since these are product IDs) --
recs = pd.read_csv('export/item_recs_top10.csv')
src = recs[['source_product_id', 'recommended_product_id', 'score', 'co_cnt']].copy()
src = src.rename(columns={'source_product_id': 'product_id'})
src['rec_type'] = 'source'

rec = recs[['source_product_id', 'recommended_product_id', 'score', 'co_cnt']].copy()
rec = rec.rename(columns={'recommended_product_id': 'product_id'})
rec['rec_type'] = 'recommended'

expanded_recs = pd.concat([src, rec], ignore_index=True)
expanded_recs.to_csv('export/item_recs_exploded.csv', index=False)
print("export/item_recs_exploded.csv written!")
