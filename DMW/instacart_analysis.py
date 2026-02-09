# ==============================
# 1) Import packages
# ==============================
import os
import duckdb
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from scipy.sparse import issparse

# ==============================
# 2) Folder setup
# ==============================
DATA_DIR = "data"
EXPORT_DIR = "export"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

required = {
    "aisles.csv",
    "departments.csv",
    "products.csv",
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
}

files_found = {f for f in os.listdir(DATA_DIR) if f.endswith(".csv")}
print("Found:", sorted(files_found))
missing = required - files_found
if missing:
    raise SystemExit(f"Missing files: {sorted(missing)}")
print("All required files present ✓")

# ===== Clean orders.csv for DuckDB compatibility =====
orders_path = os.path.join(DATA_DIR, "orders.csv")
orders_df = pd.read_csv(orders_path)
orders_df["days_since_prior_order"] = pd.to_numeric(orders_df["days_since_prior_order"], errors="coerce")
orders_df.to_csv(orders_path, index=False)
print("orders.csv cleaned ✓")

# ==============================
# 3) Load data into DuckDB
# ==============================
DB = "instacart.duckdb"
con = duckdb.connect(DB)

con.execute(f"""
CREATE OR REPLACE VIEW st_aisles      AS SELECT * FROM read_csv_auto('{DATA_DIR}/aisles.csv', header=True);
CREATE OR REPLACE VIEW st_departments AS SELECT * FROM read_csv_auto('{DATA_DIR}/departments.csv', header=True);
CREATE OR REPLACE VIEW st_products    AS SELECT * FROM read_csv_auto('{DATA_DIR}/products.csv', header=True);
CREATE OR REPLACE VIEW st_orders      AS SELECT * FROM read_csv_auto('{DATA_DIR}/orders.csv', header=True);
CREATE OR REPLACE VIEW st_op_prior    AS SELECT * FROM read_csv_auto('{DATA_DIR}/order_products__prior.csv', header=True);
CREATE OR REPLACE VIEW st_op_train    AS SELECT * FROM read_csv_auto('{DATA_DIR}/order_products__train.csv', header=True);
""")

# Dimensions
con.execute("""
CREATE OR REPLACE TABLE dim_aisle AS
SELECT CAST(aisle_id AS INT) AS aisle_id, aisle FROM st_aisles;

CREATE OR REPLACE TABLE dim_department AS
SELECT CAST(department_id AS INT) AS department_id, department FROM st_departments;

CREATE OR REPLACE TABLE dim_product AS
SELECT CAST(product_id AS INT) AS product_id, product_name,
       CAST(aisle_id AS INT) AS aisle_id,
       CAST(department_id AS INT) AS department_id
FROM st_products;

CREATE OR REPLACE TABLE dim_user AS
SELECT DISTINCT CAST(user_id AS INT) AS user_id
FROM st_orders;
""")

# Facts
con.execute("""
CREATE OR REPLACE TABLE fact_orders AS
SELECT CAST(order_id AS INT) AS order_id,
       CAST(user_id AS INT) AS user_id,
       eval_set,
       CAST(order_number AS INT) AS order_number,
       CAST(order_dow AS INT) AS order_dow,
       CAST(order_hour_of_day AS INT) AS order_hour_of_day,
       CAST(days_since_prior_order AS DOUBLE) AS days_since_prior_order
FROM st_orders;

CREATE OR REPLACE TABLE fact_order_products AS
SELECT CAST(order_id AS INT) AS order_id,
       CAST(product_id AS INT) AS product_id,
       CAST(add_to_cart_order AS INT) AS add_to_cart_order,
       CAST(reordered AS INT) AS reordered,
       'prior' AS stage
FROM st_op_prior
UNION ALL
SELECT CAST(order_id AS INT),
       CAST(product_id AS INT),
       CAST(add_to_cart_order AS INT),
       CAST(reordered AS INT),
       'train' AS stage
FROM st_op_train;
""")

def safe_count_query(con, sql):
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no result: {sql}")
    return row[0]

n_orders = safe_count_query(con, "SELECT COUNT(*) FROM fact_orders")
n_lines  = safe_count_query(con, "SELECT COUNT(*) FROM fact_order_products")
print(f"Loaded ✓ orders={n_orders:,}, order_lines={n_lines:,} into {DB}")

# ==============================
# 4) Association Rules (FPGrowth)
# ==============================
TOP_N = 300
MIN_SUPPORT = 0.005
MIN_CONF = 0.2
MAX_TX = 150_000

top_products = con.execute(f"""
SELECT product_id FROM (
  SELECT product_id, COUNT(*) cnt
  FROM fact_order_products
  WHERE stage='prior'
  GROUP BY 1 ORDER BY cnt DESC
) LIMIT {TOP_N};
""").df().product_id.tolist()

tx_df = con.execute(f"""
WITH filtered AS (
  SELECT order_id, product_id
  FROM fact_order_products
  WHERE stage='prior' AND product_id IN ({",".join(map(str, top_products))})
),
tx_orders AS (
  SELECT order_id FROM filtered GROUP BY 1
  HAVING COUNT(*) BETWEEN 2 AND 50
  ORDER BY random() LIMIT {MAX_TX}
)
SELECT list(product_id) AS items
FROM filtered f JOIN tx_orders t USING(order_id)
GROUP BY order_id;
""").df()

transactions = tx_df["items"].tolist()
print("Baskets:", len(transactions))

te = TransactionEncoder()
ohe = te.fit(transactions).transform(transactions)
if ohe is None:
    raise ValueError("TransactionEncoder.transform returned None; check your input transactions.")

if issparse(ohe):
    ohe_dense = ohe.toarray()
elif isinstance(ohe, np.ndarray):
    ohe_dense = ohe
elif hasattr(ohe, "to_numpy"):
    ohe_dense = ohe.to_numpy()
else:
    try:
        ohe_dense = np.array(ohe)
    except Exception:
        ohe_dense = [list(row) for row in ohe]

df_bin = pd.DataFrame(ohe_dense, columns=te.columns_)

freq = fpgrowth(df_bin, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONF).sort_values(
    ["lift", "confidence", "support"], ascending=False
)

pmap = con.execute("SELECT product_id, product_name FROM dim_product").df().set_index("product_id")["product_name"].to_dict()
def pretty(s): return ", ".join(pmap.get(int(i), str(i)) for i in list(s))

rules_show = rules.head(15).copy()
rules_show["antecedents"] = rules_show["antecedents"].apply(pretty)
rules_show["consequents"] = rules_show["consequents"].apply(pretty)
print(rules_show[["antecedents", "consequents", "support", "confidence", "lift"]])

# ==============================
# 5) Item-to-Item Recommendations
# ==============================
TOP_N = 500
PAIR_SAMPLE_ORDERS = 300_000
MIN_CONF = 0.03

con.execute(f"""
CREATE OR REPLACE TEMP TABLE filtered AS
SELECT order_id, product_id
FROM fact_order_products
WHERE stage='prior';

CREATE OR REPLACE TEMP TABLE tx_orders AS
SELECT order_id FROM filtered GROUP BY 1
HAVING COUNT(*) BETWEEN 2 AND 50
ORDER BY random() LIMIT {PAIR_SAMPLE_ORDERS};

CREATE OR REPLACE TEMP TABLE filtered_s AS
SELECT f.* FROM filtered f JOIN tx_orders t USING(order_id);

CREATE OR REPLACE TEMP TABLE prod_cnt AS
SELECT product_id, COUNT(*) cnt FROM filtered_s GROUP BY 1;

CREATE OR REPLACE TABLE item_assoc AS
SELECT a.product_id AS a, b.product_id AS b,
       COUNT(*) AS co_cnt,
       CAST(COUNT(*) AS DOUBLE)/pc.cnt AS p_b_given_a
FROM filtered_s a
JOIN filtered_s b
  ON a.order_id=b.order_id AND a.product_id<>b.product_id
JOIN prod_cnt pc ON pc.product_id=a.product_id
GROUP BY 1,2,pc.cnt;
""")

def recommend_for_cart(cart_product_ids, k=10, min_conf=MIN_CONF):
    cart = ",".join(map(str, cart_product_ids))
    q = f"""
    WITH scores AS (
      SELECT b AS product_id, SUM(p_b_given_a) AS score, COUNT(*) AS hits
      FROM item_assoc
      WHERE a IN ({cart}) AND p_b_given_a >= {min_conf}
      GROUP BY 1
    )
    SELECT s.product_id, s.score, s.hits, p.product_name
    FROM scores s JOIN dim_product p USING(product_id)
    WHERE s.product_id NOT IN ({cart})
    ORDER BY score DESC
    LIMIT {k};
    """
    return con.execute(q).df()

seed_cart = con.execute("""
SELECT product_id
FROM fact_order_products
WHERE stage='prior'
GROUP BY product_id
ORDER BY COUNT(*) DESC
LIMIT 3;
""").df()["product_id"].tolist()

print("Seed cart:", seed_cart)
print(recommend_for_cart(seed_cart, k=10))

# ==============================
# 6) Reorder Prediction (Logistic Regression)
# ==============================
con.execute("""
CREATE OR REPLACE TEMP VIEW train_orders AS
  SELECT order_id, user_id FROM fact_orders WHERE eval_set='train';

CREATE OR REPLACE TEMP VIEW user_agg AS
  SELECT user_id, COUNT(*) user_n_orders, AVG(days_since_prior_order) user_avg_days
  FROM fact_orders WHERE eval_set='prior' GROUP BY 1;

CREATE OR REPLACE TEMP VIEW up_agg AS
  SELECT o.user_id, op.product_id, COUNT(*) up_orders, MAX(order_number) up_last
  FROM fact_order_products op
  JOIN fact_orders o ON o.order_id=op.order_id
  WHERE op.stage='prior'
  GROUP BY 1,2;

CREATE OR REPLACE TEMP VIEW labels AS
  SELECT t.user_id, op.product_id, 1 y
  FROM train_orders t JOIN fact_order_products op ON op.order_id=t.order_id;

CREATE OR REPLACE TEMP VIEW train_set AS
  SELECT c.user_id, c.product_id, COALESCE(l.y,0) y,
         ua.user_n_orders, ua.user_avg_days, ua2.up_orders, ua2.up_last
  FROM (SELECT DISTINCT user_id, product_id FROM up_agg) c
  JOIN user_agg ua ON ua.user_id=c.user_id
  JOIN up_agg ua2 ON ua2.user_id=c.user_id AND ua2.product_id=c.product_id
  LEFT JOIN labels l ON l.user_id=c.user_id AND l.product_id=c.product_id;
""")

df = con.execute("SELECT * FROM train_set USING SAMPLE 200000 ROWS;").df().fillna(0.0)
X = df[["user_n_orders","user_avg_days","up_orders","up_last"]]
y = df["y"].astype(int)

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=300, solver="saga")
clf.fit(Xtr, ytr)
pred = clf.predict_proba(Xva)[:,1]
print("ROC AUC:", round(roc_auc_score(yva, pred), 4), " PR AUC:", round(average_precision_score(yva, pred), 4))

df["p_reorder"] = clf.predict_proba(X)[:,1]
preds = (df.sort_values(["user_id","p_reorder"], ascending=[True, False])
           .groupby("user_id").head(5))
names = con.execute("SELECT product_id, product_name FROM dim_product").df()
preds = preds.merge(names, on="product_id")
print(preds[["user_id","product_id","product_name","p_reorder"]].head(10))

# ==============================
# 7) Export CSVs
# ==============================
for tbl in ["dim_product","dim_aisle","dim_department"]:
    con.execute(f"COPY {tbl} TO '{EXPORT_DIR}/{tbl}.csv' (HEADER, DELIMITER ',');")

rules_out = rules.copy()
rules_out["antecedents"] = rules_out["antecedents"].apply(pretty)
rules_out["consequents"] = rules_out["consequents"].apply(pretty)
rules_out = rules_out[["antecedents","consequents","support","confidence","lift"]].sort_values(["lift","confidence"], ascending=False)
rules_out.to_csv(f"{EXPORT_DIR}/assoc_rules.csv", index=False)

con.execute(f"""
COPY (
  SELECT ia.a AS source_product_id, dp1.product_name AS source_product,
         ia.b AS recommended_product_id, dp2.product_name AS recommended_product,
         ia.p_b_given_a AS score, ia.co_cnt
  FROM item_assoc ia
  JOIN dim_product dp1 ON dp1.product_id=ia.a
  JOIN dim_product dp2 ON dp2.product_id=ia.b
  QUALIFY ROW_NUMBER() OVER (PARTITION BY ia.a ORDER BY ia.p_b_given_a DESC) <= 10
) TO '{EXPORT_DIR}/item_recs_top10.csv' (HEADER, DELIMITER ',' );
""")

preds[["user_id","product_id","product_name","p_reorder"]].to_csv(f"{EXPORT_DIR}/reorder_preds_sample.csv", index=False)

con.execute(f"""
COPY (
  SELECT p.product_id, p.product_name,
         COUNT(*) AS orders_cnt,
         AVG(op.reordered) AS reorder_rate,
         AVG(op.add_to_cart_order) AS avg_cart_position
  FROM fact_order_products op
  JOIN dim_product p USING(product_id)
  WHERE op.stage='prior'
  GROUP BY 1,2
  ORDER BY orders_cnt DESC
  LIMIT 5000
) TO '{EXPORT_DIR}/popular_products.csv' (HEADER, DELIMITER ',' );
""")

con.execute(f"""
COPY (
  SELECT d.department, COUNT(*) AS lines, AVG(op.reordered) AS reorder_rate
  FROM fact_order_products op
  JOIN dim_product p USING(product_id)
  JOIN dim_department d USING(department_id)
  WHERE op.stage='prior'
  GROUP BY 1 ORDER BY reorder_rate DESC
) TO '{EXPORT_DIR}/reorder_rate_by_department.csv' (HEADER, DELIMITER ',' );
""")

con.execute(f"""
COPY (
  WITH basket AS (
    SELECT order_id, COUNT(*) AS basket_size
    FROM fact_order_products WHERE stage='prior'
    GROUP BY order_id
  )
  SELECT o.order_dow, o.order_hour_of_day, AVG(b.basket_size) AS avg_basket
  FROM basket b JOIN fact_orders o ON o.order_id=b.order_id
  GROUP BY 1,2 ORDER BY 1,2
) TO '{EXPORT_DIR}/basket_size_heatmap.csv' (HEADER, DELIMITER ',' );
""")

print(f"Exported CSVs in {EXPORT_DIR} ✓")
