# Enterprise AI SQL Analytics Copilot

**Current milestone:** Phase 1 — Data and PostgreSQL Foundation

This repository builds a reproducible analytical data layer for an enterprise-style SQL copilot. Phase 1 turns the public Olist Brazilian E-Commerce CSVs into conservatively cleaned files, a relational PostgreSQL model, reusable analytical views, and a benchmark suite of manually verified business questions. It deliberately contains no LLM, Text-to-SQL, API, UI, container, or CI/CD implementation.

## Dataset

The [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) describes customers, orders, products, sellers, payments, reviews, order items, and Brazilian ZIP-prefix geolocation. Download it manually and place the source CSV files in `data/raw/`; this project never downloads data automatically.

Expected files:

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

Both raw and processed CSVs are ignored by Git. The `.gitkeep` files preserve their directories.

## Phase 1 architecture

```text
Olist CSVs (data/raw)
        │
        ├── notebooks/data_exploration.ipynb  → quality and relationship profile
        │
        └── src/data_processing.py            → conservative cleanup
                         │
                         ▼
                 data/processed/*.csv
                         │
                         └── src/load_postgres.py
                                  │
                                  ▼
                 PostgreSQL tables → analytical views → benchmark SQL
```

The processing step standardizes column labels, removes only exact duplicate rows, validates timestamp parsing, and preserves meaningful nulls. The loader uses PostgreSQL `COPY` inside one transaction and loads parent tables before their dependants.

## Repository structure

```text
.
├── data/
│   ├── raw/                         # manually supplied source CSVs
│   └── processed/                   # generated cleaned CSVs
├── database/
│   ├── schema.sql                   # tables, keys, constraints, indexes
│   ├── views.sql                    # reusable business views
│   └── benchmark_queries.sql        # 22 ground-truth analytical queries
├── notebooks/
│   └── data_exploration.ipynb       # data quality exploration
├── src/
│   ├── data_processing.py           # CSV cleaning pipeline
│   ├── db_config.py                 # environment-only DB configuration
│   └── load_postgres.py             # transactional bulk loader
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Database model

| Table | Grain and purpose |
|---|---|
| `customers` | One order-facing customer record per `customer_id`; `customer_unique_id` links repeat buyers |
| `orders` | One order, including status and purchase/approval/delivery timestamps |
| `order_items` | One item sequence within an order; product, seller, price, and freight |
| `products` | One product with Portuguese category and physical attributes |
| `sellers` | One marketplace seller and location |
| `payments` | One payment sequence within an order |
| `reviews` | One preserved source review row, keyed by a warehouse surrogate because source IDs can repeat |
| `geolocation` | ZIP-prefix coordinate observations; multiple rows per prefix are legitimate |
| `product_category_translation` | Portuguese-to-English product category lookup |

Core relationships:

```text
customers ──< orders ──< order_items >── products
                 │             │              │
                 │             └────> sellers └── category translation (optional lookup)
                 ├──< payments
                 └──< reviews
```

Foreign keys enforce relationships that are reliable in the source data. The product-to-translation relationship is indexed but not enforced because the translation file does not cover every legitimate category. Geolocation remains a separate observation table so joining it directly cannot accidentally multiply customer or seller facts; aggregate it to one row per ZIP prefix before such a join.

## Setup

Requirements are Python 3.10+ and PostgreSQL 13+.

1. Create and activate a virtual environment, then install Phase 1 dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a PostgreSQL database. For example, as a PostgreSQL administrator:

   ```bash
   createdb olist_analytics
   ```

3. Copy the environment template and replace every placeholder locally:

   ```bash
   cp .env.example .env
   ```

   ```dotenv
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=olist_analytics
   DB_USER=postgres
   DB_PASSWORD=your_real_local_password
   ```

   `.env` is ignored by Git. Credentials are never hardcoded in source code.

4. Place all nine Olist CSV files listed above in `data/raw/`.

## Explore and clean the data

Start Jupyter and run every cell in the exploration notebook:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

The notebook displays shapes, columns, types, missing values, exact duplicates, identifier cardinality, timestamp ranges and parse failures, numeric summaries, and orphan-key counts.

Generate processed CSVs:

```bash
python src/data_processing.py
```

The script logs rows loaded, exact duplicates removed, and rows saved for each dataset. It fails on invalid non-null timestamps instead of silently converting questionable business data. Custom directories may be supplied with `--raw-dir` and `--processed-dir`.

## Load PostgreSQL

Run the transactional loader after configuration and processing:

```bash
python src/load_postgres.py
```

It creates the schema, bulk-loads all nine tables in dependency-safe order, reports inserted counts, and creates the analytical views. Any database error rolls back the transaction and is reported.

A normal load is append-only and therefore fails on duplicate primary keys if run twice. To intentionally replace all existing Olist rows with the current processed files, use:

```bash
python src/load_postgres.py --truncate
```

Use `--skip-views` only when tables should be loaded without applying `database/views.sql`.

## Views and benchmark queries

The loader applies the views automatically. They can also be reapplied independently:

```bash
psql -d olist_analytics -f database/views.sql
```

Available views cover order-level revenue, monthly revenue, category performance, seller performance, delivery performance, the review/delivery relationship, and payment summaries. Revenue is defined as item price; canceled and unavailable orders are excluded in the aggregate revenue views and relevant benchmark queries. Freight and payment values remain separate measures.

Run the full benchmark suite with:

```bash
psql -d olist_analytics -f database/benchmark_queries.sql
```

The 22 queries provide ground truth for Phase 2 and include total and monthly revenue, category and seller rankings, state demand and order value, reviews, late-delivery rates, delivery-rating relationships, payment behavior, freight costs, delivery time, monthly volume, products, repeat customers, cancellations, and geographic delivery performance. Each query uses explicit columns and aliases and includes its natural-language question as a SQL comment.

## Next milestone

Phase 2 will build the LLM-powered Text-to-SQL engine and evaluate its generated SQL against these benchmark questions. That application layer is intentionally out of scope for this milestone.
