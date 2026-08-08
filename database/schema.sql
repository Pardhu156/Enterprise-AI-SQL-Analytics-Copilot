-- Enterprise AI SQL Analytics Copilot - Phase 1
-- PostgreSQL schema for the Olist Brazilian E-Commerce dataset.

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    customer_unique_id VARCHAR(32) NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city TEXT NOT NULL,
    customer_state CHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS sellers (
    seller_id VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city TEXT NOT NULL,
    seller_state CHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS product_category_translation (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_category_name TEXT,
    product_name_lenght INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    order_status TEXT NOT NULL,
    order_purchase_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    order_approved_at TIMESTAMP WITHOUT TIME ZONE,
    order_delivered_carrier_date TIMESTAMP WITHOUT TIME ZONE,
    order_delivered_customer_date TIMESTAMP WITHOUT TIME ZONE,
    order_estimated_delivery_date TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    order_id VARCHAR(32) NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id VARCHAR(32) NOT NULL,
    seller_id VARCHAR(32) NOT NULL,
    shipping_limit_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    price NUMERIC(12, 2) NOT NULL,
    freight_value NUMERIC(12, 2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    CONSTRAINT fk_order_items_order
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
    CONSTRAINT fk_order_items_product
        FOREIGN KEY (product_id) REFERENCES products (product_id),
    CONSTRAINT fk_order_items_seller
        FOREIGN KEY (seller_id) REFERENCES sellers (seller_id)
);

CREATE TABLE IF NOT EXISTS payments (
    order_id VARCHAR(32) NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type TEXT NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value NUMERIC(12, 2) NOT NULL,
    PRIMARY KEY (order_id, payment_sequential),
    CONSTRAINT fk_payments_order
        FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

-- The source contains repeated review_id and order_id values, so neither is a
-- safe primary key. review_row_id preserves every distinct source review row.
CREATE TABLE IF NOT EXISTS reviews (
    review_row_id BIGSERIAL PRIMARY KEY,
    review_id VARCHAR(32) NOT NULL,
    order_id VARCHAR(32) NOT NULL,
    review_score SMALLINT NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    review_answer_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    CONSTRAINT fk_reviews_order
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
    CONSTRAINT ck_reviews_score CHECK (review_score BETWEEN 1 AND 5)
);

-- Olist provides multiple coordinate observations per ZIP prefix. Keeping this
-- table without a synthetic business-key constraint avoids discarding valid rows.
CREATE TABLE IF NOT EXISTS geolocation (
    geolocation_zip_code_prefix INTEGER NOT NULL,
    geolocation_lat DOUBLE PRECISION NOT NULL,
    geolocation_lng DOUBLE PRECISION NOT NULL,
    geolocation_city TEXT NOT NULL,
    geolocation_state CHAR(2) NOT NULL
);

-- Join and analytical indexes.
CREATE INDEX IF NOT EXISTS idx_customers_unique_id ON customers (customer_unique_id);
CREATE INDEX IF NOT EXISTS idx_customers_state ON customers (customer_state);
CREATE INDEX IF NOT EXISTS idx_customers_zip ON customers (customer_zip_code_prefix);
CREATE INDEX IF NOT EXISTS idx_sellers_state ON sellers (seller_state);
CREATE INDEX IF NOT EXISTS idx_sellers_zip ON sellers (seller_zip_code_prefix);
CREATE INDEX IF NOT EXISTS idx_products_category ON products (product_category_name);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_purchase_timestamp ON orders (order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (order_status);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items (product_id);
CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items (seller_id);
CREATE INDEX IF NOT EXISTS idx_payments_type ON payments (payment_type);
CREATE INDEX IF NOT EXISTS idx_reviews_order_id ON reviews (order_id);
CREATE INDEX IF NOT EXISTS idx_reviews_review_id ON reviews (review_id);
CREATE INDEX IF NOT EXISTS idx_geolocation_zip ON geolocation (geolocation_zip_code_prefix);

COMMENT ON TABLE products IS
    'Product_category_name is intentionally not constrained by a foreign key: '
    'the Olist translation file does not cover every legitimate Portuguese category.';
COMMENT ON TABLE reviews IS
    'Raw review records; review_row_id is a warehouse surrogate because source IDs are not unique.';
COMMENT ON TABLE geolocation IS
    'Raw ZIP-prefix coordinate observations; multiple rows per ZIP prefix are expected.';
