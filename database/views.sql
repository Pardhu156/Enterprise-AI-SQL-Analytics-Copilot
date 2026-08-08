-- Reusable analytical views for the Olist Phase 1 warehouse.

-- One row per order with item, freight, payment, and review measures aggregated
-- independently to prevent multiplication across one-to-many joins.
CREATE OR REPLACE VIEW vw_order_revenue AS
WITH item_totals AS (
    SELECT
        oi.order_id,
        COUNT(*) AS item_count,
        SUM(oi.price) AS item_revenue,
        SUM(oi.freight_value) AS freight_revenue
    FROM order_items AS oi
    GROUP BY oi.order_id
),
payment_totals AS (
    SELECT
        p.order_id,
        SUM(p.payment_value) AS payment_value,
        COUNT(*) AS payment_count
    FROM payments AS p
    GROUP BY p.order_id
),
review_totals AS (
    SELECT
        r.order_id,
        AVG(r.review_score::NUMERIC) AS average_review_score,
        COUNT(*) AS review_count
    FROM reviews AS r
    GROUP BY r.order_id
)
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    COALESCE(it.item_count, 0) AS item_count,
    COALESCE(it.item_revenue, 0::NUMERIC) AS item_revenue,
    COALESCE(it.freight_revenue, 0::NUMERIC) AS freight_revenue,
    COALESCE(it.item_revenue, 0::NUMERIC)
        + COALESCE(it.freight_revenue, 0::NUMERIC) AS merchandise_and_freight_value,
    COALESCE(pt.payment_value, 0::NUMERIC) AS payment_value,
    COALESCE(pt.payment_count, 0) AS payment_count,
    rt.average_review_score,
    COALESCE(rt.review_count, 0) AS review_count
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
LEFT JOIN item_totals AS it ON it.order_id = o.order_id
LEFT JOIN payment_totals AS pt ON pt.order_id = o.order_id
LEFT JOIN review_totals AS rt ON rt.order_id = o.order_id;

COMMENT ON VIEW vw_order_revenue IS
    'Order-grain commercial metrics; item revenue excludes freight and canceled/unavailable filtering is left to consumers.';

-- Monthly item revenue for commercially valid orders, based on purchase month.
CREATE OR REPLACE VIEW vw_monthly_revenue AS
SELECT
    DATE_TRUNC('month', vor.order_purchase_timestamp)::DATE AS revenue_month,
    COUNT(*) AS order_count,
    SUM(vor.item_count) AS item_count,
    SUM(vor.item_revenue) AS item_revenue,
    SUM(vor.freight_revenue) AS freight_revenue,
    SUM(vor.merchandise_and_freight_value) AS merchandise_and_freight_value,
    AVG(vor.item_revenue) AS average_order_item_value
FROM vw_order_revenue AS vor
WHERE vor.order_status NOT IN ('canceled', 'unavailable')
GROUP BY DATE_TRUNC('month', vor.order_purchase_timestamp)::DATE;

COMMENT ON VIEW vw_monthly_revenue IS
    'Monthly revenue by purchase date, excluding canceled and unavailable orders.';

-- Revenue, demand, freight, and ratings by translated product category.
CREATE OR REPLACE VIEW vw_category_performance AS
WITH order_reviews AS (
    SELECT r.order_id, AVG(r.review_score::NUMERIC) AS review_score
    FROM reviews AS r
    GROUP BY r.order_id
),
category_items AS (
    SELECT
        oi.order_id,
        COALESCE(pct.product_category_name_english, p.product_category_name, 'unknown') AS category_name,
        oi.price,
        oi.freight_value
    FROM order_items AS oi
    JOIN orders AS o ON o.order_id = oi.order_id
    JOIN products AS p ON p.product_id = oi.product_id
    LEFT JOIN product_category_translation AS pct
        ON pct.product_category_name = p.product_category_name
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
category_commercial AS (
    SELECT
        ci.category_name,
        COUNT(DISTINCT ci.order_id) AS order_count,
        COUNT(*) AS item_count,
        SUM(ci.price) AS item_revenue,
        AVG(ci.price) AS average_item_price,
        AVG(ci.freight_value) AS average_freight_value
    FROM category_items AS ci
    GROUP BY ci.category_name
),
category_ratings AS (
    SELECT
        category_orders.category_name,
        AVG(orv.review_score) AS average_review_score
    FROM (
        SELECT DISTINCT ci.category_name, ci.order_id
        FROM category_items AS ci
    ) AS category_orders
    JOIN order_reviews AS orv ON orv.order_id = category_orders.order_id
    GROUP BY category_orders.category_name
)
SELECT
    cc.category_name,
    cc.order_count,
    cc.item_count,
    cc.item_revenue,
    cc.average_item_price,
    cc.average_freight_value,
    cr.average_review_score
FROM category_commercial AS cc
LEFT JOIN category_ratings AS cr ON cr.category_name = cc.category_name;

COMMENT ON VIEW vw_category_performance IS
    'Commercial and rating metrics by English category when a translation exists.';

-- Seller commercial performance and order-level customer rating.
CREATE OR REPLACE VIEW vw_seller_performance AS
WITH order_reviews AS (
    SELECT r.order_id, AVG(r.review_score::NUMERIC) AS review_score
    FROM reviews AS r
    GROUP BY r.order_id
),
seller_commercial AS (
    SELECT
        s.seller_id,
        s.seller_city,
        s.seller_state,
        COUNT(DISTINCT oi.order_id) AS order_count,
        COUNT(*) AS item_count,
        SUM(oi.price) AS item_revenue,
        SUM(oi.freight_value) AS freight_revenue
    FROM sellers AS s
    JOIN order_items AS oi ON oi.seller_id = s.seller_id
    JOIN orders AS o ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY s.seller_id, s.seller_city, s.seller_state
),
seller_ratings AS (
    SELECT
        seller_orders.seller_id,
        AVG(orv.review_score) AS average_review_score
    FROM (
        SELECT DISTINCT oi.seller_id, oi.order_id
        FROM order_items AS oi
        JOIN orders AS o ON o.order_id = oi.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
    ) AS seller_orders
    JOIN order_reviews AS orv ON orv.order_id = seller_orders.order_id
    GROUP BY seller_orders.seller_id
)
SELECT
    sc.seller_id,
    sc.seller_city,
    sc.seller_state,
    sc.order_count,
    sc.item_count,
    sc.item_revenue,
    sc.freight_revenue,
    sr.average_review_score
FROM seller_commercial AS sc
LEFT JOIN seller_ratings AS sr ON sr.seller_id = sc.seller_id;

COMMENT ON VIEW vw_seller_performance IS
    'Seller revenue, volume, freight, and ratings for non-canceled/non-unavailable orders.';

-- Delivered-order timing and late-delivery measures.
CREATE OR REPLACE VIEW vw_delivery_performance AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_state,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    EXTRACT(EPOCH FROM (
        o.order_delivered_customer_date - o.order_purchase_timestamp
    )) / 86400.0 AS delivery_days,
    EXTRACT(EPOCH FROM (
        o.order_delivered_customer_date - o.order_estimated_delivery_date
    )) / 86400.0 AS days_relative_to_estimate,
    (o.order_delivered_customer_date > o.order_estimated_delivery_date) AS was_late
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL;

COMMENT ON VIEW vw_delivery_performance IS
    'Delivered orders with elapsed days and lateness relative to the promised date.';

-- Order-grain view for analyzing whether delivery timeliness affects ratings.
CREATE OR REPLACE VIEW vw_review_delivery_relationship AS
WITH order_reviews AS (
    SELECT
        r.order_id,
        AVG(r.review_score::NUMERIC) AS average_review_score,
        COUNT(*) AS review_count
    FROM reviews AS r
    GROUP BY r.order_id
)
SELECT
    vdp.order_id,
    vdp.customer_state,
    vdp.delivery_days,
    vdp.days_relative_to_estimate,
    vdp.was_late,
    orv.average_review_score,
    orv.review_count
FROM vw_delivery_performance AS vdp
JOIN order_reviews AS orv ON orv.order_id = vdp.order_id;

COMMENT ON VIEW vw_review_delivery_relationship IS
    'Order-grain delivery timing paired with the mean score of all reviews for that order.';

-- Payment usage and value by method.
CREATE OR REPLACE VIEW vw_payment_summary AS
SELECT
    p.payment_type,
    COUNT(*) AS payment_record_count,
    COUNT(DISTINCT p.order_id) AS order_count,
    SUM(p.payment_value) AS total_payment_value,
    AVG(p.payment_value) AS average_payment_value,
    AVG(p.payment_installments::NUMERIC) AS average_installments
FROM payments AS p
GROUP BY p.payment_type;

COMMENT ON VIEW vw_payment_summary IS
    'Payment frequency, order coverage, value, and installment behavior by payment type.';
