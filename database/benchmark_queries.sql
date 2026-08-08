-- Enterprise AI SQL Analytics Copilot - Phase 1 benchmark queries
-- Revenue means order-item price and excludes canceled/unavailable orders unless noted.

-- 1. What is total revenue?
SELECT
    SUM(vor.item_revenue) AS total_item_revenue
FROM vw_order_revenue AS vor
WHERE vor.order_status NOT IN ('canceled', 'unavailable');

-- 2. What is the monthly revenue trend?
SELECT
    vmr.revenue_month,
    vmr.order_count,
    vmr.item_revenue,
    vmr.average_order_item_value
FROM vw_monthly_revenue AS vmr
ORDER BY vmr.revenue_month;

-- 3. Which product categories generate the most revenue?
SELECT
    vcp.category_name,
    vcp.item_revenue,
    vcp.order_count,
    vcp.item_count
FROM vw_category_performance AS vcp
ORDER BY vcp.item_revenue DESC NULLS LAST, vcp.category_name
LIMIT 20;

-- 4. Who are the top 10 sellers by revenue?
SELECT
    vsp.seller_id,
    vsp.seller_city,
    vsp.seller_state,
    vsp.item_revenue,
    vsp.order_count
FROM vw_seller_performance AS vsp
ORDER BY vsp.item_revenue DESC NULLS LAST, vsp.seller_id
LIMIT 10;

-- 5. Which customer states generate the most orders?
SELECT
    c.customer_state,
    COUNT(*) AS order_count
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY c.customer_state
ORDER BY order_count DESC, c.customer_state;

-- 6. What is the average order value?
SELECT
    AVG(vor.item_revenue) AS average_order_value
FROM vw_order_revenue AS vor
WHERE vor.order_status NOT IN ('canceled', 'unavailable')
  AND vor.item_count > 0;

-- 7. Which categories have the highest average review score?
SELECT
    vcp.category_name,
    vcp.average_review_score,
    vcp.order_count
FROM vw_category_performance AS vcp
WHERE vcp.average_review_score IS NOT NULL
ORDER BY vcp.average_review_score DESC, vcp.order_count DESC, vcp.category_name
LIMIT 20;

-- 8. Which sellers have high revenue but low ratings?
WITH seller_thresholds AS (
    SELECT
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vsp.item_revenue) AS high_revenue_cutoff,
        AVG(vsp.average_review_score) AS portfolio_average_rating
    FROM vw_seller_performance AS vsp
    WHERE vsp.average_review_score IS NOT NULL
)
SELECT
    vsp.seller_id,
    vsp.seller_city,
    vsp.seller_state,
    vsp.item_revenue,
    vsp.order_count,
    vsp.average_review_score
FROM vw_seller_performance AS vsp
CROSS JOIN seller_thresholds AS st
WHERE vsp.item_revenue >= st.high_revenue_cutoff
  AND vsp.average_review_score < st.portfolio_average_rating
ORDER BY vsp.item_revenue DESC, vsp.average_review_score ASC;

-- 9. What percentage of delivered orders were late?
SELECT
    COUNT(*) FILTER (WHERE vdp.was_late) AS late_order_count,
    COUNT(*) AS delivered_order_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE vdp.was_late) / NULLIF(COUNT(*), 0),
        2
    ) AS late_delivery_percentage
FROM vw_delivery_performance AS vdp;

-- 10. Are delayed deliveries associated with lower review scores?
SELECT
    CASE WHEN vrdr.was_late THEN 'late' ELSE 'on_time_or_early' END AS delivery_status,
    COUNT(*) AS reviewed_order_count,
    AVG(vrdr.average_review_score) AS average_review_score
FROM vw_review_delivery_relationship AS vrdr
GROUP BY CASE WHEN vrdr.was_late THEN 'late' ELSE 'on_time_or_early' END
ORDER BY delivery_status;

-- 11. Which payment method is used most often?
SELECT
    vps.payment_type,
    vps.order_count,
    vps.payment_record_count,
    vps.total_payment_value
FROM vw_payment_summary AS vps
ORDER BY vps.order_count DESC, vps.payment_type;

-- 12. What is the average freight cost by customer state?
SELECT
    c.customer_state,
    AVG(oi.freight_value) AS average_freight_value,
    COUNT(*) AS item_count
FROM order_items AS oi
JOIN orders AS o ON o.order_id = oi.order_id
JOIN customers AS c ON c.customer_id = o.customer_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY c.customer_state
ORDER BY average_freight_value DESC, c.customer_state;

-- 13. Which product categories have the highest average freight cost?
SELECT
    vcp.category_name,
    vcp.average_freight_value,
    vcp.item_count
FROM vw_category_performance AS vcp
ORDER BY vcp.average_freight_value DESC NULLS LAST, vcp.category_name
LIMIT 20;

-- 14. What is the average delivery time?
SELECT
    AVG(vdp.delivery_days) AS average_delivery_days,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vdp.delivery_days) AS median_delivery_days
FROM vw_delivery_performance AS vdp;

-- 15. Which months have the highest order volume?
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::DATE AS purchase_month,
    COUNT(*) AS order_count
FROM orders AS o
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)::DATE
ORDER BY order_count DESC, purchase_month
LIMIT 12;

-- 16. What are the top-selling products?
SELECT
    oi.product_id,
    COALESCE(pct.product_category_name_english, p.product_category_name, 'unknown') AS category_name,
    COUNT(*) AS units_sold,
    COUNT(DISTINCT oi.order_id) AS order_count,
    SUM(oi.price) AS item_revenue
FROM order_items AS oi
JOIN orders AS o ON o.order_id = oi.order_id
JOIN products AS p ON p.product_id = oi.product_id
LEFT JOIN product_category_translation AS pct
    ON pct.product_category_name = p.product_category_name
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY
    oi.product_id,
    COALESCE(pct.product_category_name_english, p.product_category_name, 'unknown')
ORDER BY units_sold DESC, item_revenue DESC, oi.product_id
LIMIT 20;

-- 17. Which sellers have the highest number of orders?
SELECT
    vsp.seller_id,
    vsp.seller_city,
    vsp.seller_state,
    vsp.order_count,
    vsp.item_revenue
FROM vw_seller_performance AS vsp
ORDER BY vsp.order_count DESC, vsp.item_revenue DESC, vsp.seller_id
LIMIT 20;

-- 18. Which states have the highest average order value?
SELECT
    vor.customer_state,
    COUNT(*) AS order_count,
    AVG(vor.item_revenue) AS average_order_value,
    SUM(vor.item_revenue) AS total_item_revenue
FROM vw_order_revenue AS vor
WHERE vor.order_status NOT IN ('canceled', 'unavailable')
  AND vor.item_count > 0
GROUP BY vor.customer_state
ORDER BY average_order_value DESC, vor.customer_state;

-- 19. What percentage of customers made repeat purchases?
WITH customer_order_counts AS (
    SELECT
        c.customer_unique_id,
        COUNT(*) AS order_count
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY c.customer_unique_id
)
SELECT
    COUNT(*) FILTER (WHERE coc.order_count > 1) AS repeat_customer_count,
    COUNT(*) AS purchasing_customer_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE coc.order_count > 1) / NULLIF(COUNT(*), 0),
        2
    ) AS repeat_customer_percentage
FROM customer_order_counts AS coc;

-- 20. Which categories combine high revenue with low customer ratings?
WITH category_thresholds AS (
    SELECT
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vcp.item_revenue) AS high_revenue_cutoff,
        AVG(vcp.average_review_score) AS portfolio_average_rating
    FROM vw_category_performance AS vcp
    WHERE vcp.average_review_score IS NOT NULL
)
SELECT
    vcp.category_name,
    vcp.item_revenue,
    vcp.order_count,
    vcp.average_review_score
FROM vw_category_performance AS vcp
CROSS JOIN category_thresholds AS ct
WHERE vcp.item_revenue >= ct.high_revenue_cutoff
  AND vcp.average_review_score < ct.portfolio_average_rating
ORDER BY vcp.item_revenue DESC, vcp.average_review_score ASC;

-- 21. How does order cancellation rate vary by month?
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::DATE AS purchase_month,
    COUNT(*) AS total_order_count,
    COUNT(*) FILTER (WHERE o.order_status = 'canceled') AS canceled_order_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE o.order_status = 'canceled')
        / NULLIF(COUNT(*), 0),
        2
    ) AS cancellation_percentage
FROM orders AS o
GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)::DATE
ORDER BY purchase_month;

-- 22. Which customer states experience the longest average delivery times?
SELECT
    vdp.customer_state,
    COUNT(*) AS delivered_order_count,
    AVG(vdp.delivery_days) AS average_delivery_days,
    AVG(vdp.days_relative_to_estimate) AS average_days_relative_to_estimate
FROM vw_delivery_performance AS vdp
GROUP BY vdp.customer_state
ORDER BY average_delivery_days DESC, vdp.customer_state;
