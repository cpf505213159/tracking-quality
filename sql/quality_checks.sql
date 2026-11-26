-- 六条埋点质检规则的 SQL 版本（MySQL 8.0）
-- 数仓里直接跑，口径与 Python 版完全一致；两边数字必须对得上。
-- 表：dwd_event_log(event_id, user_id, device_id, session_id, event,
--                   event_time, server_time, platform, price, utm_source)

-- ========== R1 完整性：各端「加购/商详」比值 vs 90 分位基线 ==========
WITH daily_ratio AS (
    SELECT
        DATE(event_time)                                              AS dt,
        platform,
        SUM(event = 'detail')                                         AS detail_cnt,
        SUM(event = 'add_to_cart')                                    AS cart_cnt
    FROM dwd_event_log
    GROUP BY DATE(event_time), platform
),
baseline AS (
    SELECT
        platform,
        -- MySQL 8 没有 percentile_disc，用窗口函数取 90 分位最近的一行
        MAX(CASE WHEN rn >= 0.9 * total THEN ratio END)               AS p90_ratio
    FROM (
        SELECT
            platform,
            cart_cnt / NULLIF(detail_cnt, 0)                          AS ratio,
            ROW_NUMBER() OVER (PARTITION BY platform ORDER BY cart_cnt / NULLIF(detail_cnt, 0)) AS rn,
            COUNT(*) OVER (PARTITION BY platform)                     AS total
        FROM daily_ratio
        WHERE detail_cnt >= 200                                       -- 小样本平台日不参与判定
    ) t
    GROUP BY platform
)
SELECT
    d.dt, d.platform, d.detail_cnt,
    ROUND(d.cart_cnt / NULLIF(d.detail_cnt, 0), 4)                    AS ratio,
    ROUND(b.p90_ratio, 4)                                             AS baseline,
    ROUND(1 - (d.cart_cnt / NULLIF(d.detail_cnt, 0)) / b.p90_ratio, 4) AS drop_ratio
FROM daily_ratio d
JOIN baseline b ON b.platform = d.platform
WHERE d.detail_cnt >= 200
  AND (d.cart_cnt / NULLIF(d.detail_cnt, 0)) < b.p90_ratio * 0.6      -- 跌破 40% 判定异常
ORDER BY d.dt, d.platform;

-- ========== R2 唯一性：event_id 重复率 ==========
SELECT
    COUNT(*)                                                    AS total_rows,
    COUNT(DISTINCT event_id)                                    AS unique_ids,
    COUNT(*) - COUNT(DISTINCT event_id)                         AS dup_rows,
    ROUND((COUNT(*) - COUNT(DISTINCT event_id)) / COUNT(*), 4)  AS dup_rate
FROM dwd_event_log;

-- 重复明细（定位到具体 event_id，便于找上报组件）
SELECT event_id, COUNT(*) AS n, MIN(server_time) AS first_seen
FROM dwd_event_log
GROUP BY event_id
HAVING COUNT(*) > 1
ORDER BY n DESC
LIMIT 100;

-- ========== R3 及时性：整点 p95 延迟 + 断流 ==========
WITH hourly AS (
    SELECT
        DATE_FORMAT(event_time, '%Y-%m-%d %H:00:00')            AS hour_slot,
        HOUR(event_time)                                        AS hod,
        COUNT(*)                                                AS n,
        MAX(TIMESTAMPDIFF(SECOND, event_time, server_time))     AS max_latency_s
    FROM dwd_event_log
    GROUP BY hour_slot, hod
),
expected AS (
    -- 同一小时的常态量，用来判断「这个点本来就该有数据」
    SELECT hod, AVG(n) AS avg_n FROM hourly GROUP BY hod
)
SELECT h.hour_slot, h.n, h.max_latency_s, ROUND(e.avg_n, 1) AS expected_n,
       CASE WHEN h.max_latency_s > 600 THEN 'delay'
            WHEN h.n = 0 AND e.avg_n >= 5 THEN 'gap'
       END AS issue
FROM hourly h
JOIN expected e ON e.hod = h.hod
WHERE h.max_latency_s > 600 OR (h.n = 0 AND e.avg_n >= 5)
ORDER BY h.hour_slot;

-- ========== R4 一致性：会话内下单早于商详 ==========
WITH sess AS (
    SELECT
        session_id,
        MIN(CASE WHEN event = 'detail' THEN event_time END) AS detail_at,
        MIN(CASE WHEN event = 'order'  THEN event_time END) AS order_at
    FROM dwd_event_log
    WHERE event IN ('detail', 'order')
    GROUP BY session_id
)
SELECT
    COUNT(*)                                                    AS valid_sessions,
    SUM(order_at < detail_at)                                   AS inverted_sessions,
    ROUND(SUM(order_at < detail_at) / COUNT(*), 4)              AS inverted_rate
FROM sess
WHERE detail_at IS NOT NULL AND order_at IS NOT NULL;

-- ========== R5 有效性：枚举越界 / 金额非正 ==========
SELECT
    SUM(platform NOT IN ('ios', 'android', 'web'))                              AS bad_platform,
    SUM(event = 'order' AND (price IS NULL OR price <= 0))                      AS bad_price,
    SUM(device_id IS NULL OR device_id = '')                                    AS missing_device,
    ROUND((SUM(platform NOT IN ('ios','android','web'))
           + SUM(event = 'order' AND (price IS NULL OR price <= 0))) / COUNT(*), 4) AS bad_rate
FROM dwd_event_log;

-- ========== R6 设备一致性：一号多用户 ==========
WITH dev AS (
    SELECT device_id,
           COUNT(DISTINCT user_id) AS user_cnt,
           COUNT(DISTINCT session_id) AS session_cnt
    FROM dwd_event_log
    WHERE device_id IS NOT NULL AND device_id <> ''
    GROUP BY device_id
)
SELECT
    COUNT(*)                                        AS devices,
    SUM(user_cnt > 1)                               AS reused_devices,
    ROUND(SUM(user_cnt > 1) / COUNT(*), 4)          AS reused_rate,
    SUM(CASE WHEN user_cnt > 1 THEN session_cnt ELSE 0 END) AS affected_sessions
FROM dev;
