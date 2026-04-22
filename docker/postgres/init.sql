-- PageZero Production DB Schema
-- Tables are intentionally sparse on indexes — the agent will discover and fix them.

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    department VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    user_email VARCHAR(255),
    product_name VARCHAR(200),
    amount DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200),
    price DECIMAL(10,2),
    stock INTEGER DEFAULT 100,
    category VARCHAR(50)
);

-- Seed 10,000 users
INSERT INTO users (username, email, department)
SELECT 
    'user_' || i, 
    'user_' || i || '@company.com', 
    (ARRAY['engineering','sales','marketing','finance','hr'])[i % 5 + 1]
FROM generate_series(1, 10000) AS i;

-- Seed 50,000 orders — no index on user_email intentionally!
INSERT INTO orders (user_id, user_email, product_name, amount, status)
SELECT 
    (random() * 9999 + 1)::int,
    'user_' || (random() * 9999 + 1)::int || '@company.com',
    'Product-' || (random() * 499 + 1)::int,
    (random() * 500 + 1)::decimal(10,2),
    (ARRAY['pending','completed','shipped','cancelled'])[(random()*3)::int + 1]
FROM generate_series(1, 50000);

-- Seed 500 products
INSERT INTO products (name, price, stock, category)
SELECT
    'Product-' || i,
    (random() * 500 + 5)::decimal(10,2),
    (random() * 200)::int,
    (ARRAY['electronics','clothing','food','books','tools'])[i % 5 + 1]
FROM generate_series(1, 500) AS i;

-- NOTE: Deliberately NO indexes on orders.user_email or orders.status
-- These are the "bugs" the SRE agent will discover and fix via pg_create_index
