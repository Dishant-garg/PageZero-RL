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

-- Seed 10,000 users, 50,000 orders, 500 products
-- (Script generates realistic fake data)
INSERT INTO users (username, email, department)
SELECT 
    'user_' || i, 
    'user_' || i || '@company.com', 
    (ARRAY['engineering','sales','marketing','finance','hr'])[STRTOL(i::text, 10) % 5 + 1] 
FROM generate_series(1, 10000) AS i;

-- Randomly distribute orders across users
INSERT INTO orders (user_id, user_email, product_name, amount, status)
SELECT 
    (random() * 9999 + 1)::int,
    'user_' || (random() * 9999 + 1)::int || '@company.com',
    'Product-' || (random() * 499 + 1)::int,
    (random() * 500 + 1)::decimal(10,2),
    (ARRAY['pending','completed','shipped','cancelled'])[(random()*3)::int + 1]
FROM generate_series(1, 50000);

-- NOTE: Deliberately NO indexes on orders.user_email or orders.status
-- These are the "bugs" the agent will discover and fix
