import os
import sqlite3

def create_amazon_db():
    conn = sqlite3.connect("Database/Amazon.db")
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        description TEXT,
        stock_quantity INTEGER NOT NULL,
        rating REAL
    )
    ''')
    
    # Create orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        order_date TEXT NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT NOT NULL,
        shipping_address TEXT NOT NULL
    )
    ''')
    
    # Create order_items table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Create delivery table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS delivery (
        delivery_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        carrier TEXT NOT NULL,
        tracking_number TEXT,
        estimated_delivery_date TEXT,
        actual_delivery_date TEXT,
        delivery_status TEXT NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id)
    )
    ''')
    
    # Insert sample data into products
    sample_products = [
        ('P001', 'Wireless Earbuds', 'Electronics', 79.99, 'Bluetooth wireless earbuds with noise cancellation', 120, 4.5),
        ('P002', 'Smart Watch', 'Electronics', 199.99, 'Fitness tracker with heart rate monitor', 85, 4.7),
        ('P003', 'Cotton T-Shirt', 'Clothing', 24.99, 'Soft cotton t-shirt in various colors', 300, 4.2),
        ('P004', 'Mystery Novel', 'Books', 14.99, 'Bestselling thriller novel', 200, 4.8),
        ('P005', 'Yoga Mat', 'Sports', 29.99, 'Non-slip exercise yoga mat', 150, 4.6)
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO products (product_id, name, category, price, description, stock_quantity, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', sample_products)
    
    # Insert sample data into orders
    sample_orders = [
        ('O001', 'C001', '2024-04-01', 79.99, 'Delivered', '123 Main St, City, State 12345'),
        ('O002', 'C002', '2024-04-05', 224.98, 'Shipped', '456 Oak Ave, Town, State 23456'),
        ('O003', 'C001', '2024-04-10', 14.99, 'Processing', '123 Main St, City, State 12345'),
        ('O004', 'C003', '2024-04-15', 29.99, 'Cancelled', '789 Pine Rd, Village, State 34567'),
        ('O005', 'C004', '2024-04-20', 279.98, 'Delivered', '321 Elm Blvd, County, State 45678')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO orders (order_id, customer_id, order_date, total_amount, status, shipping_address)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_orders)
    
    # Insert sample data into order_items
    sample_order_items = [
        ('O001', 'P001', 1, 79.99),
        ('O002', 'P002', 1, 199.99),
        ('O002', 'P003', 1, 24.99),
        ('O003', 'P004', 1, 14.99),
        ('O004', 'P005', 1, 29.99),
        ('O005', 'P001', 1, 79.99),
        ('O005', 'P002', 1, 199.99)
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO order_items (order_id, product_id, quantity, price)
    VALUES (?, ?, ?, ?)
    ''', sample_order_items)
    
    # Insert sample data into delivery
    sample_deliveries = [
        ('D001', 'O001', 'UPS', 'UPS12345678', '2024-04-05', '2024-04-04', 'Delivered'),
        ('D002', 'O002', 'FedEx', 'FDX87654321', '2024-04-10', None, 'In Transit'),
        ('D003', 'O003', None, None, '2024-04-15', None, 'Processing'),
        ('D004', 'O004', None, None, None, None, 'Cancelled'),
        ('D005', 'O005', 'USPS', 'USPS23456789', '2024-04-25', '2024-04-23', 'Delivered')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO delivery (delivery_id, order_id, carrier, tracking_number, estimated_delivery_date, actual_delivery_date, delivery_status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', sample_deliveries)
    
    conn.commit()
    conn.close()

def setup():
    os.makedirs("Database", exist_ok=True)
    os.makedirs("Data", exist_ok=True)
    create_amazon_db()

    # Create policy.txt if it doesn't exist
    if not os.path.exists("Data/policy.txt"):
        with open("Data/policy.txt", "w") as f:
            f.write("""
        ELECTRONICS PRODUCT POLICY

        Effective Date: [Insert Date]
        Company: [Your Company Name]
        Contact: [email@example.com] | [Phone Number] | [Address]

        1. PRODUCT WARRANTY
        -------------------
        We offer a [12]-month limited warranty on all electronics products purchased directly from [Your Company Name] or authorized retailers. This warranty covers defects in materials and workmanship under normal use.

        1.1 WARRANTY COVERAGE
        - Hardware defects due to manufacturing issues
        - Battery failure (if applicable)
        - Malfunctioning internal components (CPU, display, sensors, etc.)

        1.2 WARRANTY EXCLUSIONS
        This warranty does not cover:
        - Damage caused by misuse, accidents, or unauthorized repairs
        - Normal wear and tear (scratches, battery degradation)
        - Damage due to exposure to liquids (unless rated waterproof)
        - Software-related issues due to third-party applications or user error

        1.3 WARRANTY CLAIM PROCESS
        To initiate a warranty claim:
        - Provide proof of purchase (invoice or receipt)
        - Describe the issue in detail
        - Ship the product to our service center (shipping cost may apply)

        2. RETURN AND REFUND POLICY
        ----------------------------
        We accept returns within 30 days of delivery for most electronics, provided the item is:
        - In original packaging
        - Unused or minimally used
        - Accompanied by a valid proof of purchase

        2.1 NON-RETURNABLE ITEMS
        - Opened software or digital downloads
        - Customized or personalized products
        - Items marked as final sale or clearance

        2.2 REFUND PROCESS
        Refunds will be processed within 7–10 business days after receiving the returned item and verifying its condition. Refunds will be issued to the original payment method.

        3. TECHNICAL SUPPORT
        ---------------------
        We provide free technical support for up to 6 months from the purchase date. Support is available through:
        - Email: [support@example.com]
        - Live Chat: [Your Website]
        - Phone: [Support Line]

        3.1 SUPPORT HOURS
        Monday–Friday: 9:00 AM – 6:00 PM (Local Time)
        Saturday–Sunday: Closed

        4. PRODUCT USAGE & SAFETY
        --------------------------
        Customers are expected to follow usage instructions as provided in the user manual. Improper use may void the warranty and pose safety risks.

        4.1 SAFETY PRECAUTIONS
        - Do not expose the product to extreme temperatures or moisture.
        - Use only compatible accessories and chargers.
        - Keep away from small children unless otherwise stated.

        5. PRIVACY POLICY
        ------------------
        Personal information collected during registration, purchase, or support interactions is processed in accordance with our Privacy Policy [link to full policy].

        6. DISCLAIMER OF LIABILITY
        ---------------------------
        [Your Company Name] is not liable for any indirect, incidental, or consequential damages resulting from the use or misuse of the product. Responsibility is limited to repair or replacement under warranty terms.

        7. GOVERNING LAW
        -----------------
        This policy is governed by the laws of [Your Country/State]. Disputes will be resolved through arbitration or in the courts of competent jurisdiction.

        For questions or concerns, contact us at: [email@example.com]
                    """)


