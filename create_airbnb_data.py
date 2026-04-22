import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子
np.random.seed(42)
random.seed(42)

# 创建listings.csv数据
n_listings = 500
listings_data = {
    'id': range(1, n_listings + 1),
    'name': [f'Listing {i}' for i in range(1, n_listings + 1)],
    'host_id': np.random.randint(1000, 9999, n_listings),
    'host_name': [f'Host {i}' for i in range(1, n_listings + 1)],
    'neighbourhood_group': np.random.choice(['Ciutat Vella', 'Eixample', 'Sants-Montjuïc', 'Gràcia', 'Horta-Guinardó'], n_listings),
    'neighbourhood': np.random.choice(['Barri Gòtic', 'El Raval', 'Sant Pere', 'La Barceloneta', 'Sants', 'Poble Sec'], n_listings),
    'latitude': np.random.uniform(41.35, 41.45, n_listings),
    'longitude': np.random.uniform(2.1, 2.25, n_listings),
    'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_listings, p=[0.6, 0.3, 0.1]),
    'price': np.random.randint(50, 500, n_listings),
    'minimum_nights': np.random.randint(1, 14, n_listings),
    'number_of_reviews': np.random.randint(0, 200, n_listings),
    'last_review': [datetime.now() - timedelta(days=random.randint(0, 365)) if random.random() > 0.2 else None for _ in range(n_listings)],
    'reviews_per_month': np.random.uniform(0, 5, n_listings),
    'calculated_host_listings_count': np.random.randint(1, 10, n_listings),
    'availability_365': np.random.randint(0, 365, n_listings)
}

listings_df = pd.DataFrame(listings_data)
listings_df['last_review'] = listings_df['last_review'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
listings_df.to_csv('/workspace/data/listings.csv', index=False)

# 创建calendar.csv数据
n_days = 365
calendar_data = []
for listing_id in range(1, n_listings + 1):
    base_price = listings_df.loc[listings_df['id'] == listing_id, 'price'].values[0]
    for day in range(n_days):
        date = datetime.now() + timedelta(days=day)
        available = random.random() > 0.3  # 70%的概率可预订
        price = base_price * (0.8 + 0.4 * random.random())  # 价格在基础价格的80%-120%之间
        calendar_data.append({
            'listing_id': listing_id,
            'date': date.strftime('%Y-%m-%d'),
            'available': 't' if available else 'f',
            'price': f'${price:.2f}'
        })

calendar_df = pd.DataFrame(calendar_data)
calendar_df.to_csv('/workspace/data/calendar.csv', index=False)

# 创建reviews.csv数据
n_reviews = 2000
review_data = {
    'listing_id': np.random.randint(1, n_listings + 1, n_reviews),
    'id': range(1, n_reviews + 1),
    'date': [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(n_reviews)],
    'reviewer_id': np.random.randint(10000, 99999, n_reviews),
    'reviewer_name': [f'Reviewer {i}' for i in range(1, n_reviews + 1)],
    'comments': [random.choice(['Great place!', 'Clean and comfortable.', 'Excellent location.', 'Host was very helpful.', 'Would stay again.', 'Not as described.', 'Disappointed with the experience.']) for _ in range(n_reviews)]
}

reviews_df = pd.DataFrame(review_data)
reviews_df['date'] = reviews_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
reviews_df.to_csv('/workspace/data/reviews.csv', index=False)

print("模拟数据集创建完成！")
print(f"listings.csv: {len(listings_df)} 条记录")
print(f"calendar.csv: {len(calendar_df)} 条记录")
print(f"reviews.csv: {len(reviews_df)} 条记录")