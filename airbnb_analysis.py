import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据读取与初步探索
def load_data():
    print("\n=== 1. 数据读取与初步探索 ===")
    listings_df = pd.read_csv('/workspace/data/listings.csv')
    calendar_df = pd.read_csv('/workspace/data/calendar.csv')
    reviews_df = pd.read_csv('/workspace/data/reviews.csv')
    
    print(f"listings.csv 形状: {listings_df.shape}")
    print(f"calendar.csv 形状: {calendar_df.shape}")
    print(f"reviews.csv 形状: {reviews_df.shape}")
    
    print("\nlistings.csv 前5行:")
    print(listings_df.head())
    
    print("\ncalendar.csv 前5行:")
    print(calendar_df.head())
    
    print("\nreviews.csv 前5行:")
    print(reviews_df.head())
    
    # 检查缺失值
    print("\nlistings.csv 缺失值情况:")
    print(listings_df.isnull().sum())
    
    print("\ncalendar.csv 缺失值情况:")
    print(calendar_df.isnull().sum())
    
    print("\nreviews.csv 缺失值情况:")
    print(reviews_df.isnull().sum())
    
    return listings_df, calendar_df, reviews_df

# 2. 数据预处理
def preprocess_data(listings_df, calendar_df, reviews_df):
    print("\n=== 2. 数据预处理 ===")
    
    # 处理listings.csv
    # 填充缺失值
    listings_df['reviews_per_month'] = listings_df['reviews_per_month'].fillna(0)
    listings_df['last_review'] = listings_df['last_review'].fillna('2000-01-01')
    
    # 处理calendar.csv
    # 转换price为数值型
    calendar_df['price'] = calendar_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # 处理reviews.csv
    # 填充缺失值
    reviews_df['comments'] = reviews_df['comments'].fillna('No comment')
    
    return listings_df, calendar_df, reviews_df

# 3. 探索性数据分析（EDA）
def perform_eda(listings_df, calendar_df, reviews_df):
    print("\n=== 3. 探索性数据分析（EDA）===")
    
    # 创建输出目录
    import os
    if not os.path.exists('/workspace/figures'):
        os.makedirs('/workspace/figures')
    
    # 3.1 描述性统计
    print("\nlistings.csv 描述性统计:")
    print(listings_df.describe())
    
    # 3.2 分布分析
    # 房源类型分布
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=listings_df)
    plt.title('Room Type Distribution')
    plt.savefig('/workspace/figures/room_type_distribution.png')
    plt.close()
    
    # 价格分布
    plt.figure(figsize=(10, 6))
    sns.histplot(listings_df['price'], bins=50)
    plt.title('Price Distribution')
    plt.savefig('/workspace/figures/price_distribution.png')
    plt.close()
    
    # 3.3 相关性分析
    numeric_cols = listings_df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = listings_df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('/workspace/figures/correlation_matrix.png')
    plt.close()
    
    # 3.4 区域房源分布
    plt.figure(figsize=(12, 8))
    sns.countplot(y='neighbourhood_group', data=listings_df, order=listings_df['neighbourhood_group'].value_counts().index)
    plt.title('Neighbourhood Group Distribution')
    plt.savefig('/workspace/figures/neighbourhood_distribution.png')
    plt.close()
    
    # 3.5 价格与区域关系
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='neighbourhood_group', y='price', data=listings_df)
    plt.title('Price by Neighbourhood Group')
    plt.xticks(rotation=45)
    plt.savefig('/workspace/figures/price_by_neighbourhood.png')
    plt.close()
    
    print("EDA 完成，图表已保存到 /workspace/figures 目录")

# 4. 特征工程
def feature_engineering(listings_df, calendar_df, reviews_df):
    print("\n=== 4. 特征工程 ===")
    
    # 4.1 基于calendar.csv提取特征
    # 计算每个房源的平均价格和价格方差
    calendar_features = calendar_df.groupby('listing_id').agg({
        'price': ['mean', 'std'],
        'available': lambda x: sum(x == 't') / len(x)
    }).reset_index()
    calendar_features.columns = ['listing_id', 'avg_price', 'price_std', 'availability_rate']
    
    # 4.2 基于reviews.csv提取特征
    # 计算每个房源的评论数量和评论情感得分
    def sentiment_score(comment):
        positive_words = ['great', 'excellent', 'helpful', 'clean', 'comfortable', 'stay again']
        negative_words = ['disappointed', 'not as described', 'bad', 'poor']
        score = 0
        for word in positive_words:
            if word in comment.lower():
                score += 1
        for word in negative_words:
            if word in comment.lower():
                score -= 1
        return score
    
    reviews_df['sentiment'] = reviews_df['comments'].apply(sentiment_score)
    review_features = reviews_df.groupby('listing_id').agg({
        'id': 'count',
        'sentiment': 'mean'
    }).reset_index()
    review_features.columns = ['listing_id', 'review_count', 'avg_sentiment']
    
    # 4.3 合并特征
    # 合并listings_df与calendar_features
    merged_df = pd.merge(listings_df, calendar_features, left_on='id', right_on='listing_id', how='left')
    # 合并review_features
    merged_df = pd.merge(merged_df, review_features, left_on='id', right_on='listing_id', how='left')
    
    # 填充缺失值
    merged_df['avg_price'] = merged_df['avg_price'].fillna(merged_df['price'])
    merged_df['price_std'] = merged_df['price_std'].fillna(0)
    merged_df['availability_rate'] = merged_df['availability_rate'].fillna(0)
    merged_df['review_count'] = merged_df['review_count'].fillna(0)
    merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
    
    # 4.4 特征转换
    # 编码类别特征
    le = LabelEncoder()
    merged_df['room_type_encoded'] = le.fit_transform(merged_df['room_type'])
    merged_df['neighbourhood_group_encoded'] = le.fit_transform(merged_df['neighbourhood_group'])
    
    # 4.5 选择特征
    features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                'calculated_host_listings_count', 'availability_365', 'avg_price',
                'price_std', 'availability_rate', 'review_count', 'avg_sentiment',
                'room_type_encoded', 'neighbourhood_group_encoded']
    
    # 目标变量：我们将使用neighbourhood_group作为目标变量进行分类
    target = 'neighbourhood_group'
    
    return merged_df, features, target

# 5. 模型构建与训练
def build_and_train_models(merged_df, features, target):
    print("\n=== 5. 模型构建与训练 ===")
    
    # 准备数据
    X = merged_df[features]
    y = merged_df[target]
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5.1 逻辑回归模型
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # 5.2 随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 预测
    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test)
    
    return lr_model, rf_model, X_test, y_test, lr_pred, rf_pred, X_train, y_train

# 6. 模型评估与对比
def evaluate_models(y_test, lr_pred, rf_pred, rf_model, features):
    print("\n=== 6. 模型评估与对比 ===")
    
    # 计算评估指标
    metrics = {}
    
    # 逻辑回归
    metrics['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, lr_pred),
        'Precision': precision_score(y_test, lr_pred, average='weighted'),
        'Recall': recall_score(y_test, lr_pred, average='weighted'),
        'F1 Score': f1_score(y_test, lr_pred, average='weighted')
    }
    
    # 随机森林
    metrics['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred, average='weighted'),
        'Recall': recall_score(y_test, rf_pred, average='weighted'),
        'F1 Score': f1_score(y_test, rf_pred, average='weighted')
    }
    
    # 打印评估结果
    print("\n模型评估结果:")
    for model_name, scores in metrics.items():
        print(f"\n{model_name}:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
    
    # 特征重要性
    print("\n随机森林特征重要性:")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.savefig('/workspace/figures/feature_importance.png')
    plt.close()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('/workspace/figures/confusion_matrix.png')
    plt.close()
    
    return metrics

# 7. 主函数
def main():
    # 1. 数据读取与初步探索
    listings_df, calendar_df, reviews_df = load_data()
    
    # 2. 数据预处理
    listings_df, calendar_df, reviews_df = preprocess_data(listings_df, calendar_df, reviews_df)
    
    # 3. 探索性数据分析（EDA）
    perform_eda(listings_df, calendar_df, reviews_df)
    
    # 4. 特征工程
    merged_df, features, target = feature_engineering(listings_df, calendar_df, reviews_df)
    
    # 5. 模型构建与训练
    lr_model, rf_model, X_test, y_test, lr_pred, rf_pred, X_train, y_train = build_and_train_models(merged_df, features, target)
    
    # 6. 模型评估与对比
    metrics = evaluate_models(y_test, lr_pred, rf_pred, rf_model, features)
    
    # 7. 保存结果
    print("\n=== 7. 保存结果 ===")
    # 保存合并后的数据集
    merged_df.to_csv('/workspace/data/merged_data.csv', index=False)
    
    # 保存评估结果
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('/workspace/results/metrics.csv')
    
    print("实验完成！结果已保存。")

if __name__ == "__main__":
    # 创建结果目录
    import os
    if not os.path.exists('/workspace/results'):
        os.makedirs('/workspace/results')
    main()