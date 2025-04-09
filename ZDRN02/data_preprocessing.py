
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


# 定义停用词列表（简单示例，可根据需要扩展）
stopwords = ["的", "是", "有", "在", "和", "等", "也", "这", "那", "了"]

# 定义珠宝知识类别
jewelry_classes = ["钻石", "翡翠", "珍珠", "黄金"]
num_classes = len(jewelry_classes)

# 定义知识类型类别
knowledge_types = ["特点", "鉴别", "保养", "文化历史", "搭配", "适合人群"]
num_knowledge_types = len(knowledge_types)

# 简化后的知识信息字典
jewelry_extended_info = {
    "钻石": {
        "特点": [{"info": "硬度高、光泽好、化学稳定", "suitable_for": ["所有人"]}],
        "鉴别": [{"info": "看硬度、切工、折射率", "suitable_for": ["所有人"]}],
        "保养": [{"info": "避免碰撞、定期清洁、单独存放", "suitable_for": ["所有人"]}],
        "文化历史": [{"info": "象征永恒爱情，曾是权力象征", "suitable_for": ["所有人"]}],
        "搭配": [{"info": "简约金属项链", "suitable_for": ["年轻人"]}],
        "适合人群": [{"info": "追求时尚年轻人", "suitable_for": ["年轻人"]}]
    },
    "翡翠": {
        "特点": [{"info": "质地温润、颜色多样、有灵性", "suitable_for": ["所有人"]}],
        "鉴别": [{"info": "看光泽、内部结构、听声音", "suitable_for": ["所有人"]}],
        "保养": [{"info": "避免化学物、单独存放、定期补水", "suitable_for": ["所有人"]}],
        "文化历史": [{"info": "传统饰品，寓意吉祥", "suitable_for": ["所有人"]}],
        "搭配": [{"info": "中式服装", "suitable_for": ["所有人"]}],
        "适合人群": [{"info": "喜爱传统文化者", "suitable_for": ["所有人"]}]
    },
    "珍珠": {
        "特点": [{"info": "光泽柔和、天然形成、有保健作用", "suitable_for": ["所有人"]}],
        "鉴别": [{"info": "看表面纹理、感受重量、摩擦法", "suitable_for": ["所有人"]}],
        "保养": [{"info": "避免暴晒、软布擦拭、防硬物摩擦", "suitable_for": ["所有人"]}],
        "文化历史": [{"info": "皇室饰品、曾作货币、象征纯洁", "suitable_for": ["所有人"]}],
        "搭配": [{"info": "白色连衣裙", "suitable_for": ["女性", "年轻人"]}],
        "适合人群": [{"info": "优雅女士", "suitable_for": ["女性"]}]
    },
    "黄金": {
        "特点": [{"info": "延展性好、导电性强、抗腐蚀", "suitable_for": ["所有人"]}],
        "鉴别": [{"info": "测试硬度、查看印记、化学试剂测试", "suitable_for": ["所有人"]}],
        "保养": [{"info": "避免接触化妆品、定期清洁", "suitable_for": ["所有人"]}],
        "文化历史": [{"info": "长期作为货币和财富象征", "suitable_for": ["所有人"]}],
        "搭配": [{"info": "简约日常服饰", "suitable_for": ["所有人"]}],
        "适合人群": [{"info": "适合各年龄段人群", "suitable_for": ["所有人"]}]
    }
}


# 提取关键词特征，增加停用词处理
def extract_keyword_features(text):
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(filtered_words)])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_features = tfidf_matrix.toarray()[0]
    keyword_features = []
    for i, feature in enumerate(tfidf_features):
        if feature > 0:
            keyword_features.append((feature_names[i], feature))
    return keyword_features


def convert_to_balance_data(jewelry_extended_info):
    data_samples = []
    for jewelry in jewelry_classes:
        for knowledge in knowledge_types:
            info_list = jewelry_extended_info.get(jewelry, {}).get(knowledge, [])
            for info in info_list:
                keyword_features = extract_keyword_features(info["info"])
                feature_vector = [feature[1] for feature in keyword_features]
                sample = {
                    "class_label": f"{jewelry}-{knowledge}",
                    "features": feature_vector,
                    "info": info["info"],
                    "suitable_for": info["suitable_for"]
                }
                data_samples.append(sample)
    return data_samples


# 数据预处理
def preprocess_data(data_samples):
    texts = [sample['text'] for sample in data_samples]
    labels = [sample['class_label'] for sample in data_samples]

    # 中文分词
    texts = [" ".join(jieba.lcut(text)) for text in texts]

    # 使用 TF-IDF 提取文本特征
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).toarray()

    y = np.array(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if np.isnan(X).any() or np.isinf(X).any():
        import logging
        logging.warning("Scaled data contains NaN or Inf values.")
    return X, y


# 自定义抽样
def custom_sampling(X, y, base_count_factor=0.8):
    df = pd.DataFrame(X)
    df['label'] = y
    class_counts = df['label'].value_counts()
    base_count = int(np.mean(class_counts) * base_count_factor)
    resampled_dfs = []
    for class_label in class_counts.index:
        class_df = df[df['label'] == class_label]
        if len(class_df) > base_count:
            resampled_df = class_df.sample(base_count, random_state=42)
        else:
            num_to_sample = base_count - len(class_df)
            if num_to_sample > 0:
                class_features = class_df.drop('label', axis=1).values
                class_mean = np.mean(class_features, axis=0)
                class_cov = np.cov(class_features, rowvar=False)
                try:
                    np.linalg.cholesky(class_cov)
                except np.linalg.LinAlgError:
                    class_cov += np.eye(class_cov.shape[0]) * 1e-6
                new_samples = np.random.multivariate_normal(class_mean, class_cov, num_to_sample)
                new_labels = np.full((num_to_sample,), class_label)
                new_df = pd.DataFrame(new_samples)
                new_df['label'] = new_labels
                resampled_df = pd.concat([class_df, new_df])
            else:
                resampled_df = class_df
        resampled_dfs.append(resampled_df)
    resampled_df = pd.concat(resampled_dfs)
    X_resampled = resampled_df.drop('label', axis=1).values
    y_resampled = resampled_df['label'].values
    return X_resampled, y_resampled

