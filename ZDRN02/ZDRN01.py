import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import logging
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 matplotlib 字体，解决中文显示问题
plt.rcParams['font.family'] = 'simsun'
plt.rcParams['axes.unicode_minus'] = False

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

# 记录用户姓名、年龄和偏好
user_info = {
    "name": None,
    "age": None,
    "preferences": []
}

# 记录历史问题和答案，包含更多上下文信息
history = []

# 超参数配置
hyperparameters = {
    'lr': 0.001,
    'batch_size': 32,
    'weight_decay': 0.0001,
    'num_branches': 3,
    'num_layers': 2,
    'l1_lambda': 0.001,
    'total_iterations': 500,
    'patience': 20,
    'min_loss_improvement': 0.001
}


# 设置路径 1 的平衡目标
def set_path1_balance_targets(initial_counts, feature_dim, target_total_samples, category_params):
    num_classes = len(initial_counts)
    target_class_counts = {}
    for class_label in initial_counts.keys():
        factor = category_params.get(class_label, {}).get('path1_balance_factor', 1 + 1 / np.sqrt(feature_dim))
        initial_count = initial_counts[class_label]
        initial_proportion = initial_count / target_total_samples
        target_count = int(target_total_samples * initial_proportion * factor)
        target_class_counts[class_label] = target_count
    return target_class_counts


# 设置路径 2 的平衡目标
def set_path2_balance_targets(initial_counts, feature_dim, target_total_samples, category_params):
    num_classes = len(initial_counts)
    total_initial_samples = sum(initial_counts.values())
    target_class_counts = {}
    for class_label in initial_counts.keys():
        factor = category_params.get(class_label, {}).get('path2_balance_factor', 1 + 0.2 / np.sqrt(feature_dim))
        initial_proportion = initial_counts[class_label] / total_initial_samples
        target_count = int(target_total_samples * initial_proportion * factor)
        target_class_counts[class_label] = target_count
    return target_class_counts


# 路径 1 过采样
def path1_oversample(data_samples, num_to_add, feature_dim, class_label, category_params):
    class_samples = [sample for sample in data_samples if sample["class_label"] == class_label]
    new_samples = []
    for _ in range(num_to_add):
        # 随机选择一个同类别样本
        random_sample = random.choice(class_samples)
        new_features = []
        for i in range(feature_dim):
            valid_samples = [sample for sample in class_samples if len(sample["features"]) > i]
            if valid_samples:
                # 筛选出 features 列表长度足够的样本
                feature_values = [sample["features"][i] for sample in valid_samples]
                min_val = min(feature_values)
                max_val = max(feature_values)
                new_feature = random.uniform(min_val, max_val)
                new_features.append(new_feature)
            else:
                # 如果没有合适的样本，可根据需求处理，这里简单添加 0
                new_features.append(0)
        new_sample = {
            "features": new_features,
            "class_label": class_label,
            "info": random_sample["info"],
            "suitable_for": random_sample["suitable_for"]
        }
        new_samples.append(new_sample)
    return new_samples


# 路径 2 欠采样
def path2_undersample(data_samples, target_count, feature_dim, class_label, category_params):
    majority_samples = [sample for sample in data_samples if sample['class_label'] == class_label]
    if feature_dim == 2:
        grid_size = category_params.get(class_label, {}).get('path2_grid_size', 8)
        x_min = np.min([sample['features'][0] for sample in majority_samples])
        x_max = np.max([sample['features'][0] for sample in majority_samples])
        y_min = np.min([sample['features'][1] for sample in majority_samples])
        y_max = np.max([sample['features'][1] for sample in majority_samples])
        grid_x = np.linspace(x_min, x_max, grid_size)
        grid_y = np.linspace(y_min, y_max, grid_size)
        grid_counts = np.zeros((grid_size, grid_size))
        for sample in majority_samples:
            x_index = np.digitize(sample['features'][0], np.array(grid_x)) - 1
            y_index = np.digitize(sample['features'][1], np.array(grid_y)) - 1
            grid_counts[x_index, y_index] += 1

        sorted_indices = np.dstack(np.unravel_index(np.argsort(grid_counts.ravel()), grid_counts.shape))[0]
        remaining_samples = []
        num_removed = 0
        undersample_threshold = category_params.get(class_label, {}).get('path2_undersample_threshold', 0.8)
        for index in sorted_indices:
            x, y = index
            grid_samples = [sample for sample in majority_samples if
                            np.digitize(sample['features'][0], np.array(grid_x)) - 1 == x and
                            np.digitize(sample['features'][1], np.array(grid_y)) - 1 == y]
            if num_removed + len(grid_samples) <= len(majority_samples) - target_count and len(
                    grid_samples) > undersample_threshold * grid_size:
                remaining_samples.extend(grid_samples)
                num_removed += len(grid_samples)
            else:
                remaining_samples.extend(
                    grid_samples[:target_count - (len(majority_samples) - num_removed)])
                break
        non_majority_samples = [sample for sample in data_samples if sample['class_label'] != class_label]
        return non_majority_samples + remaining_samples
    return data_samples


# 平衡数据
def balance_data(data_samples, initial_class_counts, feature_dim, target_total_samples, category_params):
    num_classes = len(initial_class_counts)
    balanced_data_samples = data_samples.copy()
    target_class_counts_path1 = set_path1_balance_targets(initial_class_counts, feature_dim, target_total_samples,
                                                          category_params)
    target_class_counts_path2 = set_path2_balance_targets(initial_class_counts, feature_dim, target_total_samples,
                                                          category_params)

    for class_label in initial_class_counts.keys():
        if initial_class_counts[class_label] < target_total_samples / num_classes:
            max_iterations = 10
            for _ in range(max_iterations):
                current_count = collections.Counter([sample['class_label'] for sample in balanced_data_samples])[
                    class_label]
                if current_count < target_class_counts_path1[class_label]:
                    num_to_add = target_class_counts_path1[class_label] - current_count
                    new_samples = path1_oversample(balanced_data_samples, num_to_add, feature_dim, class_label,
                                                   category_params)
                    balanced_data_samples.extend(new_samples)
        else:
            max_iterations = 10
            for _ in range(max_iterations):
                current_count = collections.Counter([sample['class_label'] for sample in balanced_data_samples])[
                    class_label]
                if current_count > target_class_counts_path2[class_label]:
                    balanced_data_samples = path2_undersample(balanced_data_samples,
                                                              target_class_counts_path2[class_label], feature_dim,
                                                              class_label, category_params)

    final_class_counts = collections.Counter([sample['class_label'] for sample in balanced_data_samples])
    return balanced_data_samples, final_class_counts


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


# 自定义分层抽样器
class StratifiedSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.indices_per_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        batch_indices = []
        for _ in range(self.num_batches):
            batch = []
            for c in self.classes:
                num_samples_per_class = self.batch_size // self.num_classes
                indices = self.indices_per_class[c]
                if len(indices) > 0:
                    selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
                    batch.extend(selected_indices)
            batch_indices.extend(batch)
        return iter(batch_indices)


# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_length, in_channels = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_length, in_channels)
        if output.size(1) == 1:
            output = output.squeeze(1)
        return output


# 定义并行分支模块
class ParallelBranch(nn.Module):
    def __init__(self, input_size, hidden_size,
                 reg_type='l2', alpha=0.5, l1_ratio=0.5,
                 connection_density=0.7):
        super(ParallelBranch, self).__init__()
        self.scattered_layer = nn.Linear(input_size, hidden_size)
        self.attention = AttentionModule(hidden_size)
        self.activation = nn.SiLU()
        self.shortcut = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.scattered_layer(x)
        out = self.attention(out)
        return self.activation(out + identity)


# 定义 ZDRN 模型
class ZDRN(nn.Module):
    def __init__(self, input_size, num_branches, num_layers, output_size,
                 reg_types=('l2', 'l1'), alpha=0.5, l1_ratio=0.5, connection_density=0.7):
        super(ZDRN, self).__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for _ in range(num_layers):
            layer_branches = nn.ModuleList([
                ParallelBranch(
                    input_size=in_size,
                    hidden_size=64,
                    reg_type=reg_types[i % len(reg_types)],
                    alpha=alpha * (i + 1) / num_branches,
                    l1_ratio=l1_ratio,
                    connection_density=connection_density
                )
                for i in range(num_branches)
            ])
            self.layers.append(layer_branches)
            in_size = 64 * num_branches
        self.fc = nn.Linear(in_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            branch_outputs = []
            for branch in layer:
                branch_outputs.append(branch(x))
            x = torch.cat(branch_outputs, dim=1)
        return self.fc(x)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, scheduler, device, l1_lambda, gradient_clip=1.0):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) <= 1:
            continue
        if np.random.random() < 0.2:
            for layer in model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = np.random.uniform(0.2, 0.5)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + l1_lambda * l1_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        valid_batches += 1
    if valid_batches == 0:
        return 0
    return running_loss / valid_batches


# 评估模型
def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        f1 = f1_score(y_test.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

        y_scores = torch.softmax(outputs, dim=1).cpu().numpy()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test.cpu().numpy() == i, y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    return accuracy, f1, fpr, tpr, roc_auc


# 训练并评估模型
def train_and_evaluate(X_train, y_train, X_test, y_test, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    model = ZDRN(input_size, hyperparameters['num_branches'], hyperparameters['num_layers'], output_size).to(device)

    class_counts = np.bincount(y_train)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    total_iterations = hyperparameters['total_iterations']
    scheduler = OneCycleLR(optimizer, max_lr=hyperparameters['lr'], total_steps=total_iterations)
    plateau_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    train_losses = []
    test_accuracies = []
    test_f1_scores = []
    best_loss = float('inf')
    best_accuracy = 0
    patience = hyperparameters['patience']
    no_improvement_count = 0
    min_loss_improvement = hyperparameters['min_loss_improvement']

    iteration_count = 0

    while iteration_count < total_iterations:
        batch_size = int(hyperparameters['batch_size'])
        if batch_size < 1:
            batch_size = 1

        sampler = StratifiedSampler(y_train, batch_size)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        for inputs, labels in train_loader:
            if iteration_count >= total_iterations:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.size(0) <= 1:
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_reg = torch.tensor(0., requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            loss = loss + hyperparameters['l1_lambda'] * l1_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if iteration_count < total_iterations:
                scheduler.step()
            iteration_count += 1

        epoch_loss = train_model(model, train_loader, criterion, optimizer, None, device, hyperparameters['l1_lambda'])
        train_losses.append(epoch_loss)

        test_accuracy, test_f1, fpr, tpr, roc_auc = evaluate_model(model, X_test_tensor, y_test_tensor, device)
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)

        if iteration_count > 0:
            plateau_scheduler.step(test_accuracy)

        if test_accuracy > best_accuracy or (epoch_loss < best_loss and epoch_loss - best_loss > min_loss_improvement):
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            if epoch_loss < best_loss:
                best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            logging.info(f'Early stopping at iteration {iteration_count} due to no improvement in accuracy or loss.')
            break

        logging.info(
            f'Iteration {iteration_count}/{total_iterations}, Loss: {epoch_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}')

    try:
        model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    except RuntimeError as e:
        logging.error(f"Error loading model state_dict: {e}")
        logging.info("Using current model state instead.")
    else:
        logging.info("Successfully loaded best model state_dict.")

    final_accuracy, final_f1, fpr, tpr, roc_auc = evaluate_model(model, X_test_tensor, y_test_tensor, device)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {jewelry_classes[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi - class')
    plt.legend(loc="lower right")
    plt.show()

    return model, final_accuracy, final_f1


# 打印分类结果和评分标准
def print_classification_results(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"准确率: {accuracy:.2f}")
    print(f"精确率: {precision:.2f}")
    print(f"召回率: {recall:.2f}")
    print(f"F1 分数: {f1:.2f}")


# 生成答案
def generate_answer(question):
    start_time = time.time()
    # 这里简单模拟答案生成，实际中可根据模型预测和知识信息生成
    answer = "暂未提供具体答案，后续可根据模型预测和知识信息生成。"
    elapsed_time = time.time() - start_time
    return answer, elapsed_time


# 主函数
def main():
    # 模拟珠宝相关数据
    data_size = 30000
    diamond_texts = [
        "钻石的硬度非常高，是自然界最硬的物质之一。",
        "钻石的4C标准很重要，包括克拉、颜色、净度和切工。",
        "优质的钻石具有很高的光泽和火彩。"
    ]
    jade_texts = [
        "翡翠是一种珍贵的玉石，有多种颜色。",
        "翡翠的质地温润，佩戴起来很舒服。",
        "鉴别翡翠真假可以看光泽和内部结构。"
    ]
    pearl_texts = [
        "珍珠是由贝类分泌珍珠质形成的。",
        "珍珠的光泽柔和，有独特的美感。",
        "保养珍珠要避免接触化学物质。"
    ]
    gold_texts = [
        "黄金具有良好的延展性和导电性。",
        "黄金的纯度用K表示，24K是纯金。",
        "黄金价格受市场供需和国际形势影响。"
    ]

    data_samples = []
    base_num_per_class = data_size // num_classes
    remainder = data_size % num_classes
    for i in range(base_num_per_class):
        data_samples.append({'text': diamond_texts[i % len(diamond_texts)], 'class_label': 0})
        data_samples.append({'text': jade_texts[i % len(jade_texts)], 'class_label': 1})
        data_samples.append({'text': pearl_texts[i % len(pearl_texts)], 'class_label': 2})
        data_samples.append({'text': gold_texts[i % len(gold_texts)], 'class_label': 3})

    for i in range(remainder):
        data_samples.append({'text': diamond_texts[i % len(diamond_texts)], 'class_label': 0})

    # 数据预处理
    X, y = preprocess_data(data_samples)

    # 自定义抽样
    X_resampled, y_resampled = custom_sampling(X, y)

    # 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_accuracies = []
    all_f1_scores = []
    classification_times = []

    for train_index, test_index in kf.split(X_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        # 记录分类开始时间
        start_time = time.time()

        # 训练并评估模型
        model, accuracy, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, hyperparameters)

        # 记录分类结束时间和用时
        end_time = time.time()
        elapsed_time = end_time - start_time
        classification_times.append(elapsed_time)

        all_accuracies.append(accuracy)
        all_f1_scores.append(f1)

    # 输出每次分类用时
    for i, time_taken in enumerate(classification_times):
        print(f"第 {i + 1} 次分类用时: {time_taken:.4f} 秒")

    # 计算平均准确率和平均 F1 分数
    average_accuracy = np.mean(all_accuracies)
    average_f1 = np.mean(all_f1_scores)

    print(f"平均准确率: {average_accuracy:.4f}")
    print(f"平均 F1 分数: {average_f1:.4f}")

    return model


if __name__ == "__main__":
    main()
