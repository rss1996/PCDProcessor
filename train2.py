import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
threshold = 0.9
class PointCloudDataProcessor:
    def __init__(self, data):
        """
        初始化函数，传入模拟数据
        data: pd.DataFrame，包含x, z, ID, label, region列
        """
        self.data = data
    
    def filter_by_label(self, label_dict):
        """
        根据标签筛选信号或噪声
        label: 0 (噪声) 或 1 (信号)
        """
        data = self.data
        for k, v in label_dict.items():
            data = data[data[k]==v]
            data = data.reset_index(drop=True)
        return data
    
    def plot_distribution(self, label_dict={}):
        """
        可视化点云数据分布
        label: 如果指定，过滤指定标签的信号或噪声；否则显示所有数据
        """
        if label_dict !={}:
            data_to_plot = self.filter_by_label(label_dict)
        else:
            data_to_plot = self.data
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_to_plot['x'], data_to_plot['z'], c=data_to_plot['label'], cmap='viridis', s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Point Cloud Distribution'+str(label_dict)+"count:"+str(len(data_to_plot)))
        plt.show()
        
    def plot_2d_distribution(self, label_dict={}):
        """
        绘制2D分布图：横轴为根号下(x^2 + y^2)，纵轴为z
        """
        # 计算欧几里得距离 (x^2 + y^2)^(1/2)
        if label_dict !={}:
            data_to_plot = self.filter_by_label(label_dict)
        else:
            data_to_plot = self.data
        # data_to_plot['distance'] = np.sqrt(data_to_plot['x']**2 + data_to_plot['y']**2)
        
        # 绘制2D散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(data_to_plot['x'], data_to_plot['z'], c=data_to_plot['label'], cmap='viridis', s=10)
        plt.xlabel('Euclidean Distance (x)')
        plt.ylabel('Z Coordinate')
        plt.title('2D Distribution: Distance vs Z')
        plt.colorbar(label='Label (0=Noise, 1=Signal)')
        plt.show()

    def plot_histogram(self, bins=50):
        """
        绘制x和z坐标的直方图
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        axs[0].hist(self.data['x'], bins=bins, color='blue', alpha=0.7)
        axs[0].set_title('X Coordinate Histogram')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Frequency')
        
        axs[1].hist(self.data['z'], bins=bins, color='green', alpha=0.7)
        axs[1].set_title('Z Coordinate Histogram')
        axs[1].set_xlabel('Z')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def summary_statistics(self):
        """
        显示数据的基本统计信息
        """
        return self.data.describe()

    def load_data_from_folder(self, base_path):
        """
        从指定的根目录动态加载数据，识别所有主文件夹、子文件夹及文件
        base_path: str，根目录路径
        """
        all_data = []
        
        # 遍历根目录下的所有主文件夹
        for main_folder in os.listdir(base_path):
            main_folder_path = os.path.join(base_path, main_folder)
            if not os.path.isdir(main_folder_path):  # 如果不是文件夹，跳过
                continue

            # 判断主文件夹中是否直接包含txt文件
            has_txt_files = any(file_name.endswith('.txt') for file_name in os.listdir(main_folder_path))
            if has_txt_files:
                # 如果直接包含txt文件，将其当作一个区域
                region_label = main_folder
                for file_name in os.listdir(main_folder_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(main_folder_path, file_name)
                        data = self.read_data(file_path, region_label)
                        all_data.append(data)
            else:
                # 遍历主文件夹中的子文件夹（区域）
                for sub_folder in os.listdir(main_folder_path):
                    sub_folder_path = os.path.join(main_folder_path, sub_folder)
                    if not os.path.isdir(sub_folder_path):  # 如果不是文件夹，跳过
                        continue
                    
                    # 动态生成区域标签，例如 "MainFolder-SubFolder"
                    region_label = f"{main_folder}-{sub_folder}"
                    
                    # 遍历子文件夹中的txt文件
                    for file_name in os.listdir(sub_folder_path):
                        if file_name.endswith('.txt'):
                            file_path = os.path.join(sub_folder_path, file_name)
                            data = self.read_data(file_path, region_label)
                            all_data.append(data)
        
        # 将所有数据合并为一个DataFrame
        self.data = pd.concat(all_data, ignore_index=True)
    
    def read_data(self, file_path, region_label):
        """
        从txt文件中读取数据并返回DataFrame，并加入区域标签
        file_path: str，txt文件路径
        region_label: str，区域标签，例如 "Forest-area1" 或 "ICE-region1"
        """
        data = np.loadtxt(file_path, delimiter=',', dtype={'names': ('x', 'z', 'ID', 'label'),
                                                            'formats': ('f4', 'f4', 'i4', 'i4')})
        # 将区域标签添加为新列
        df = pd.DataFrame(data)
        df['region'] = region_label
        return df

# 示例使用
# 设置数据文件夹路径
base_path = './real ICESat-2 data'  # 根目录路径，包含Forest和ICE文件夹

# 创建数据处理器
processor = PointCloudDataProcessor(pd.DataFrame())

# 加载数据
processor.load_data_from_folder(base_path)
data = processor.data
df = data[data["region"]=="night"].reset_index(drop=True)
print(len(df))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers


class SetAbstraction(layers.Layer):
    def __init__(self, n_points, radius=None, nsample=None, mlps=None, k_nn=None, use_xyz=True):
        super(SetAbstraction, self).__init__()
        self.n_points = n_points
        self.radius = radius
        self.nsample = nsample
        self.mlps = mlps
        self.k_nn = k_nn
        self.use_xyz = use_xyz
    
        
    def build(self, input_shape):
        # 创建MLP层
        self.mlp = []
        for dim in self.mlps:
            self.mlp.append(layers.Conv2D(dim, 1, activation='relu'))
        super(SetAbstraction, self).build(input_shape)
        
            
    def get_knn_indices(self, x, k):
        """
        计算k近邻的索引（简化版，基于欧氏距离计算）
        point_cloud: 输入的点云数据，形状为 (batch_size, num_points, 2)
        k: 需要返回的邻居数量
        """
        dist = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(x, 2) - tf.expand_dims(x, 1)), axis=-1))
        return tf.argsort(dist, axis=-1, direction='ASCENDING')[:, :, :k]


    def get_ball_query(self, point_cloud, radius, nsample):
        """
        获取球形邻域内的点的索引
        point_cloud: 输入的点云数据，形状为 (batch_size, num_points, 2)
        radius: 邻域半径
        nsample: 每个点采样的最大点数
        """
        batch_size, num_points, _ = point_cloud.shape
        # 计算每对点之间的欧氏距离
        dists = tf.norm(point_cloud[:, :, None, :] - point_cloud[:, None, :, :], axis=-1)  # (batch_size, num_points, num_points)
        
        # 获取每个点在给定半径内的邻居
        ball_query_indices = tf.where(dists <= radius)  # (num_valid_pairs, 3), 3是(batch_idx, point_idx_1, point_idx_2)
        
        # 为每个点分配邻居
        # 我们用 tf.scatter_nd 来分配邻居
        point_indices = tf.unique(ball_query_indices[:, 1])[0]  # 获取所有的点索引
        ball_indices = []
        
        for i in range(batch_size):
            # 获取当前批次所有点的邻居索引
            batch_mask = ball_query_indices[:, 0] == i
            filtered_neighbors = ball_query_indices[batch_mask]
            
            # 获取每个点的邻居
            point_indices_list = []
            for j in range(num_points):
                neighbors_of_point_j = filtered_neighbors[filtered_neighbors[:, 1] == j][:, 2]  # 获取点j的邻居
                
                num_neighbors = tf.shape(neighbors_of_point_j)[0]
                
                # 如果邻居数量少于nsample，随机填充
                if num_neighbors < nsample:
                    additional_indices = tf.random.shuffle(neighbors_of_point_j)
                    neighbors_of_point_j = tf.tile(additional_indices, [nsample // num_neighbors + 1])[:nsample]
                elif num_neighbors > nsample:
                    neighbors_of_point_j = neighbors_of_point_j[:nsample]
                    
                point_indices_list.append(neighbors_of_point_j)
            
            # 堆叠所有点的邻居索引
            ball_indices.append(tf.stack(point_indices_list, axis=0))  # shape: (num_points, nsample)
        
        # 转换为(batch_size, num_points, nsample)的形状
        ball_indices = tf.stack(ball_indices, axis=0)  # shape: (batch_size, num_points, nsample)
        return ball_indices
            

    def call(self, inputs):
        if isinstance(inputs, (tuple, list)):
            points, global_features = inputs
        else:
            points = inputs
            global_features = None

        batch_size = tf.shape(points)[0]
        num_points = tf.shape(points)[1]
        num_features = tf.shape(points)[2]

        # 获取邻居索引
        if self.k_nn is not None:
            # 使用 k-NN 聚合
            knn_indices = self.get_knn_indices(points, self.k_nn)  # 获取最近的 k 个邻居
            # 创建一个TensorArray来存储所有的邻居特征
            # 使用邻居索引来提取邻居特征
            neighbors = tf.gather(points, knn_indices, batch_dims=1)  # 形状: (batch_size, num_points, k, num_features)

        else:
            # 使用球形邻域聚合
            ball_indices = self.get_ball_query(points, self.radius, self.nsample)  # 获取邻域内的点
            # 创建一个TensorArray来存储所有的邻居特征
            # 获取批次索引、源点索引和目标点索引
            # 将球形邻域的邻居特征进行聚合
            neighbors = tf.gather(points, ball_indices, batch_dims=1)  # 形状: (batch_size, num_points, k, num_features)

            
        # 将原始的点云特征和经过聚合的邻域特征一起输入到 MLP 层
        # 对邻域特征进行MLP处理
        feature = neighbors
        for mlp_layer in self.mlp:
            feature = mlp_layer(feature)

        # 对每个点的邻域特征进行最大池化
        feature = tf.reduce_mean(feature, axis=2)

        if global_features is not None:
            feature = tf.concat([feature, global_features], axis=-1)

        return feature


class FeaturePropagation(layers.Layer):
    def __init__(self, mlp_dims=[64, 64], **kwargs):
        super(FeaturePropagation, self).__init__(**kwargs)
        self.mlp_dims = mlp_dims

    def build(self, input_shape):
        # 创建MLP层
        self.mlp = []
        for dim in self.mlp_dims:
            self.mlp.append(layers.Dense(dim, activation='relu'))
        super(FeaturePropagation, self).build(input_shape)

    def call(self, inputs):
        # 输入是 (local_features, global_features)
        local_features, global_features = inputs

        # 对每个点的特征进行MLP处理
        feature = local_features
        for mlp_layer in self.mlp:
            feature = mlp_layer(feature)

        # 将local_features和global_features拼接
        feature = tf.concat([feature, global_features], axis=-1)

        return feature

class PointNetPlusPlus(tf.keras.Model):
    def __init__(self, num_points=100, num_classes=2, **kwargs):
        super(PointNetPlusPlus, self).__init__(**kwargs)
        # SetAbstraction layers
        # self.sa1 = SetAbstraction(num_neighbors=num_neighbors, mlp_dims=[64, 128])
        # self.sa2 = SetAbstraction(num_neighbors=num_neighbors, mlp_dims=[128, 256])
        # self.sa3 = SetAbstraction(num_neighbors=num_neighbors, mlp_dims=[256, 512])
        self.sa1 = SetAbstraction(n_points=num_points, k_nn=32, mlps=[64, 64])  # 第一层使用k-NN
        self.sa2 = SetAbstraction(n_points=num_points, radius=0.2, nsample=32, mlps=[128, 128])  # 第二层使用球形邻域
        self.sa3 = SetAbstraction(n_points=num_points, radius=0.4, nsample=64, mlps=[256, 256])   # 第三层使用球形邻域


        # Feature Propagation layers
        self.fp1 = FeaturePropagation(mlp_dims=[512, 512])
        self.fp2 = FeaturePropagation(mlp_dims=[256, 256])

        # Fully connected layers
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        # 最后分类层，每个点的输出是 1 个二分类结果（噪声或信号）
        self.output_layer = layers.Dense(1, activation='sigmoid')  # 输出一个二分类值
        
    def call(self, inputs):
        """
        :param inputs: 输入点云数据，形状为 (batch_size, num_points, num_features)
        :return: 分类结果
        """
        x = inputs
        # Set Abstraction 层
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)

        # Feature Propagation 层
        x = self.fp1([x2, x3])
        x = self.fp2([x1, x])

        # 全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        
        # 最后一层分类
        x = self.output_layer(x)  # 输出每个点的标签（0或1）
        
        return x
    
    def train_step(self, data):
        """
        自定义训练步骤，接收数据并计算损失与梯度
        """
        # 获取数据
        inputs, labels = data
        with tf.GradientTape() as tape:
            # 前向传播
            predictions = self(inputs, training=True)
            # 计算损失
            loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        
        # 计算梯度并更新模型参数
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 计算正样本的精确率和召回率
        y_pred = tf.cast(predictions > threshold, tf.float32)  # 将得分大于 0.1 的预测为正样本（1），否则为负样本（0）
        tp = tf.reduce_sum(labels * y_pred)  # 真阳性
        fp = tf.reduce_sum((1 - labels) * y_pred)  # 假阳性
        fn = tf.reduce_sum(labels * (1 - y_pred))  # 假阴性
        labels_p = tf.reduce_sum(labels)
        count = labels.shape[0]*labels.shape[1]
        
        precision = tp / (tp + fp + tf.keras.backend.epsilon())  # 加入 epsilon 防止除零错误
        recall = tp / (tp + fn + tf.keras.backend.epsilon())  # 加入 epsilon 防止除零错误

        accuracy = tf.keras.metrics.binary_accuracy(labels, y_pred)
        accuracy = tf.reduce_mean(accuracy)
        
        # 返回 loss 和 metrics
        return {"loss": loss,"accuracy":accuracy, "precision": precision, "recall": recall,
                "tp":tp,"fp":fp,"labels_p":labels_p,"count":count}



    def test_step(self, data):
        """
        自定义测试步骤，用于评估模型
        """
        inputs, labels = data
        predictions = self(inputs, training=False)
        loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        # 计算正样本的精确率和召回率
        y_pred = tf.cast(predictions > threshold, tf.float32)  # 将得分大于 0.1 的预测为正样本（1），否则为负样本（0）
 
        tp = tf.reduce_sum(labels * y_pred)  # 真阳性
        fp = tf.reduce_sum((1 - labels) * y_pred)  # 假阳性
        fn = tf.reduce_sum(labels * (1 - y_pred))  # 假阴性
        labels_p = tf.reduce_sum(labels)
        
        precision = tp / (tp + fp + tf.keras.backend.epsilon())  # 加入 epsilon 防止除零错误
        recall = tp / (tp + fn + tf.keras.backend.epsilon())  # 加入 epsilon 防止除零错误
        accuracy = tf.keras.metrics.binary_accuracy(labels, y_pred)
        accuracy = tf.reduce_mean(accuracy)
        # 返回 loss 和 metrics
        return {"loss": loss,"accuracy":accuracy, "precision": precision, "recall": recall,
                "tp":tp,"fp":fp,"labels_p":labels_p}


    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, callbacks=None):
        """
        自定义 fit 函数，每个 epoch 打印训练过程中的损失和准确率，并支持验证集与回调函数
        """
        # 转换为 tf.data.Dataset 类型
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        
        # 处理验证集
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        else:
            val_dataset = None
        
        # 初始化回调函数
        if callbacks is None:
            callbacks = []
        
        # 迭代每个 epoch
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = 0
            train_accuracy = 0
            train_precision = 0
            train_recall = 0
            train_tp = 0
            train_fp = 0
            train_labels_p = 0
            num_batches = 0
            
            # 训练步骤
            for step, (inputs, labels) in enumerate(train_dataset):
                # 执行一个训练步骤
                metrics = self.train_step((inputs, labels))
                train_loss += metrics["loss"]
                train_accuracy += metrics["accuracy"]
                train_precision += metrics["precision"]
                train_recall += metrics["recall"]
                train_tp += metrics["tp"]
                train_fp += metrics["fp"]
                train_labels_p += metrics["labels_p"]
                num_batches += 1
                
                # 每 10 个 batch 打印一次当前进度
                if step % 1 == 0:
                    print(f"Step {step}/{len(train_dataset)} - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, precision: {metrics['precision']:.4f},recall: {metrics['recall']:.4f},labels_p: {metrics['labels_p']},tp: {metrics['tp']},fp: {metrics['fp']},count: {metrics['count']}")

            # 计算每个 epoch 的平均训练损失和准确度
            train_loss /= num_batches
            train_accuracy /= num_batches
            train_precision /= num_batches
            train_recall /= num_batches
            train_tp /= num_batches
            train_fp = num_batches
            train_labels_p = num_batches
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f},labels_p: {train_labels_p},tp: {train_tp},fp: {train_fp}")

            # 验证阶段
            if val_dataset is not None:
                val_loss = 0
                val_accuracy = 0
                val_precision = 0
                val_recall = 0
                val_tp = 0
                val_fp = 0
                val_labels_p = 0
                num_batches = 0
                for step, (inputs, labels) in enumerate(val_dataset):
                    metrics = self.test_step((inputs, labels))
                    val_loss += metrics["loss"]
                    val_accuracy += metrics["accuracy"]
                    val_precision += metrics["precision"]
                    val_recall += metrics["recall"]
                    val_tp += metrics["tp"]
                    val_fp += metrics["fp"]
                    val_labels_p += metrics["labels_p"]
                    num_batches += 1
                val_loss /= num_batches
                val_accuracy /= num_batches
                val_precision /= num_batches
                val_recall /= num_batches
                val_tp /= num_batches
                val_fp = num_batches
                val_labels_p = num_batches
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f},labels_p: {val_labels_p},tp: {val_tp},fp: {val_fp}")
            else:
                print("Validation data not provided.")
            # model.save('/save/'+model_name+"_epoch"+str(epoch), save_format='tf')
            # 调用回调函数
            # for callback in callbacks:
            #     callback.on_epoch_end(epoch, logs={"loss": train_loss, "accuracy": train_accuracy, "val_loss": val_loss if val_dataset is not None else None, "val_accuracy": val_accuracy if val_dataset is not None else None})
            
            print("-" * 50)
            
            

from tensorflow.keras.callbacks import TensorBoard
# 创建一个 TensorBoard 回调，指定日志目录
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_rate, step_size):
        # 确保 initial_lr 和 decay_rate 是 float 类型
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.step_size = tf.cast(step_size, tf.int32)

    def __call__(self, step):
        # step 应该是一个 TensorFlow 张量，因此直接使用 TensorFlow 的运算
        step = tf.cast(step, tf.int32)  # 确保 step 是 int32 类型
        return self.initial_lr * tf.pow(self.decay_rate, tf.cast(step // self.step_size, tf.float32))


# 创建自定义学习率调度器
lr_schedule = LearningRateScheduler(initial_lr=0.01, decay_rate=0.9, step_size=1000)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metrics.AUC()])

# 自定义加权二元交叉熵损失函数
def weighted_binary_crossentropy(y_true, y_pred, weight_pos=1, weight_neg=100):
    # 计算二元交叉熵损失
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce(y_true, y_pred)
    
    # 根据标签调整损失的权重
    weights = weight_pos * y_true + weight_neg * (1 - y_true)
    weighted_loss = loss * weights
    
    return tf.reduce_mean(weighted_loss)


# 假设你的点云数据每个样本有512个点，每个点有2个特征（x, z）
num_points = 256  # 每个点云包含512个点
num_features = 2   # 每个点有2个特征（x, z）
# 确保数据的总量符合要求
num_samples = len(df) // (num_points * num_features)
df = df.iloc[:num_samples * num_points * num_features]

# 创建模型实例
model = PointNetPlusPlus(num_points = num_points)
# 为模型指定输入形状，假设每个点云有1024个点，每个点有2个特征（x, z）
@tf.function
def build_model(input_data):
    return model(input_data)

# 创建虚拟数据并通过装饰的函数运行
dummy_input = np.zeros((1, num_points, num_features), dtype=np.float32)

# 调用模型，这时模型会被构建并优化为静态图
build_model(dummy_input)
# 编译模型
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy'])
# 打印模型概况
model.summary()


# 特征和标签
X = df[['x', 'z']].values  # 特征：x, z
y = df['label'].values     # 标签：0 (噪声), 1 (非噪声)
y = y.astype(np.float32)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据调整为正确的形状
X_reshaped = X_scaled.reshape(-1, num_points, num_features)
print(X_reshaped.shape)  # 输出调整后的数据形状

# 确保标签与特征的样本数量一致
y = y.reshape(-1, num_points, 1)  # 确保标签形状为 (batch_size, num_points, 1)
print(y.shape)


# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42,shuffle=False)

# 输出划分后的训练集和测试集的形状
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")
model_name = "night_train2"
# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))