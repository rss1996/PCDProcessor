# 获取邻居索引
if self.k_nn is not None:
    # 使用 k-NN 聚合
    knn_indices = self.get_knn_indices(x, self.k_nn)  # 获取最近的 k 个邻居
    # 将 k-NN 的邻居特征进行聚合
    knn_features = []
    for i in range(batch_size):
        # 获取当前点的所有邻居索引
        indices = knn_indices[i]
        # 获取邻居的特征
        neighbors = tf.gather(x[i], indices, axis=0)  # 通过索引获取邻居的点坐标
        # 对邻居进行平均池化（或者最大池化等其他方法）
        knn_feature = tf.reduce_mean(neighbors, axis=0, keepdims=True)
        knn_features.append(knn_feature)

    knn_features = tf.concat(knn_features, axis=0)  # 聚合 batch 中的所有点
else:
    # 使用球形邻域聚合
    ball_indices = self.get_ball_query(x, self.radius, self.nsample)  # 获取邻域内的点
    # 将球形邻域的邻居特征进行聚合
    ball_features = []
    for i in range(batch_size):
        # 获取当前点的所有邻居索引
        indices = ball_indices[i]
        # 获取邻居的特征
        neighbors = tf.gather(x[i], indices, axis=0)  # 通过索引获取邻居的点坐标
        # 对邻居进行平均池化（或者最大池化等其他方法）
        ball_feature = tf.reduce_mean(neighbors, axis=0, keepdims=True)
        ball_features.append(ball_feature)

    ball_features = tf.concat(ball_features, axis=0)  # 聚合 batch 中的所有点

