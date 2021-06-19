import numpy as np



class LogisticRegression(object):
    def __init__(self, dim, learning_rate=0.01, max_iter=100, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter  # 定义学习率和训练轮数
        # 可在此处补充类的属性
        self.w = np.random.normal(1, 0.1, [dim + 1, 1])

    def fit(self, X, Y):
        # 请在此处补充类的方法：训练函数，返回每轮loss的列表

        loss = []
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])  ##第一步，在矩阵X后面加了一列1
        for i in range(self.max_iter):
            self.__train_step(X1, Y)

            loss.append(self.__loss(X1, Y))
        return loss

    def __train_step(self, X, Y):  ##训练的函数类,每一次迭代更新一次
        d_w = self.__calc_gradient(X, Y)
        self.w = self.w - self.lr * d_w
        return self.w

    def predict(self, X):
        # 请在此处补充类的方法：测试函数，返回对应X的预测值和预测列表号
        X = np.hstack([X, np.ones((X.shape[0], 1))])  ##第一步，在矩阵X后面加了一列1
        Y_pred_label = []

        Y_pred = self.model(X,self.w.T)
        for y in Y_pred:
            if y >= 0.5:
                Y_pred_label.append(1)
            else:
                Y_pred_label.append(0)
        return np.array(Y_pred), np.array(Y_pred_label)

    def __loss(self, X, Y):  ##计算误差函数
        N = X.shape[0]
        w = self.w.reshape(self.w.shape[0], )
        y = Y.reshape(Y.shape[0], )
        lost = 0
        for i in range(len(X)):
            p = self.model(X[i], w.T)
            left = (-1) * y[i] * np.log(p)
            right = (y[i] - 1) * np.log(1 - p)
            lost += left + right

        return lost
    def __calc_gradient(self, X, Y):  ##求导函数类成功
        N = X.shape[0]

        w = np.array(self.w)
        P = self.model(X,w.T)
        d_w = np.dot(X.T, (P - Y))

        return d_w

    def model(self, X, w):
        return self.sigmoid(np.dot(X, w.T))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# 人工数据生成、训练测试数据划分、可视化，请勿修改本cell代码
import matplotlib.pyplot as plt


def generateData(opt='linear', num_perclass=100, seed=1):
    np.random.seed(seed)
    # 正负样本个数
    m_pos = num_perclass;
    m_neg = num_perclass;
    X = np.zeros((2, m_pos + m_neg))
    Y = np.zeros((1, m_pos + m_neg))

    # 分布类型：环形、线性函数
    if opt == 'circle':
        R1_range = 10
        R2_range = 5
        R_pos = R1_range * np.random.rand(1, m_pos)
        R_neg = R2_range * np.random.rand(1, m_neg) + 0.9 * R1_range
        Theta_pos = np.pi * np.random.randn(1, m_pos)
        Theta_neg = np.pi * np.random.randn(1, m_neg)

        X[0:1, 0:m_pos] = R_pos * np.cos(Theta_pos);
        X[1:2, 0:m_pos] = R_pos * np.sin(Theta_pos);
        Y[0, 0:m_pos] = 1;

        X[0:1, -m_neg:] = R_neg * np.cos(Theta_neg);
        X[1:2, -m_neg:] = R_neg * np.sin(Theta_neg);
        Y[0, -m_neg:] = 0;

    if opt == 'linear':
        x1 = np.random.normal(loc=-1, scale=3, size=(1, m_pos))
        X[0:1, 0:m_pos] = x1;
        # 整体线性分布
        X[1:2, 0:m_pos] = 2 * x1 + 10 + 0.1 * x1 ** 2;
        # 加噪声
        X[1:2, 0:m_pos] += np.random.normal(loc=0, scale=5, size=(1, m_pos));
        Y[0, 0:m_pos] = 1;

        x1 = np.random.normal(loc=1, scale=3, size=(1, m_neg))
        X[0:1, -m_neg:] = x1;
        X[1:2, -m_neg:] = 2 * x1 - 5 - 0.1 * x1 ** 2
        X[1:2, -m_neg:] += np.random.normal(loc=0, scale=5, size=(1, m_neg))

    return X.T, Y.T


def featureNormalize(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    X_norm = (X - mu) / sigma
    return mu, sigma, X_norm


def plotData(X, Y):
    plt.figure()
    pos_idx = (Y == 1);
    # size m,1
    pos_idx = pos_idx[:, 0];
    # size m, 这时才可用来索引某[一]个维度
    neg_idx = (Y == 0);
    neg_idx = neg_idx[:, 0];

    plt.plot(X[pos_idx, 0], X[pos_idx, 1], 'r+')
    plt.plot(X[neg_idx, 0], X[neg_idx, 1], 'bo')


def plotDecisioinBoundary(X, Y, mu, sigma, regr):
    plotData(X, Y)

    plot_num = 50;
    plot_num_2D = plot_num ** 2;

    x_plot = np.linspace(start=X[:, 0].min(), stop=X[:, 0].max(), num=plot_num)
    y_plot = np.linspace(start=X[:, 1].min(), stop=X[:, 1].max(), num=plot_num)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    X_array = np.zeros((plot_num_2D, 2))
    X_array[:, 0:1] = X_plot.reshape(plot_num_2D, 1)
    X_array[:, 1:2] = Y_plot.reshape(plot_num_2D, 1)

    X_norm = (X_array - mu) / sigma  # polyFeature(X_array,p)

    p_array, _ = regr.predict(X_norm)
    P_matrix = p_array.reshape((plot_num, plot_num))

    plt.contour(X_plot, Y_plot, P_matrix, np.array([0.5]))


def test(y_pred, y_true):
    true = 0.
    for j in range(y_pred.shape[0]):
        if y_true[j] == y_pred[j]:
            true += 1
    acc = true / y_pred.shape[0]
    return acc


# 第二组测试
dim = 2
num_perclass = 100
seed = 210404
x,y = generateData(opt='circle', num_perclass = num_perclass, seed = seed)
shuffled_index = np.random.permutation(num_perclass*2)
x = x[shuffled_index,:]
y = y[shuffled_index,:]
split_index = int(2* num_perclass * 0.7)
x_train = x[:split_index,:]
y_train = y[:split_index,:]
x_test = x[split_index:,:]
y_test = y[split_index:,:]
mu,sigma,x_train_norm = featureNormalize(x_train)
x_test_norm = (x_test-mu)/sigma

# 训练逻辑回归模型
regr = LogisticRegression(dim, learning_rate=0.01, max_iter=100, seed=seed)
loss = regr.fit(x_train_norm, y_train)
print(regr.w)
# 打印损失
plt.figure()
plt.scatter(np.arange(len(loss)), loss, marker='o', c='green')
plt.show()

# 显示训练集和测试中的分类界面
plt.figure()
plotDecisioinBoundary(x_train,y_train,mu,sigma,regr)
plt.figure()
plotDecisioinBoundary(x_test,y_test,mu,sigma,regr)


y_pred,y_pred_label = regr.predict(x_test)
acc = test(y_pred_label, y_test)
print('训练集上正确率:{}'.format(acc))