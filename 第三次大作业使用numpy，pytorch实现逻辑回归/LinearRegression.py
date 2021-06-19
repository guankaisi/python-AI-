import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, learning_rate=0.01, max_iter=100, seed=None):
        """
        一元线性回归类的构造函数：
        参数 学习率：learning_rate
        参数 最大迭代次数：max_iter
        参数 seed：产生随机数的种子
        从正态分布中采样w和b的初始值
        """
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.w = np.random.normal(1, 0.1)
        self.b = np.random.normal(1, 0.1)
        self.loss_arr = []

    def fit(self, x, y):
        """
        类的方法：训练函数
        参数 自变量：x
        参数 因变量：y
        返回每一次迭代后的损失函数
        """
        for i in range(self.max_iter):
            self.__train_step(x, y)
            y_pred = self.predict(x)
            self.loss_arr.append(self.loss(y, y_pred))

    def __f(self, x, w, b):
        '''
        类的方法：计算一元线性回归函数在x处的值
        '''
        return x * w + b

    def predict(self, x):
        '''
        类的方法：预测函数
        参数：自变量：x
        返回：对x的回归值
        '''
        y_pred = self.__f(x, self.w, self.b)
        return y_pred

    def loss(self, y_true, y_pred):
        '''
        类的方法：计算损失
        参数 真实因变量：y_true
        参数 预测因变量：y_pred
        返回：MSE损失
        '''
        return np.mean((y_true - y_pred) ** 2)

    def __calc_gradient(self, x, y):
        '''
        类的方法：分别计算对w和b的梯度
        '''
        d_w = np.mean(2 * (x * self.w + self.b - y) * x)##这里面的都是向量，可以相当于n个维度r
        d_b = np.mean(2 * (x * self.w + self.b - y))
        return d_w, d_b

    def __train_step(self, x, y):##一次迭代
        '''
        类的方法：单步迭代，即一次迭代中对梯度进行更新
        '''
        d_w, d_b = self.__calc_gradient(x, y)
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b

def show_data(x, y, w=None, b=None):
    plt.scatter(x, y, marker='.')
    if w is not None and b is not None:
        plt.plot(x, w * x + b, c='red')
    plt.show()

# data generation
np.random.seed(272)
data_size = 100
x = np.random.uniform(low=1.0, high=10.0, size=data_size)
y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)
print(x.shape,y.shape)

# train / test split
shuffled_index = np.random.permutation(data_size)
print(shuffled_index)
x = x[shuffled_index]
y = y[shuffled_index]
split_index = int(data_size * 0.7)
print(split_index)
x_train = x[:split_index]##取前七十个点
y_train = y[:split_index]
x_test = x[split_index:]##取后七十个点
y_test = y[split_index:]

# train the liner regression model
regr = LinearRegression(learning_rate=0.01, max_iter=10, seed=0)#创建对象
regr.fit(x_train, y_train)#注意接口，拟合里面只有训练集
print('w: \t{:.3}'.format(regr.w))
print('b: \t{:.3}'.format(regr.b))
show_data(x, y, regr.w, regr.b)

# plot the evolution of cost
plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
plt.show()
