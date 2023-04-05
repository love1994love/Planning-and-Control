import math
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# 车辆运动学模型
class KinematicModel:
    """
    假设控制量为转角 delta 和加速度 a
    """
    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        # 实现的是离散的模型
        self.dt = dt

    # 阿克曼转向运动学模型里程计估计
    def update_state(self, a, delta):
        self.x = self.x + self.v * math.cos(self.psi) * self.dt
        self.y = self.y + self.v * math.sin(self.psi) * self.dt
        self.psi = self.psi + self.v * math.tan(delta) / self.L * self.dt
        self.v = self.v + a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v

# 计算参考轨迹上车辆当前位置的前视目标点
def cal_target_index(state, refer_path, l_d):
    """得到前视目标点
    Args:
        state(index):当前车辆位置
        refer_path(index):参考轨迹
        l_d:前视距离
    Returns:
        index:前视目标点的索引
    """
    dists = []
    for xy in refer_path:
        dist = np.linalg.norm(state - xy)
        dists.append(dist)

        min_index = np.argmin(dists)

        delta_l = np.linalg.norm(refer_path[min_index + 1] - state)
        # 搜索前视目标点
        while l_d > delta_l and (min_index + 1) < len(refer_path):
            delta_l = np.linalg.norm(refer_path[min_index + 1] - state)
            min_index += 1
        """另外一种方法: while循环的条件检查了下一个目标点与车辆当前状态之间的距离是否小于或等于前视距离。如果是，则更新到下一个目标点并继续循环。如果不是，则退出循环并返回当前目标点的索引。
        while (min_index + 1) < len(refer_path):
            next_delta_l = np.linalg.norm(refer_path[min_index + 1] - state)
            if next_delta_l <= l_d:
                delta_l = next_delta_l
                min_index += 1
            else: break
        """
    return min_index

# Pure Pursuit 算法
def pure_pursuit_control(state, current_ref_point, l_d):
    """Pure Puisuit
    Args:
        state(index):当前车辆位置
        current_ref_point(index):当前参考路点
        l_d:前视距离
    Returns:
        delta:前轮转向角
    """
    alpha = math.atan2(current_ref_point[1] - state[1], current_ref_point[0] - state[0]) - ugv.psi
    delta = math.atan2(2 * L * np.sin(alpha), l_d)

    return delta

# 主函数
# 相关参数
L = 2           # 车辆轴距,单位:m
v = 2           # 初始速度
x_0 = 0         # 初始 x
y_0 = 0         # 初始 y
psi_0 = 0       # 初始航向角(速度低横摆角 = 航向角(车速方向与 x 轴所成角，[-pi-pi]))
dt = 0.1        # 时间间隔,单位:s
lam = 0.1       # 前视距离系数
c = 2           # 前视距离


# 设置参考轨迹
refer_path = np.zeros((1000, 2))
# 直线
refer_path[:, 0] = np.linspace(0, 100, 1000)
# 生成正弦轨迹
refer_path[:, 1] = 2 * np.sin(refer_path[:, 0] / 3.0) + 2.5 * np.cos(refer_path[:, 0] / 2.0)
# 运动学模型
ugv = KinematicModel(x_0, y_0, psi_0, v, L, dt)

x_ = []
y_ = []
fig = plt.figure(1)
# 保存动态图用
camera = Camera(fig)

def main():
    for i in range(600):
        state = np.zeros(2)
        state[0] = ugv.x
        state[1] = ugv.y

        l_d = lam * ugv.v + c
        # 搜索前视路点
        ind = cal_target_index(state, refer_path, l_d)

        delta = pure_pursuit_control(state, refer_path[ind], l_d)

        # 加速度设为 0, 恒速
        ugv.update_state(0, delta)

        print("x = ", ugv.x, ", y = ", ugv.y)

        x_.append(ugv.x)
        y_.append(ugv.y)

        # 显示动图
        plt.cla()
        plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
        plt.plot(x_, y_, "-r", label="Pure Pursuit Control")
        plt.plot(refer_path[ind,  0], refer_path[ind, 1], "go", label="target")
        plt.grid(True)
        plt.pause(0.001)

    plt.figure(2)
    plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(x_, y_, 'r')
    plt.show()

if __name__=='__main__':
    main()





