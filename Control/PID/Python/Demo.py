import math
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy.spatial import KDTree
from PID_controller import PID_posi, PID_posi_2, PID_inc

# 位置式
PID = PID_posi_2(k=[2, 0.025, 20], target=0, upper=np.pi/6, lower=-np.pi/6)
# 增量式
# PID = PID_inc(k=[2.5, 0.175, 30], target=0, upper=np.pi/6, lower=-np.pi/6)

# 车辆运动学模型
class KinematicModel:
    """假设控制量为转角 delta 和加速度 a
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

# 搜索目标临近点
def cal_target_index(state, refer_path):
    """得到临近的路点
    :param state: 当前车辆位置
    :param refer_path: 参考轨迹
    :return: 最近路点的索引
    """
    dists = []
    for xy in refer_path:
        dist = np.linalg.norm(state - xy)
        dists.append(dist)

    min_index = np.argmin(dists)
    return min_index

def main():
    # 设置参考轨迹
    refer_path = np.zeros((1000, 2))
    # 直线
    refer_path[:, 0] = np.linspace(0, 100, 1000)
    # 生成正弦轨迹
    refer_path[:, 1] = 2 * np.sin(refer_path[:, 0] / 3.0)#+ 2.5 * np.cos(refer_path[:, 0] / 2.0)
    # 使用 KDTree 搜索最临近的航路点
    refer_tree = KDTree(refer_path)

    # 假设初始状态为 x=0,y=-1,横摆角=0.5rad,前后轴距离2m,速度为2m/s,时间步长为0.1秒
    ugv = KinematicModel(0, -1, 0.5, 2, 2, 0.1)
    k = 0.1
    c = 2
    x_ = []
    y_ = []
    fig = plt.figure(1)
    # 保存动图用
    camera = Camera(fig)

    for i in range(550):
        state = np.zeros(2)
        state[0] = ugv.x
        state[1] = ugv.y
        # 方法1：在参考轨迹上查询离 state 最近的点
        dist, ind = refer_tree.query(state)
        # 方法2：使用简单的一个函数实现查询离 state 最近的点，耗时比较长

        alpha = math.atan2(refer_path[ind, 1] - state[1], refer_path[ind, 0] - state[0])
        # l_d 的一种求解方法：l_d = k * ugv.v + c
        l_d = np.linalg.norm(refer_path[ind] - state)
        theta_e = alpha - ugv.psi
        # e_y 误差的一种表示方法： e_y = -l_d * np.sign(math.sin(theta_e))
        # e_y 的另外一种表示方法：e_y = state[1] - refer_path[ind, 1]
        e_y = -l_d * math.sin(theta_e)

        delta_f = PID.cal_output(e_y)

        ugv.update_state(0, delta_f)

        x_.append(ugv.x)
        y_.append(ugv.y)

        # 显示动画
        plt.cla()
        plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
        plt.plot(x_, y_, "-r", label="trajectory")
        plt.plot(refer_path[ind, 0], refer_path[ind, 1], "go", label="target")
        plt.grid(True)
        plt.pause(0.001)
        # camera.snap()
        # animation = camera.animate()
        # animation.save("PID.gif")

    plt.figure(2)
    plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(x_, y_, 'r')
    plt.show()

if __name__=='__main__':
    main()

























