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

# 相关参数设置
k = 0.2       # 增益系数
dt = 0.1      # 时间间隔，单位：s
L = 0.2         # 车辆轴距，单位：m
v = 2         # 初始速度
x_0 = 0       # 初始 x
y_0 = -3      # 初始 y
psi_0 = 0     # 初始航向角(车速方向与 x 轴所成角，[-pi-pi])

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

# 角度归一化
def normalize_angle(angle):
    """Normalize an angle to [-pi, pi], for atan2 return angle in [-pi, pi]
    :param angle: (float)
    :return: (float) angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

# Stanley 算法实现
def stanley_control(state, refer_path, refer_path_psi):
    """Stanley 控制
    :param state: 位姿,包括 x, y, yaw, v
    :param refer_path: 参考轨迹的位置
    :param refer_path_psi: 参考轨迹上点的切向方向的角度
    :return:
    """
    current_target_index = cal_target_index(state[0:2], refer_path)

    # 当计算出来的目标临近点索引大于等于参考轨迹上的最后一个点索引时
    if current_target_index >=  len(refer_path):
        current_target_index = len(refer_path) - 1
        current_ref_point = refer_path[-1]
        psi_t = refer_path_psi[-1]
    else:
        current_ref_point = refer_path[current_target_index]
        psi_t = refer_path_psi[current_target_index]

    # 计算横向误差 e_y = (-sin \theta, cos \theta) ((x - x_min), (y - y_min)), e_y 为正, 在轨迹前进方向的左边, 往右打方向盘, 为负, 在轨迹前进方向的右边, 往左打方向盘
    # if (state[1] - current_ref_point[1]) * math.cos(psi_t) - (state[0] - current_ref_point[0]) * math.sin(psi_t) <= 0:
    #     e_y = -((state[1] - current_ref_point[1]) * math.cos(psi_t) - (state[0] - current_ref_point[0]) * math.sin(psi_t))
    # else:
    #     e_y = ((state[1] - current_ref_point[1]) * math.cos(psi_t) - (state[0] - current_ref_point[0]) * math.sin(psi_t))

    if (state[1] - current_ref_point[1]) * math.cos(psi_t) - (state[0] - current_ref_point[0]) * math.sin(psi_t) <= 0:
        e_y = np.linalg.norm(state[0:2] - current_ref_point)
    else:
        e_y = -np.linalg.norm(state[0:2] - current_ref_point)

    # 计算转角
    psi = state[2]
    v = state[3]
    #  psi_t 的另外一种计算方式: math.atan2(current_ref_point[1] - state[1], current_ref_point[0] - state[0])
    theta_e = psi_t - psi
    delta_e = math.atan2(k * e_y, v)
    delta = normalize_angle(theta_e + delta_e)
    return delta, current_target_index

# 主函数
def main():
    # set reference trajectory
    refer_path = np.zeros((1000, 2))
    # 直线
    refer_path[:, 0] = np.linspace(0, 100, 1000)
    # 生成正弦轨迹
    refer_path[:, 1] = 2 * np.sin(refer_path[:, 0] / 3.0) + 2.5 * np.cos(refer_path[:, 0] / 2.0)
    # 参考轨迹上点的切线方向的角度,近似计算
    refer_path_psi = [math.atan2(refer_path[i + 1, 1] - refer_path[i, 1], refer_path[i + 1, 0] - refer_path[i, 0])
                      for i  in range(len(refer_path) - 1)]
    # 运动学模型
    ugv = KinematicModel(x_0, y_0, psi_0, v, L, dt)
    goal = refer_path[-2]

    x_ = []
    y_ = []
    fig = plt.figure(1)
    # 保存动图用
    camera = Camera(fig)

    for _ in range(500):
        state = np.zeros(4)
        state[0] = ugv.x
        state[1] = ugv.y
        state[2] = ugv.psi
        state[3] = ugv.v

        delta, ind = stanley_control(state, refer_path, refer_path_psi)

        # 加速度设为0，恒速
        ugv.update_state(0, delta)

        x_.append(ugv.x)
        y_.append(ugv.y)

        # 显示动图
        plt.cla()
        plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
        plt.plot(x_, y_, "-r", label="Stanley")
        plt.plot(refer_path[ind, 0], refer_path[ind, 1], "go", label="target")
        plt.grid(True)
        plt.pause(0.001)
        camera.snap()
        if ind >= len(refer_path_psi) - 1:
            break
    animation = camera.animate()
    animation.save('Stanley.gif')
    plt.figure(2)
    plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(x_, y_, 'r')
    plt.show()

if __name__ == '__main__':
    main()
