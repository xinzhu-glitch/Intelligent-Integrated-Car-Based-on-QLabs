import time
import math
import numpy as np
import matplotlib.pyplot as plt

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.real_time import QLabsRealTime
from qvl.spline_line import QLabsSplineLine
from pal.products.qcar import QCar
import pal.resources.rtmodels as rtmodels

# ==========================================
# 1. 基础参数与核心调优区
# ==========================================
# >>>>> 【手动调参核心区】：您可以直接在此处修改各停车阶段的阈值 <<<<<

# 【判定参数1】向右插头阶段的收手角度（决定第一把方向倒多深）
# 物理意义：偏离初始YAW（即90度）的角度值，越小（减去的度数大，如45度）代表车尾深深扎入车位；太大时容易扎不进顺带擦外马路牙子。
STATE1_ADJUST_DEGREE = 35.0

# 【判定参数2】后退到底时的断线判定 X 坐标增量（决定直退倒多深）
# 物理意义：当左后轮跨过马路右侧线（ROAD_RIGHT_X）多少距离时开始切向内揉盘。想要彻底停进红线，需要大幅增加此值（例如+1.2）。
STATE2_LR_WHEEL_OFFSET_X = -0.60

# 【判定参数3】向内摆平车身的顺位截点角偏差（决定左打死何时结束）
# 物理意义：适当放大提前余量，提前结束倒车阶段（缩短倒车距离），剩下的部分交给第四阶段前进时“边前行边摆正”。
STATE3_YAW_ERROR_DEGREE = 19

# 【物理硬碰撞侦测参数】：这是用于防死机的墙避碰
# 防撞预留边界距离（米）。如果想允许轻微切线擦边可以设为0.1~0.2，设大则非常保守，发现快撞上就会立刻终止当前倒车强转入前进。
COLLISION_MARGIN = 0.05

# 【力度与速度参数指令】（速度全局缩放至原本的0.3，遏制因为惯性导致的阶段性冲刺过猛和最后越线）
S_REVERSE_THROTTLE = -0.015
S_FORWARD_THROTTLE = 0.012


# --- 车辆物理尺寸参数 ---
WHEELBASE = 2.4             # 车辆轴距（前后轮中心距），直接作为阿克曼转向几何和后轮位置的依据
CAR_WIDTH = 1.8             # 车辆横向宽度，加上镜子后的物理极限，用于计算外围红框压线
CAR_LENGTH = 3.6            # 车辆纵向最远距离极限，主要防碰撞
CAR_BODY_SCALE = 1.0        # 检测盒收缩安全系数（微缩外盒能避免模拟器中的极小冗余导致触发刚体碰撞而卡死）

# --- 道路坐标系参数（全局绝对坐标系） ---
# 坐标说明：X轴(+)为场地右侧；Y轴(+)为马路正前方。车辆绝对端正时 YAW 应恰好等于 pi/2(90°)
ROAD_LEFT_X = -3.0          # 马路左车道线端（虚线或对侧马路牙）
ROAD_RIGHT_X = 3.0          # 马路右侧线（同时是侧方位停车位的左外开口线）
ROAD_START_Y = 20.0         # 主车道视野向上+Y方向伸向的远端边界
ROAD_END_Y = -25.0          # 主车道向下伸展倒车的远端边界

# --- 侧方位停车位几何构建 ---
SPOT_SCALE = 1.2            # 停车位难度控制系数（当前设为放大1.5倍长宽的简化难度版）
SPOT_START_Y = -5.0         # 停车位的Y轴上边缘（靠近车头的车位前底线）
SPOT_BASE_WIDTH = 2.5      # 1.0倍原尺寸下车位底横向深度
SPOT_BASE_DEPTH = 5.0      # 1.0倍原尺寸下车位底纵向长度
SPOT_WIDTH = SPOT_BASE_WIDTH * SPOT_SCALE # 放大后真实判定用的车位横向开深
SPOT_DEPTH = SPOT_BASE_DEPTH * SPOT_SCALE # 放大后真实判定的车位南北向长度
SPOT_END_Y = SPOT_START_Y - SPOT_DEPTH    # 停车位的Y轴后边缘（后方红线界线）
SPOT_RIGHT_X = ROAD_RIGHT_X + SPOT_WIDTH  # 停车池最最右侧死胡同（右方红线界线）

# --- 小车初始位置 ---
INIT_X = 1.5                # 起步中心X轴：1.5位于右侧车道中间，代表平稳靠右行驶
INIT_Y = -5.0               # 起步中心Y轴：和 SPOT_START_Y 平齐，模拟刚好把车开到前车位平行
INIT_YAW = math.pi / 2.0    # 起步姿态角：精准平齐pi/2

# --- 底层硬件控制限制 ---
CONTROL_DT = 0.02           # 步进时长，0.02s对应50Hz真实硬件循环
MAX_STEER = math.radians(40.0) # 仿真器中前轮机械允许打死最大极限（转为弧度约 0.697）
REVERSE_THROTTLE = 0.03     # 倒车起步固定给的油门强度绝对值

# ==========================================
# 2. 场景映射
# ==========================================
def setup_map(qlabs):
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5)

    spline_lines = QLabsSplineLine(qlabs)
    bg_height, bg_width, bg_color = 0.0, 15.0, [0, 0, 0]
    
    spline_lines.spawn(location=[0, 0, bg_height], scale=[1, 1, 1], configuration=1)
    spline_lines.set_points(color=bg_color, pointList=[[0, ROAD_START_Y, bg_height, bg_width], [0, ROAD_END_Y, bg_height, bg_width]], alignEndPointTangents=False)
    
    line_h, line_w, line_c = 0.02, 0.1, [1, 1, 1]
    
    lines = [
        [[ROAD_LEFT_X, ROAD_START_Y, line_h, line_w], [ROAD_LEFT_X, ROAD_END_Y, line_h, line_w]],
        [[ROAD_RIGHT_X, ROAD_START_Y, line_h, line_w], [ROAD_RIGHT_X, SPOT_START_Y, line_h, line_w]],
        [[ROAD_RIGHT_X, SPOT_END_Y, line_h, line_w], [ROAD_RIGHT_X, ROAD_END_Y, line_h, line_w]],
        [[ROAD_RIGHT_X, SPOT_START_Y, line_h, line_w], [SPOT_RIGHT_X, SPOT_START_Y, line_h, line_w]],
        [[SPOT_RIGHT_X, SPOT_START_Y, line_h, line_w], [SPOT_RIGHT_X, SPOT_END_Y, line_h, line_w]],
        [[SPOT_RIGHT_X, SPOT_END_Y, line_h, line_w], [ROAD_RIGHT_X, SPOT_END_Y, line_h, line_w]]
    ]
    for pts in lines:
        spline_lines.spawn(location=[0, 0, line_h], scale=[1, 1, 1], configuration=1)
        spline_lines.set_points(color=line_c, pointList=pts, alignEndPointTangents=False)

    # === 在真实场景中绘制停车位入口处的虚白线 ===
    dash_length = 0.4   # 虚线线段长度（米）
    gap_length = 0.4    # 虚线间隙长度（米）
    current_y = SPOT_START_Y
    while current_y > SPOT_END_Y:
        next_y = max(current_y - dash_length, SPOT_END_Y)
        spline_lines.spawn(location=[0, 0, line_h], scale=[1, 1, 1], configuration=1)
        spline_lines.set_points(color=line_c, pointList=[
            [ROAD_RIGHT_X, current_y, line_h, line_w],
            [ROAD_RIGHT_X, next_y, line_h, line_w]
        ], alignEndPointTangents=False)
        current_y -= (dash_length + gap_length)


def draw_static_2d_map(ax):
    ax.set_aspect("equal")
    ax.plot([ROAD_RIGHT_X, ROAD_RIGHT_X], [ROAD_START_Y, SPOT_START_Y], "k-", lw=2)
    ax.plot([ROAD_RIGHT_X, ROAD_RIGHT_X], [SPOT_END_Y, ROAD_END_Y], "k-", lw=2)
    ax.plot([ROAD_RIGHT_X, SPOT_RIGHT_X, SPOT_RIGHT_X, ROAD_RIGHT_X],
            [SPOT_START_Y, SPOT_START_Y, SPOT_END_Y, SPOT_END_Y], "k-", lw=2)
    
    margin = 0.2
    ax.plot([ROAD_LEFT_X, ROAD_LEFT_X], [ROAD_START_Y, ROAD_END_Y], "k-", lw=2)
    ax.plot([ROAD_LEFT_X + margin, ROAD_LEFT_X + margin], [ROAD_START_Y, ROAD_END_Y], "r-", lw=1)
    ax.plot([ROAD_RIGHT_X - margin, ROAD_RIGHT_X - margin], [ROAD_START_Y, SPOT_START_Y], "r-", lw=1)
    ax.plot([ROAD_RIGHT_X - margin, ROAD_RIGHT_X - margin], [SPOT_END_Y, ROAD_END_Y], "r-", lw=1)
    ax.plot([ROAD_RIGHT_X - margin, SPOT_RIGHT_X - margin, SPOT_RIGHT_X - margin, ROAD_RIGHT_X - margin],
            [SPOT_START_Y - margin, SPOT_START_Y - margin, SPOT_END_Y + margin, SPOT_END_Y + margin], "r-", lw=1)
    ax.plot([ROAD_RIGHT_X, ROAD_RIGHT_X], [SPOT_START_Y, SPOT_END_Y], "k--", lw=2)

    ax.set_xlim(ROAD_LEFT_X - 2.0, SPOT_RIGHT_X + 2.0)
    ax.set_ylim(SPOT_END_Y - 2.0, ROAD_START_Y + 2.0)
    ax.grid(False)


# ==========================================
# 3. 实体车控制
# ==========================================
class QCarStateMachine:
    """
    实体/仿真车基于状态机的【四步侧方位极简开环停靠器】。
    无需解算复杂的函数轨道，通过采集当下物理角与位移与【特定阈值】对比即可推进停车过程。
    """
    def __init__(self, hqcar):
        self.hqcar = hqcar
        self.qcar_hw = QCar(readMode=1, frequency=100) # 生成交互层硬件控制柄
        
        # 内部动态物理镜像
        self.x = INIT_X
        self.y = INIT_Y
        self.yaw = INIT_YAW

        self.state = 1          # 维护四步流程当前走到了第几阶段
        self.history_x = []     # 维护画图用的全生命周期车辆质心横纵向记录
        self.history_y = []

    def update_state(self):
        """实时捕获真实/Qlabs环境中的数据注入给判断器"""
        success, loc, rot, _ = self.hqcar.get_world_transform()
        if success:
             self.x = loc[0]   # 绝对位置 X
             self.y = loc[1]   # 绝对位置 Y
             self.yaw = rot[2] # 绝对旋转角 yaw (rad)
             self.history_x.append(self.x)
             self.history_y.append(self.y)
        self.qcar_hw.read()    # 更新通讯底座

    def drive(self, throttle, steering):
        """封装调用写入，在此方法向底层发送速度和方向指令"""
        self.qcar_hw.write(throttle, steering)

    def compute_left_rear_wheel_x(self):
        """
        利用车宽、方位角进行二维平移变换，推算出【左后轮】的绝对X轴像素位置 。
        由于原点位于后轴中心，因此纵向相对坐标 lr_dx 为 0.0。
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lr_dx, lr_dy = 0.0, CAR_WIDTH / 2.0
        return self.x + (lr_dx * c - lr_dy * s)

    def compute_car_corners(self):
        """
        全量推导整车四向尖角坐标的高精盒子计算。
        真实情况下小车的原点 (sm.x, sm.y) 位于【后轮轴中心】。
        因此车框的几何中心在原点前方 WHEELBASE/2 处。
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        # 车身后悬与前悬 (假设车长均匀分布在轴距前后)
        rear_overhang = (CAR_LENGTH - WHEELBASE) / 2.0
        front_overhang = rear_overhang
        # 加上缩放安全系数
        l_back = -rear_overhang * CAR_BODY_SCALE
        l_front = (WHEELBASE + front_overhang) * CAR_BODY_SCALE
        w = (CAR_WIDTH * CAR_BODY_SCALE) / 2.0
        
        # 排列为：[右上(主驾前), 右下(后备箱右), 左下(后备箱左), 左上(副驾前)]
        # X向前，Y向左
        corners_dx_dy = [(l_front, -w), (l_back, -w), (l_back, w), (l_front, w)]
        corners = []
        for dx, dy in corners_dx_dy:
            cx = self.x + (dx * c - dy * s)
            cy = self.y + (dx * s + dy * c)
            corners.append((cx, cy))
        return corners

    def step(self):
        # ================= [动作判定与防碰撞阈值取用] =================
        # 根据开头预设的参数进行实时解析转化
        # 注意物理航向：倒车时向右打方向，车尾向右(+X)，车头其实是向左边偏(-X)，相对+Y的夹角(Yaw)是【增加】的！
        STATE1_TARGET_YAW = INIT_YAW + math.radians(STATE1_ADJUST_DEGREE) 
        STATE2_TARGET_LR_X = ROAD_RIGHT_X + STATE2_LR_WHEEL_OFFSET_X 
        # 同样，在第三把向左边揉盘摆正时，车头向右回偏，Yaw会【减小】回到 pi/2 附近
        STATE3_TARGET_YAW = INIT_YAW + math.radians(STATE3_YAW_ERROR_DEGREE) 
        
        # 使用动态传入的碰撞边界参数定义物理框
        LIMIT_RIGHT = SPOT_RIGHT_X - COLLISION_MARGIN # 最右胡同界。大即向右触壁。        
        LIMIT_BACK = SPOT_END_Y + COLLISION_MARGIN    # 最底端后界。小即菊花触壁。        
        LIMIT_FRONT = SPOT_START_Y - COLLISION_MARGIN # 最靠前门沿线，大即前冲失控触别车。
        # ==============================================================

        throttle = 0.0
        steering = 0.0
        
        # 使用上面的精算结果进行【动态高频撞壁感应】，如果有任何一角碰壁，布尔值告警触发：
        corners = self.compute_car_corners()
        # 由于Y正轴向前，碰后界相当于车角的y轴分量掉过界：cy < LIMIT_BACK；右界则是cx冲出 LIMIT_RIGHT。
        is_hitting_back_or_right = any(cy < LIMIT_BACK for cx, cy in corners) or any(cx > LIMIT_RIGHT for cx, cy in corners)
        is_hitting_front = any(cy > LIMIT_FRONT for cx, cy in corners)

        # >>>>>> 基于几何的五阶段自动泊车机执行块 <<<<<<
        if self.state == 1:
            # 【阶段1：右倒打死】——让车尖以最大的急弯进入右侧区域。
            steering = -MAX_STEER                # Qcar内负值表向右打摆满前轴
            throttle = S_REVERSE_THROTTLE        # 切后退挡位
            if self.yaw >= STATE1_TARGET_YAW:    # 修正：Yaw因为后轮往右走而增加到了阈值
                self.state = 2                   # 完成姿态1，流动到维持后退
            elif is_hitting_back_or_right:
                print("【警告】阶段1倒车压线！提前提车防撞！")
                self.state = 4

        elif self.state == 2:
            # 【阶段2：端正推移】——不打乱上环节创造的斜度角，正推将整车揉塞进停车位。
            steering = 0.0                       # 洗掉所有偏置角，完全笔直滑行
            throttle = S_REVERSE_THROTTLE        # 切后退挡位不歇
            lr_x = self.compute_left_rear_wheel_x() # 监控左身突兀度
            if lr_x > STATE2_TARGET_LR_X:        # 当发觉车左侧屁股已经探路越进了红框白线 
                self.state = 3                   # 立刻着手第三把防甩阶段 
            elif is_hitting_back_or_right:
                print("【警告】阶段2推太深压线！提早抢救前进转正！")
                self.state = 4
        elif self.state == 3:
            # 【阶段3：左倒打死】——往里摆正车头。
            steering = MAX_STEER                 # 往左使劲反打将车头顺进车位
            throttle = S_REVERSE_THROTTLE        # 依然向后挤进
            
            # 【防线紧急制转机制】：为了防止你上面参数太野导致车屁股已经撞烂后墙红线还没转过身来！
            if is_hitting_back_or_right:
                 # 若被发现已碰后侧沿壁
                print("【警告-碰边干预】距墙近到压红线！停止往后强塞，提前切入补救状态4！")
                self.state = 4                   # 提早进入前挡揉盘模式
            
            # 正常的达成路径：车辆角已经被端正为理想朝向阈值了
            elif self.yaw <= STATE3_TARGET_YAW:
                print("【提示-姿态完成】车身顺直！进入收尾纠错补正。")
                self.state = 4

        elif self.state == 4:
            # 【阶段4：前行精准微调】——负责扫平一切剩余误差，向着【pi/2和车位中心】抹平修图。
            yaw_error = INIT_YAW - self.yaw      # 测量与最完美标准纯直角的残差(正差朝右,负差偏左)

            # 加入一个极为简易有效的 【P 比例修正器】：对偏角误差施放大系数的左/右追赶舵力。
            # 这里偏置倍数为 2.5 ，方向盘会跟上角差迅速矫捷。并由clip限制不损坏物理方向轴极界。
            steering = np.clip(yaw_error * 4.5, -MAX_STEER, MAX_STEER)
            
            throttle = S_FORWARD_THROTTLE        # 改切前大档（此时必须往前窜去争取把车头回偏拉平的空间）

            # 触发关停的检查A（圆满成功）：不仅朝向角度纯顺，为了减短运动距离，只要角度顺直且整车完全进入了底线即可停息，不再非得跑到死板的正中心！
            is_perfect_straight = abs(yaw_error) < math.radians(1.0) # 再次适度放宽直顺判断（1度以内即可），由于速度更慢，在此刻很容易停在正中心
            # 不再强制必须到达中心才停车，大幅缩短第4阶段无意义的长距离直行
            if is_perfect_straight:
                print("【提示-完美泊平】朝向直顺，无需强制拖拽至中心点，快速泊车成功。")
                self.state = 5                   # 胜利切至停息关核
                
            # 越线绝对惩罚：如果在前进微调时蹭到了右侧红线，给予满级向左打盘惩罚强行拽离墙壁
            if any(cx > LIMIT_RIGHT for cx, cy in corners):
                steering = MAX_STEER # 强制往左打死，剥离墙壁
                
            # 触发关停的检查B（防前碰）：如果你往前顺直走的途中，由于空间过窄即将怼着停在前车位的人了（超出了起始端线）。
            if is_hitting_front:
                print("【警告-逼近碰前】已经触及最顶侧前方红线禁区极值，自动拉手刹强迫停止。")
                self.state = 5                   # 被迫切至停息关核

        elif self.state == 5:
            # 【阶段5：彻底关闭息位】
            throttle = 0.0                       # 撤掉一切动能
            steering = 0.0                       # 方向复位为空
            
        return throttle, steering

    def shutdown(self):
        try:
            self.drive(0.0, 0.0)
            self.qcar_hw.terminate()
        except:
            pass


def main():
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("无法连接到 QLabs！")
        return

    setup_map(qlabs)

    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=[INIT_X, INIT_Y, 0.0],
        rotation=[0.0, 0.0, INIT_YAW],
        waitForConfirmation=True,
    )
    hqcar.possess()
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)

    sm = QCarStateMachine(hqcar)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))
    draw_static_2d_map(ax)

    line_car, = ax.plot([], [], "b-", lw=3, zorder=5, alpha=0.55)
    line_front, = ax.plot([], [], "r-", lw=2, zorder=6)
    track_line, = ax.plot([], [], "m-", lw=2, zorder=4, label="Trajectory")
    
    wheel_lf, = ax.plot([], [], "go", markersize=8, zorder=10)
    wheel_rf, = ax.plot([], [], "go", markersize=8, zorder=10)
    wheel_lr, = ax.plot([], [], "go", markersize=8, zorder=10)
    wheel_rr, = ax.plot([], [], "go", markersize=8, zorder=10)
    car_center_dot, = ax.plot([], [], "ro", markersize=6, zorder=12, label="Car Center")
    ax.legend(loc="upper right")

    running = [True]

    def on_key(event):
        if event.key == "escape":
            sm.drive(0.0, 0.0)
            running[0] = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    last_control_time = time.time()

    try:
        while running[0]:
            sm.update_state()

            now = time.time()
            if now - last_control_time >= CONTROL_DT:
                throttle, steering = sm.step()
                sm.drive(throttle, steering)
                last_control_time = now

            c, s = math.cos(sm.yaw), math.sin(sm.yaw)
            rear_overhang = (CAR_LENGTH - WHEELBASE) / 2.0
            front_overhang = rear_overhang
            l_back = -rear_overhang * CAR_BODY_SCALE
            l_front = (WHEELBASE + front_overhang) * CAR_BODY_SCALE
            w = (CAR_WIDTH * CAR_BODY_SCALE) / 2.0
            
            # 画车框
            rect_pts = np.array([[l_front, w], [l_front, -w], [l_back, -w], [l_back, w], [l_front, w]])   
            rot_matrix = np.array([[c, -s], [s, c]])
            rotated_pts = rect_pts @ rot_matrix.T + [sm.x, sm.y]
            line_car.set_data(rotated_pts[:, 0], rotated_pts[:, 1])

            # 画车头方向红线
            front_center = np.array([[l_front, 0.0]]) @ rot_matrix.T + [sm.x, sm.y]   
            line_front.set_data([sm.x, front_center[0][0]], [sm.y, front_center[0][1]])

            # 画四个轮子（基于后轴为(0,0)，前轴为(WHEELBASE, 0)）
            wheels_pts = np.array([
                [WHEELBASE, CAR_WIDTH / 2.0],          # 前左
                [WHEELBASE, -CAR_WIDTH / 2.0],         # 前右
                [0.0, CAR_WIDTH / 2.0],                # 后左
                [0.0, -CAR_WIDTH / 2.0],               # 后右
            ])
            rot_wheels = wheels_pts @ rot_matrix.T + [sm.x, sm.y]
            wheel_lf.set_data([rot_wheels[0, 0]], [rot_wheels[0, 1]])
            wheel_rf.set_data([rot_wheels[1, 0]], [rot_wheels[1, 1]])
            wheel_lr.set_data([rot_wheels[2, 0]], [rot_wheels[2, 1]])
            wheel_rr.set_data([rot_wheels[3, 0]], [rot_wheels[3, 1]])

# 更新轨迹线（基于后轴追踪）
            if len(sm.history_x) > 0:
                track_line.set_data(sm.history_x, sm.history_y)

            # 更新车身可视中心红点（让用户的视觉感受不再割裂）
            car_geo_center = np.array([[ (l_front + l_back)/2.0, 0.0]]) @ rot_matrix.T + [sm.x, sm.y]
            car_center_dot.set_data([car_geo_center[0][0]], [car_geo_center[0][1]])

            ax.set_xlim(sm.x - 8, sm.x + 8)
            ax.set_ylim(sm.y - 12, sm.y + 12)
            ax.set_title(f"State Machine Parking | State: {sm.state}")

            fig.canvas.draw()
            fig.canvas.flush_events()

            if sm.state == 5:
                if not hasattr(sm, 'parked_printed'):
                    print(">>> 停车完成，抵达阶段5 (已保持地图界面开启，您可以按ESC键退出)")
                    sm.parked_printed = True
                sm.drive(0.0, 0.0)
                # 不执行 break，保证 plt 的画布事件循环一直在更新，不断刷新保持画面

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        sm.shutdown()
        qlabs.close()

if __name__ == "__main__":
    main()
