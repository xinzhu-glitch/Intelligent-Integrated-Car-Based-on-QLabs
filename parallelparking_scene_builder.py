import time
import math
import numpy as np
import matplotlib.pyplot as plt
from qvl.real_time import QLabsRealTime
from qvl.spline_line import QLabsSplineLine

def setup_map(qlabs, 
              road_left_x, road_right_x, 
              road_start_y, road_end_y, 
              spot_start_y, spot_end_y, spot_right_x):
    """
    在QLabs中创建和配置仿真环境场景的代码。
    包含环境清理、背景路面绘制和车位白线的绘制功能。
    
    配置参数详细说明：
    :param qlabs: 已连接的 QuanserInteractiveLabs 实例。
    :param road_left_x: 马路左车道线端（虚线或对侧马路牙）的X坐标。
    :param road_right_x: 马路右侧线（同时是侧方位停车位的左外开口线）的X坐标。
    :param road_start_y: 主车道视野向上+Y方向伸向的远端边界。
    :param road_end_y: 主车道向下伸展的远端边界。
    :param spot_start_y: 停车位的前端Y坐标极限。
    :param spot_end_y: 停车位的底端Y坐标极限。
    :param spot_right_x: 停车池最深（右侧）的X坐标极界。
    """
    
    # 1. 环境清理：销毁之前生成的所有演员（车辆、物体等），并终止所有运行的实时模型
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5) # 给定缓冲时间让环境完全清理

    # 2. 基础底面铺设
    spline_lines = QLabsSplineLine(qlabs)
    
    # 【底面配置区】
    # bg_height: 底面的Z轴高度（默认0即贴地）
    # bg_width: 底面的宽度（模拟整管马路的大小，15.0足够宽敞）
    # bg_color: RGB颜色配置 [R, G, B]，[0, 0, 0] 表示纯黑色（柏油沥青路面）
    bg_height, bg_width, bg_color = 0.0, 15.0, [0, 0, 0]
    
    # 将此铺装视为一块宽大的画板覆盖于(0,0,0)位置
    spline_lines.spawn(location=[0, 0, bg_height], scale=[1, 1, 1], configuration=1)
    spline_lines.set_points(color=bg_color, 
                            pointList=[
                                [0, road_start_y, bg_height, bg_width], 
                                [0, road_end_y, bg_height, bg_width]
                            ], 
                            alignEndPointTangents=False)
    
    # 3. 实绩路缘线（白线）绘制
    # 【边界线配置区】
    # line_h: 线条高度（需略微高出底面bg_height以免被遮挡导致闪烁，设为0.02米）
    # line_w: 线条宽度（标准白线宽度设定，设为0.1米）
    # line_c: 白线RGB颜色设为纯白色 [1, 1, 1]
    line_h, line_w, line_c = 0.02, 0.1, [1, 1, 1]
    
    # 所有的红外、物理约束参照线：
    # 包含马路左边缘、马路右边缘（车位之外的部分）、车位的上沿、下侧和深沿
    lines = [
        # 马路左车宽线
        [[road_left_x, road_start_y, line_h, line_w], [road_left_x, road_end_y, line_h, line_w]],
        # 马路右侧线（侧方位停车之外的直路）
        [[road_right_x, road_start_y, line_h, line_w], [road_right_x, spot_start_y, line_h, line_w]],
        [[road_right_x, spot_end_y, line_h, line_w], [road_right_x, road_end_y, line_h, line_w]],
        # 侧方位停车位的三面“墙”
        [[road_right_x, spot_start_y, line_h, line_w], [spot_right_x, spot_start_y, line_h, line_w]], # 顶端线
        [[spot_right_x, spot_start_y, line_h, line_w], [spot_right_x, spot_end_y, line_h, line_w]],   # 右底线
        [[spot_right_x, spot_end_y, line_h, line_w], [road_right_x, spot_end_y, line_h, line_w]]      # 下端线
    ]
    
    # 逐一生成并渲染以上的边界实线
    for pts in lines:
        spline_lines.spawn(location=[0, 0, line_h], scale=[1, 1, 1], configuration=1)
        spline_lines.set_points(color=line_c, pointList=pts, alignEndPointTangents=False)

    # 4. === 在真实场景中绘制停车位入口处的虚白线 ===
    # 【虚线配置区】
    # dash_length: 单条虚线段的长短 (米)
    # gap_length:  虚线之间的空白间隔长度 (米)
    dash_length = 0.4
    gap_length = 0.4
    
    current_y = spot_start_y
    while current_y > spot_end_y:
        # 下一个画线的末端（最远不能超过停车位的下边沿 spot_end_y）
        next_y = max(current_y - dash_length, spot_end_y)
        
        spline_lines.spawn(location=[0, 0, line_h], scale=[1, 1, 1], configuration=1)
        spline_lines.set_points(color=line_c, pointList=[
            [road_right_x, current_y, line_h, line_w],
            [road_right_x, next_y, line_h, line_w]
        ], alignEndPointTangents=False)
        
        # 移动指针留出虚线空隙，准备绘制下一段虚线
        current_y -= (dash_length + gap_length)

def draw_static_2d_map(ax, 
                       road_left_x, road_right_x, 
                       road_start_y, road_end_y, 
                       spot_start_y, spot_end_y, spot_right_x):
    """
    基于Matplotlib绘制路面2D参考图（俯视图），展示边界与红线规则。
    参数与 setup_map 的参数保持一致。
    """
    ax.set_aspect("equal")
    # 1. 绘制马路边缘实线（黑色）
    ax.plot([road_right_x, road_right_x], [road_start_y, spot_start_y], "k-", lw=2)
    ax.plot([road_right_x, road_right_x], [spot_end_y, road_end_y], "k-", lw=2)
    ax.plot([road_right_x, spot_right_x, spot_right_x, road_right_x],
            [spot_start_y, spot_start_y, spot_end_y, spot_end_y], "k-", lw=2)
    
    # 2. 绘制防碰撞边界红线（margin边距设定0.2，与主程序保持类似观感）
    margin = 0.2
    # 左主车道防碰红线
    ax.plot([road_left_x, road_left_x], [road_start_y, road_end_y], "k-", lw=2)
    ax.plot([road_left_x + margin, road_left_x + margin], [road_start_y, road_end_y], "r-", lw=1)
    
    # 右侧马路长边红线
    ax.plot([road_right_x - margin, road_right_x - margin], [road_start_y, spot_start_y], "r-", lw=1)
    ax.plot([road_right_x - margin, road_right_x - margin], [spot_end_y, road_end_y], "r-", lw=1)
    
    # 停车池深处的封闭U型防碰红槽
    ax.plot([road_right_x - margin, spot_right_x - margin, spot_right_x - margin, road_right_x - margin],
            [spot_start_y - margin, spot_start_y - margin, spot_end_y + margin, spot_end_y + margin], "r-", lw=1)
    
    # 3. 停车位入口虚线补充绘制
    ax.plot([road_right_x, road_right_x], [spot_start_y, spot_end_y], "k--", lw=2)

    # 4. 图布视野裁剪与网格
    ax.set_xlim(road_left_x - 2.0, spot_right_x + 2.0)
    ax.set_ylim(spot_end_y - 2.0, road_start_y + 2.0)
    ax.grid(False)


if __name__ == "__main__":
    from qvl.qlabs import QuanserInteractiveLabs
    from qvl.qcar2 import QLabsQCar2
    import pal.resources.rtmodels as rtmodels
    
    print("正在连接 QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("【错误】无法连接到 QLabs！请确保 QLabs 已经启动。")
    else:
        print("连接成功！开始构建侧方位停车场景...")
        # 默认参数与主程序中配置相同
        params = {
            "road_left_x": -3.0, 
            "road_right_x": 3.0, 
            "road_start_y": 20.0, 
            "road_end_y": -25.0, 
            "spot_start_y": -5.0, 
            "spot_end_y": -11.0, 
            "spot_right_x": 6.0
        }
        
        # 1. 搭建场景地图
        setup_map(qlabs, **params)
        print("场景物理模型构建完成！")
        
        # 2. 初始并生成小车
        print("开始创建并初始化小车...")
        INIT_X = 1.5
        INIT_Y = -5.0
        INIT_YAW = math.pi / 2.0
        
        hqcar = QLabsQCar2(qlabs)
        hqcar.spawn_id(
            actorNumber=0,
            location=[INIT_X, INIT_Y, 0.0],
            rotation=[0.0, 0.0, INIT_YAW],
            waitForConfirmation=True,
        )
        hqcar.possess()
        QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
        print("小车生成完毕！(仅初始化状态，未开启控制回路)")
        
        # 3. 绘制带有状态标定预览的 2D 俯视图
        print("正在拉起2D俯视图...（关闭Matplotlib绘图窗口即可退出运行）")
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 8))
        
        # 传入统一字典重绘
        draw_static_2d_map(ax, **params)
        
        # 标注小车生成点位置与车身轮廓
        CAR_LENGTH = 3.6
        WHEELBASE = 2.4
        CAR_WIDTH = 1.8
        CAR_BODY_SCALE = 1.0
        
        line_car, = ax.plot([], [], "b-", lw=3, zorder=5, alpha=0.55)
        line_front, = ax.plot([], [], "r-", lw=2, zorder=6)
        wheel_lf, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_rf, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_lr, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_rr, = ax.plot([], [], "go", markersize=8, zorder=10)
        car_center_dot, = ax.plot([], [], "ro", markersize=6, zorder=12, label="Car Center")
        ax.legend(loc="upper right")
        
        c, s = math.cos(INIT_YAW), math.sin(INIT_YAW)
        rear_overhang = (CAR_LENGTH - WHEELBASE) / 2.0
        front_overhang = rear_overhang
        l_back = -rear_overhang * CAR_BODY_SCALE
        l_front = (WHEELBASE + front_overhang) * CAR_BODY_SCALE
        w = (CAR_WIDTH * CAR_BODY_SCALE) / 2.0
        
        # 画车框
        rect_pts = np.array([[l_front, w], [l_front, -w], [l_back, -w], [l_back, w], [l_front, w]])   
        rot_matrix = np.array([[c, -s], [s, c]])
        rotated_pts = rect_pts @ rot_matrix.T + [INIT_X, INIT_Y]
        line_car.set_data(rotated_pts[:, 0], rotated_pts[:, 1])

        # 画车头方向红线
        front_center = np.array([[l_front, 0.0]]) @ rot_matrix.T + [INIT_X, INIT_Y]   
        line_front.set_data([INIT_X, front_center[0][0]], [INIT_Y, front_center[0][1]])

        # 画四个轮子（基于后轴为(0,0)，前轴为(WHEELBASE, 0)）
        wheels_pts = np.array([
            [WHEELBASE, CAR_WIDTH / 2.0],          # 前左
            [WHEELBASE, -CAR_WIDTH / 2.0],         # 前右
            [0.0, CAR_WIDTH / 2.0],                # 后左
            [0.0, -CAR_WIDTH / 2.0],               # 后右
        ])
        rot_wheels = wheels_pts @ rot_matrix.T + [INIT_X, INIT_Y]
        wheel_lf.set_data([rot_wheels[0, 0]], [rot_wheels[0, 1]])
        wheel_rf.set_data([rot_wheels[1, 0]], [rot_wheels[1, 1]])
        wheel_lr.set_data([rot_wheels[2, 0]], [rot_wheels[2, 1]])
        wheel_rr.set_data([rot_wheels[3, 0]], [rot_wheels[3, 1]])

        # 更新车身可视中心红点
        car_geo_center = np.array([[ (l_front + l_back)/2.0, 0.0]]) @ rot_matrix.T + [INIT_X, INIT_Y]
        car_center_dot.set_data([car_geo_center[0][0]], [car_geo_center[0][1]])
        
        ax.set_title("Scene Layout Preview (Static)")
        
        # 保持显示阻塞阻断控制台返回
        plt.show(block=True)
        
        # 退出后清理断开连接
        qlabs.close()
