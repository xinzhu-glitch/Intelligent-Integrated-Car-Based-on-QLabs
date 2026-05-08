import os
import sys
import time
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

# --- 引入五次多项式与 Stanley 控制器辅助函数 ---
def get_quintic_polynomial_reference(x, L, W, shift_ratio=1.0):
    """
    [改进版：分段五次多项式几何路径]：
    核心原理不变，但引入了 shift_ratio（变道压缩比例）。
    系统会在 (L * shift_ratio) 的距离内就提前完成所有的侧向平移，
    剩余的距离完全用于直行回正和姿态稳定，满足"一开始转角大，提早回正拉平"的需求。
    """
    L_shift = L * shift_ratio
    
    if x <= 0:
        return 0.0, 0.0
    elif x >= L_shift:
        # 已经完成变道，后续保持在 W 的位置直行（横向位置W，航向角0）
        return W, 0.0
        
    ratio = x / L_shift
    y = W * (10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5)
    dy_dx = (W / L_shift) * (30 * ratio**2 - 60 * ratio**3 + 30 * ratio**4)
    psi = np.arctan(dy_dx)
    return y, psi

def get_stanley_steering(v, e_y, e_psi, k, max_steering):
    """
    基于 Stanley 算法的偏航角计算
    v: 速度
    e_y: 横向偏差 (即 y_real - y_ref，需注意符号)
    e_psi: 航向偏差 (即 psi_ref - psi)
    k: Stanley 增益
    """
    # 限制最小有效速度，防止除 0 与微速下舵角激增
    v_eff = max(v, 0.8) 
    cross_track_steer = np.arctan2(k * e_y, v_eff)
    steering_cmd = e_psi + cross_track_steer
    return np.clip(steering_cmd, -max_steering, max_steering)

def setup_env():
    # 1. 连接 QLabs
    os.system('cls' if os.name == 'nt' else 'clear')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs. 请确保已打开 QLabs 并加载场景。")
        sys.exit()
    print("Connected to QLabs")

    # 2. 清理历史模型，避免场景冲突
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5)

    # 3. 在指定位置生成 QCar2
    initialPosition = [1.014, -6.15, 1.131]
    initialOrientation = [0, 0, 0]
    
    #initialPosition=[-4446.456, -3708.386, 88.051]
    #initialOrientation=[0, 0, np.pi/2*1.2]
    
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )
    
    # 将 QLabs 视角绑定到这辆小车上
    hqcar.possess()
    
    # 放置交通锥（雪糕筒）作为障碍物
    try:
        from qvl.traffic_cone import QLabsTrafficCone
        cone1 = QLabsTrafficCone(qlabs)
        cone1.spawn(location=[57.495, -6.012, 1.131], scale=[3.0, 3.0, 3.0])
        
        cone2 = QLabsTrafficCone(qlabs)
        cone2.spawn(location=[171.493, -2.457, 1.131], scale=[3.0, 3.0, 3.0])
        
        cone3 = QLabsTrafficCone(qlabs)
        cone3.spawn(location=[311.659, -6.559, 1.131], scale=[3.0, 3.0, 3.0])
    except ImportError:
        pass
    
    # 4. 启动底层物理运动模型（硬编码使用 QCAR2）
    print("Starting Real-Time Model...")
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    print("QCar2 初始化成功！[y = -2.52]")
    
    return qlabs, hqcar

def check_obstacle(rawRGB):
    if rawRGB is None or rawRGB.size == 0:
        return False, 999.0
        
    height, width = rawRGB.shape[:2]
    # 为障碍物检测建立更远/更宽的 Mask 区域（扩大上方视野，提前发现障碍物！）
    roi_mask = np.zeros_like(rawRGB[:, :, 0])
    polygon = np.array([[
        (100, height - 50),
        (width - 100, height - 50),
        (width // 2 + 150, height // 2 - 150),  # 向上延伸到图像更偏上的位置
        (width // 2 - 150, height // 2 - 150)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    
    # 转换到HSV用于提取橙色/红色（雪糕筒常用色）
    hsvImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2HSV)
    lower_orange1 = np.array([0, 120, 70])
    upper_orange1 = np.array([20, 255, 255])
    lower_orange2 = np.array([160, 120, 70])
    upper_orange2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsvImage, lower_orange1, upper_orange1)
    mask2 = cv2.inRange(hsvImage, lower_orange2, upper_orange2)
    obstacle_mask = cv2.bitwise_or(mask1, mask2)
    final_mask = cv2.bitwise_and(obstacle_mask, roi_mask)
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_detected = False
    min_dist = 999.0
    
    # 画出障碍物检测区域（黄色虚线框概念）
    cv2.polylines(rawRGB, polygon, True, (0, 255, 255), 2)
    
    for cnt in contours:
        # 面积阈值降低，为了能捕捉到远处的较小红/橙色块
        if cv2.contourArea(cnt) > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            bottom_y = y + h
            # 使用识别框底端像素位置估算物理距离 (单目测距简单模型)
            # 防止除以0且修正远距离测距系数，让它能算出更符合实际的距离
            estim_dist = 600.0 / max(bottom_y - height/2 + 60, 5.0)
            if estim_dist < min_dist:
                min_dist = estim_dist
                
            # 使用红色方框标记雪糕筒
            cv2.rectangle(rawRGB, (x, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.putText(rawRGB, f"OBSTACLE: {estim_dist:.1f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            obstacle_detected = True
            
    return obstacle_detected, min_dist

def get_lane_error(rawRGB, memory_width, window_name, draw_color=(0, 255, 0), rec_l="UNKNOWN", rec_r="UNKNOWN", speed=None):
    if rawRGB is None or rawRGB.size == 0:
        return 0, memory_width, None, None, "UNKNOWN", "UNKNOWN"
        
    height, width = rawRGB.shape[:2]
    roi_mask = np.zeros_like(rawRGB[:, :, 0])
    
    # 定义梯形多边形，覆盖下半部分道路区域，适当向上延伸提升预瞄距离
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2 + 300, height // 2 + 20),
        (width // 2 - 300, height // 2 + 20)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    
    grayImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2GRAY)
    hlsImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2HLS)
    hsvImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2HSV)
    
    _, gray_mask = cv2.threshold(grayImage, 220, 255, cv2.THRESH_BINARY)
    
    lower_hls = np.array([0, 210, 0])
    upper_hls = np.array([180, 255, 255])
    hls_mask = cv2.inRange(hlsImage, lower_hls, upper_hls)
    white_mask = cv2.bitwise_and(gray_mask, hls_mask)
    
    # 增加黄色车道线检测，以防换到左侧车道时左边界是黄色实线无法被白色掩码识别
    lower_yellow = np.array([15, 80, 150])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsvImage, lower_yellow, upper_yellow)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    final_mask = cv2.bitwise_and(combined_mask, roi_mask)
    
    # 增加前视预瞄距离（原本是 height - 150），提前对弯道或车道线做反应
    scan_y = height - 200
    scan_band_half_height = 10
    mid_x = width // 2
    
    scan_band = final_mask[scan_y - scan_band_half_height : scan_y + scan_band_half_height, :]
    scan_line_1d = np.max(scan_band, axis=0) 
    
    left_edge_x = None
    right_edge_x = None
    
    for x in range(mid_x, 0, -1):
        if scan_line_1d[x] > 0:
            left_edge_x = x
            break
            
    for x in range(mid_x, width):
        if scan_line_1d[x] > 0:
            right_edge_x = x
            break
            
    display_img = rawRGB.copy()
    cv2.polylines(display_img, polygon, True, draw_color, 2)
    cv2.rectangle(display_img, (0, scan_y - scan_band_half_height), (width, scan_y + scan_band_half_height), (255, 255, 0), 2)
    
    target_center_x = mid_x
    
    if left_edge_x is not None and right_edge_x is not None:
        target_center_x = (left_edge_x + right_edge_x) // 2
        memory_width = right_edge_x - left_edge_x
        cv2.circle(display_img, (left_edge_x, scan_y), 8, (0, 0, 255), -1)
        cv2.circle(display_img, (right_edge_x, scan_y), 8, (255, 0, 0), -1)
    elif left_edge_x is not None:
        target_center_x = left_edge_x + (memory_width // 2)
        cv2.circle(display_img, (left_edge_x, scan_y), 8, (0, 0, 255), -1)
    elif right_edge_x is not None:
        target_center_x = right_edge_x - (memory_width // 2)
        cv2.circle(display_img, (right_edge_x, scan_y), 8, (255, 0, 0), -1)
    else:
        target_center_x = mid_x 
        
    error = target_center_x - mid_x
    
    cv2.circle(display_img, (target_center_x, scan_y), 10, draw_color, -1)
    cv2.line(display_img, (mid_x, height), (mid_x, height-250), (255, 0, 255), 3) 
    cv2.line(display_img, (mid_x, scan_y), (target_center_x, scan_y), (0, 165, 255), 4) 
    
    cv2.putText(display_img, f"Error: {error} px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, draw_color, 3)
    
    left_line_type = "UNKNOWN"
    right_line_type = "UNKNOWN"
    
    # 扩大垂直扫描视野以更准确涵盖断断续续的虚线
    scan_y_start = int(height / 2) + 80
    scan_y_end = height - 50
    total_rows = scan_y_end - scan_y_start
    
    def check_line_continuous(crop, total_y):
        # 将半屏区域按行压缩，如果有任何非零像素视为该行有线段
        row_act = np.max(crop, axis=1).astype(np.uint8)
        # 用形态学闭运算填补15像素以内的细微间隙（忽略噪点或小段反光缺失）
        kern = np.ones((15, 1), np.uint8)
        row_act = cv2.morphologyEx(row_act.reshape(-1, 1), cv2.MORPH_CLOSE, kern).flatten()
        
        idx = np.where(row_act > 0)[0]
        if len(idx) == 0: 
            return "UNKNOWN"
            
        span = idx[-1] - idx[0] + 1
        fill_ratio = len(idx) / span if span > 0 else 0
        
        # 真正虚线判断逻辑：
        # 1. 存在大的真实间断导致填充率低于0.85 -> 虚线
        # 2. 或者整体只是一小截，连视野垂直长度的40%都不到 -> 虚线
        # 3. 否则就是具备长跨度且未中断的大段连续线条 -> 实线
        if fill_ratio < 0.95 or span < total_y * 0.5:
            return "DASHED"
        else:
            return "SOLID"
            
    if left_edge_x is not None:
        # 使用整个平面的左半边，以应对车道转弯时的自然倾斜
        left_crop = final_mask[scan_y_start:scan_y_end, 0:mid_x]
        left_line_type = check_line_continuous(left_crop, total_rows)
        
    if right_edge_x is not None:
        right_crop = final_mask[scan_y_start:scan_y_end, mid_x:width]
        right_line_type = check_line_continuous(right_crop, total_rows)
        
    # cv2 默认不支持直接往图上画中文，所以用英文结合拼音注释显示判断结果（Solid=实线, Dashed=虚线/断线）
    str_rec_l = "DASHED(Xu)" if rec_l == "DASHED" else ("SOLID(Shi)" if rec_l == "SOLID" else "WAIT")
    str_rec_r = "DASHED(Xu)" if rec_r == "DASHED" else ("SOLID(Shi)" if rec_r == "SOLID" else "WAIT")
    str_cur_l = "DASHED(Xu)" if left_line_type == "DASHED" else ("SOLID(Shi)" if left_line_type == "SOLID" else "WAIT")
    str_cur_r = "DASHED(Xu)" if right_line_type == "DASHED" else ("SOLID(Shi)" if right_line_type == "SOLID" else "WAIT")
    
    cv2.putText(display_img, f"Status: L:{str_cur_l} | R:{str_cur_r}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(display_img, f"Memory: L:{str_rec_l} | R:{str_rec_r}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    
    if speed is not None:
        cv2.putText(display_img, f"Speed: {speed:.2f} m/s", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    cv2.imshow(window_name, cv2.resize(display_img, (820, 410)))
    
    return error, memory_width, left_edge_x, right_edge_x, left_line_type, right_line_type

if __name__ == '__main__':
    import cv2
    from pal.utilities.vision import Camera2D
    from pal.products.qcar import QCar

    qlabs, hqcar = setup_env()

    # QCar 基本参数
    wheel_base = 0.256 
    
    # 构建局部坐标下的运动推算变量 (用于换道时盲脱无参考线的情况)
    odom_x = 0.0
    odom_y = 0.0
    odom_yaw = 0.0
    
    # ==============================================================================
    # 【核心避障调参区：五次多项式规划 & Stanley跟线反馈】
    # - 所有因为场景尺度、障碍物远近而导致换道效果的问题，全部仅在此处调参 -
    # ==============================================================================
    
    # [调参项1]：横向期望抛移宽度 W (单位：米)
    # 决定换道时小车最终躲到旁边多远的偏移位。
    # -- 设大了：容易撞旁边那条道的护墙或者超出可视识别区。
    # -- 设小了：可能没从雪糕筒所在的道里完全移出来而被挂到。
    TUNING_TARGET_WIDTH_W = 0.2

    # [调参项2]：安全换道的最远缓冲纵向理论极限 L (单位：米)
    # 决定生成多项式曲线从打方向起到摆正回正用多大物理跨度。
    # -- 设短了：曲线被压缩成非常陡急的转折，导致严重偏航方向盘失控。
    # -- 设长了：多项式会慢慢舒展拉得很远才转过去，可能还没拐一半就撞上眼前的障碍。
    # (备注：代码遇到近距离障碍时，会自动将避障提前量强行缩短到不撞，这里是提供一个最大的自然规划容忍跨度)
    TUNING_PLANNING_SPAN_L = 3.0

    # [调参项3]：Stanley反馈修正纠偏增益 K 
    # 决定小手控制方向盘强制跟理论上五次多项式虚线的激烈程度脾气：
    # -- 偏向太小 (如0.3) -> 十分佛系懒散，打盘软弱无力，导致车转弯半径飘到九霄云外，跑出轨不回来。
    # -- 偏向太大 (如2.0) -> 性格神经质，稍微偏离一点虚线中心，就狂猛狠打盘，从而左右“画S蛇形龙”剧烈抖动。
    TUNING_STANLEY_GAIN_K = 1.5

    # [调参项4]：换道特有的速度控制降流系数 (占基础车速基值的惩分比率，例如 0.6)
    # Qcar轴距短，不应该油门踩死高速变轨，必须要有个适当降速过小弯的安全倍率。
    TUNING_MANEUVER_THROTTLE_RATIO = 0.7

    # [调参项5]：换道空间压缩比例 (范围：0.2 ~ 1.0)
    # 决定在总纵深 L 中，用多大比例的距离来提前完成横向变道。
    # -- 设为 1.0：原版效果，整个 L 都在慢慢变道，呈现对称S型。
    # -- 设为 0.5：要求在前 50% 的距内就猛打方向跨过去，剩下的 50% 距离纯直行。比例越小，起步转角越猛悍！
    TUNING_SHIFT_RATIO = 0.42

    # ==============================================================================
    
    # 后台传递承载变量
    local_target_y = 0.0
    local_obstacle_x = 0.0
    lane_change_L = TUNING_PLANNING_SPAN_L  
    stanley_k = TUNING_STANLEY_GAIN_K      
    
    imageWidth = 1640
    imageHeight = 820
    
    # 恢复前摄像头的调用逻辑
    cameraID_front = "2@tcpip://localhost:18963"
    myCam_front = Camera2D(cameraId=cameraID_front, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=60)
    
    # 使用类似的逻辑调用后摄像头
    cameraID_rear = "1@tcpip://localhost:18962" # 也可以尝试3@...18964
    myCam_rear = Camera2D(cameraId=cameraID_rear, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=60)
    
    myCar = QCar(readMode=1, frequency=60)
    
    Kp = 0.15     # 结合微分后，可稍微减弱比例强度
    Ki = 0.0000
    Kd = 0.08     # 引入微分阻尼，抑制振荡

    error_sum = 0
    last_error = 0
    last_time = time.time()

    lane_width_front = 800
    lane_width_rear = 800

    base_throttle = 0.10
    max_steering = np.pi / 5
    
    car_state = "NORMAL"
    lane_change_timer = 0
    correction_timer = 0
    ignore_obstacle_timer = 0
    
    recorded_left_type = "SOLID"
    recorded_right_type = "SOLID"
    
    last_encoder_count = 0
    last_encoder_time = time.time()
    current_speed = 0.0

    current_throttle = base_throttle
    last_steering_angle = 0.0
    recovery_start_time = 0
    low_error_start_time = 0
    
    print("开始获取双摄像头图像并自主循迹！")
    print("注意退出方法：请【点击弹出的图像窗口】使其获得焦点，然后按 'q' 或 'ESC' 键退出；或者在当前终端按 Ctrl+C 退出。")
    
    try:
        while True:
            # 分别读取两路摄像头图像
            myCam_front.read()
            myCam_rear.read()
            
            # 读取车辆传感器数据
            myCar.read()
            current_t = time.time()
            dt_enc = current_t - last_encoder_time
            if dt_enc > 0:
                encoder_speed = (myCar.motorEncoder[0] - last_encoder_count) / dt_enc
                # QCar速度换算公式
                raw_speed = abs((1/720/4) * ((13*19)/(70*37)) * 1 * 2*np.pi * 0.0342 * encoder_speed)
                # 简单低通滤波使显示更平滑
                current_speed = 0.9 * current_speed + 0.1 * raw_speed
            last_encoder_count = myCar.motorEncoder[0]
            last_encoder_time = current_t

            rawRGB_front = myCam_front.imageData
            rawRGB_rear = myCam_rear.imageData

            error_front = 0
            error_rear = 0
            l_edge_f, r_edge_f = None, None
            l_edge_r, r_edge_r = None, None
            l_type_f, r_type_f = "UNKNOWN", "UNKNOWN"
            l_type_r, r_type_r = "UNKNOWN", "UNKNOWN"

            current_time = time.time()
            if rawRGB_front is not None and rawRGB_front.size > 0:
                # 首先检测雪糕筒，只在正常状态下检测且忽略刚变完道的时间段
                if car_state in ["NORMAL", "RECOVERY_WAIT_5S", "RECOVERY_WAIT_ERROR", "RECOVERY_SPEED_UP"] and current_time > ignore_obstacle_timer:
                    brake_flag, obs_dist = check_obstacle(rawRGB_front)
                    if brake_flag:
                        if recorded_left_type == "DASHED" or recorded_right_type == "DASHED":
                            car_state = "STANLEY_MANEUVER"
                            lane_change_timer = current_time
                            local_obstacle_x = obs_dist
                            
                            # 【调参逻辑应用】动态计算换道纵向跨度 L
                            v_est = max(current_speed, 1.2)
                            # 留0.4米的安全纵向空隙作为硬要求，确保一定在撞上之前动作做完
                            max_available_L = max(obs_dist - 0.4, 1.5)
                            # 合并我们的基础预设限制，曲线不会无限舒展
                            lane_change_L = min(TUNING_PLANNING_SPAN_L, max_available_L)
                            
                            if recorded_left_type == "DASHED":
                                local_target_y = TUNING_TARGET_WIDTH_W
                                print(f"检测到障碍物 ({obs_dist:.1f}m)，启用 Stanley+多项式(L={lane_change_L:.1f}m) 向左平移 {local_target_y:.1f} 米...")
                            else:
                                local_target_y = -TUNING_TARGET_WIDTH_W
                                print(f"检测到障碍物 ({obs_dist:.1f}m)，启用 Stanley+多项式(L={lane_change_L:.1f}m) 向右平移 {local_target_y:.1f} 米...")
                                
                            # 同步读取调参区的设定增益
                            stanley_k = TUNING_STANLEY_GAIN_K
                            odom_x, odom_y, odom_yaw = 0.0, 0.0, 0.0
                        else:
                            car_state = "STOP"
                            print("检测到障碍物！两侧均为实线，无法变道，紧急停止！")

                # 然后再进行车道边缘提取，同时传入当前记忆的类型以便在画面显示
                error_front, lane_width_front, l_edge_f, r_edge_f, l_type_f, r_type_f = get_lane_error(
                    rawRGB_front, lane_width_front, 'Front Camera', draw_color=(0, 255, 0),
                    rec_l=recorded_left_type, rec_r=recorded_right_type, speed=current_speed
                )                          
                
                # 处于正常循迹或刚刚恢复正常循迹后，实时更新当前已知的车道虚实线类型
                # 注意：只有在车身拉直、循迹稳定(经过冷却时间)且明确检测到线段时才去更新记忆属性
                if car_state in ["NORMAL", "RECOVERY_WAIT_5S", "RECOVERY_WAIT_ERROR", "RECOVERY_SPEED_UP"] and current_time > ignore_obstacle_timer + 1.0:
                    updated = False
                    if l_type_f != "UNKNOWN" and l_type_f != recorded_left_type:
                        recorded_left_type = l_type_f
                        updated = True
                    if r_type_f != "UNKNOWN" and r_type_f != recorded_right_type:
                        recorded_right_type = r_type_f
                        updated = True
                    
                    if updated:
                        type_map = {"SOLID": "实线", "DASHED": "虚线"}
                        print(f"[{current_time:.1f}] 车道记忆已更新: 左侧为[{type_map.get(recorded_left_type)}], 右侧为[{type_map.get(recorded_right_type)}]")
            
            if rawRGB_rear is not None and rawRGB_rear.size > 0:
                error_rear, lane_width_rear, l_edge_r, r_edge_r, l_type_r, r_type_r = get_lane_error(rawRGB_rear, lane_width_rear, 'Rear Camera', draw_color=(255, 165, 0))                             
            
            # --- 融合双摄像头的控制误差与状态机决策 ---

            dt = current_time - last_time
            if dt <= 0:
                dt = 0.001

            if car_state in ["NORMAL", "RECOVERY_WAIT_5S", "RECOVERY_WAIT_ERROR", "RECOVERY_SPEED_UP"]:
                # 正常使用标准PID循迹，get_lane_error内已经利用 memory_width 较好地处理了单边线掉线的情况
                norm_front = error_front / 1000.0
                norm_rear = error_rear / 1000.0
                error_combined = -0.7 * norm_front #目前没有使用后面的误差
                error_sum += error_combined * dt
                error_sum = np.clip(error_sum, -1, 1)

                error_diff = (error_combined - last_error) / dt

                target_steering = Kp * error_combined + Ki * error_sum + Kd * error_diff                                                                                         
                target_steering = np.clip(target_steering, -max_steering, max_steering)                                                                               
                last_error = error_combined

                # 处理状态变迁与速度/转向缓变
                if car_state == "RECOVERY_WAIT_5S":
                    # 这一个阶段由于上方的跳过逻辑，目前通常会被闲置不进，也可以作为备用保留
                    current_throttle = base_throttle * 0.4
                    if current_time - recovery_start_time > 1.0: # 压缩硬等时间
                        car_state = "RECOVERY_WAIT_ERROR"
                        low_error_start_time = current_time
                        print("结束盲目硬等，开始监测误差...")
                elif car_state == "RECOVERY_WAIT_ERROR":
                    current_throttle = base_throttle * 0.6
                    # 判断前后误差是否都较小 
                    # 刚变过去的时候，因为没有了原有车道记忆，所以双目一开始一定在找新道线。
                    if abs(error_front) < 100 and abs(error_rear) < 100: 
                        if current_time - low_error_start_time > 1.5: # 从 5 秒压缩到 2 秒，能认出中间就行
                            car_state = "RECOVERY_SPEED_UP"
                            print("误差对齐，开始缓慢提速...")
                    else:
                        low_error_start_time = current_time # 重置时间
                elif car_state == "RECOVERY_SPEED_UP":
                    # 每秒增加0.02的throttle
                    current_throttle += 0.5 * dt
                    if current_throttle >= base_throttle:
                        current_throttle = base_throttle
                        car_state = "NORMAL"
                        print("已恢复至正常速度。")
                else: # NORMAL
                    current_throttle = base_throttle

                # 增加转向缓变逻辑，使其平滑过渡（低通滤波思想）
                # 这个过程对刚退出换道那一瞬间有很好的平滑效果
                alpha = 3.0 * dt # 渐变速率
                alpha = min(alpha, 1.0)
                smooth_steering = last_steering_angle + alpha * (target_steering - last_steering_angle)
                
                myCar.write(current_throttle, smooth_steering)
                last_steering_angle = smooth_steering

            elif car_state == "STANLEY_MANEUVER":
                # 五次多项式几何规划 + Stanley 反馈避障换道
                x_pos, y_pos, yaw = odom_x, odom_y, odom_yaw
                
                # 1. 使用当前车纵向坐标 x_pos 查询这瞬间我们在理想曲线上的预期位置 y_ref 和 预期航向角 psi_ref
                y_ref, psi_ref = get_quintic_polynomial_reference(x_pos, lane_change_L, local_target_y, TUNING_SHIFT_RATIO)
                
                # 2. 计算横向追踪误差 e_y （注意方向系符号，让正负抵消方向正确）
                e_y = y_ref - y_pos
                # 3. 计算航向误差 e_psi
                e_psi = psi_ref - yaw
                
                # 4. Stanley 动态控制器输出方向角
                steering_cmd = get_stanley_steering(current_speed, e_y, e_psi, stanley_k, max_steering)
                
                # 5. 【调参应用】换道固定应用设降速倍率系数
                throttle_cmd = base_throttle * TUNING_MANEUVER_THROTTLE_RATIO
                current_throttle = throttle_cmd
                
                # 增加转向小幅度平滑，防止由于瞬间 e_y 计算带来的毛刺
                alpha = 8.0 * dt
                alpha = min(alpha, 1.0)
                smooth_steering = last_steering_angle + alpha * (steering_cmd - last_steering_angle)
                
                # 发送给车辆
                myCar.write(current_throttle, smooth_steering)
                last_steering_angle = smooth_steering
                
                # 更新推算位姿供下一帧使用
                odom_x += dt * current_speed * np.cos(odom_yaw)
                odom_y += dt * current_speed * np.sin(odom_yaw)
                odom_yaw += dt * (current_speed / wheel_base) * np.tan(smooth_steering)
                odom_yaw = np.arctan2(np.sin(odom_yaw), np.cos(odom_yaw)) # 角度归一化
                
                # 退出条件检测：过障碍物 或者（横向接近目标位 并且车身摆正）
                y_err = abs(y_pos - local_target_y)
                yaw_err = abs(yaw)
                
                # 【修改逻辑】：只要换道姿态收拢（且进度超过变道压缩点），立刻退出盲算，切回双目视觉跟线！
                # 不再死板地等 x_pos 跨过障碍物甚至跑完 L。只要在隔壁道平了，就用视觉抓隔壁道的线。
                is_lane_changed = x_pos > (lane_change_L * TUNING_SHIFT_RATIO) and y_err < 0.15 and yaw_err < 0.15
                
                if is_lane_changed or (x_pos > local_obstacle_x + 0.3):
                    car_state = "RECOVERY_WAIT_ERROR" # 直接跳过硬等 3 秒的 5S 阶段，直接进入误差对线！
                    low_error_start_time = current_time
                    ignore_obstacle_timer = current_time + 6.0 # 长时间屏蔽障碍物检测，防止视觉切到隔壁道时误识别
                    
                    # 【防微分爆炸】由于切回视觉跟线的第一帧会产生新的误差计算，如果保留进入前的last_error，其产生的巨大error_diff会引起剧烈转向
                    last_error = -0.7 * (error_front / 1000.0)
                    
                    print(f"动态避障结束：当前坐标 (x={x_pos:.2f}, y={y_pos:.2f})，期望目标 (y_ref={y_ref:.2f})。变道已平稳，立刻切回视觉跟线！")
                
                error_sum = 0
                
            elif car_state == "STOP":
                last_steering_angle = 0.0
                # 两侧都为实线，无法变道，直接停止
                myCar.write(0.0, 0.0)
                error_sum = 0

            last_time = current_time

            # cv2.waitKey 需要图像窗口处于激活/选中状态才能捕获键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27: # 27 是 ESC 键
                print("检测到退出按键，正在紧急停车...")
                myCar.write(0.0, 0.0) # 速度置0
                time.sleep(0.1) # 停顿确保指令发送下去
                break
    except KeyboardInterrupt:
        print("检测到 Ctrl+C 中断，正在紧急停车...")
        myCar.write(0.0, 0.0) # 速度置0
        time.sleep(0.1)
        pass
    finally:
        try:
            print("清理环境：车速强制置 0.0")
            myCar.write(0.0, 0.0) # 终极速度置0
            time.sleep(0.2)
            myCar.terminate()
        except:
            pass

        try:
            myCam_front.terminate()
        except:
            pass
        
        try:
            myCam_rear.terminate()
        except:
            pass

        cv2.destroyAllWindows()
        QLabsRealTime().terminate_all_real_time_models()
        print("已清理后台模型。")
