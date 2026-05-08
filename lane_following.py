import os
import sys
import time
import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

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
    #initialPosition = [1.014, -2.52, 1.131]
    #initialOrientation = [0, 0, 0]
    
    initialPosition=[-4444.245, -3703.253, 88.489]
    initialOrientation=[0, 0, np.pi/2*1.2]
    
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )
    
    # 将 QLabs 视角绑定到这辆小车上
    hqcar.possess()
    
    # 4. 启动底层物理运动模型（硬编码使用 QCAR2）
    print("Starting Real-Time Model...")
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    print("QCar2 初始化成功！[y = -2.52]")
    
    return qlabs, hqcar

def get_lane_error(rawRGB, memory_width, window_name, draw_color=(0, 255, 0)):
    if rawRGB is None or rawRGB.size == 0:
        return 0, rawRGB, memory_width
        
    height, width = rawRGB.shape[:2]
    roi_mask = np.zeros_like(rawRGB[:, :, 0])
    
    # 定义梯形多边形，覆盖下半部分道路区域
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2 + 200, height // 2 + 70),
        (width // 2 - 200, height // 2 + 70)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    
    grayImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2GRAY)
    hlsImage = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2HLS)
    
    _, gray_mask = cv2.threshold(grayImage, 220, 255, cv2.THRESH_BINARY)
    
    lower_hls = np.array([0, 210, 0])
    upper_hls = np.array([180, 255, 255])
    hls_mask = cv2.inRange(hlsImage, lower_hls, upper_hls)
    
    white_mask = cv2.bitwise_and(gray_mask, hls_mask)
    final_mask = cv2.bitwise_and(white_mask, roi_mask)
    
    scan_y = height - 150
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
    cv2.imshow(window_name, cv2.resize(display_img, (820, 410)))
    
    return error, memory_width

if __name__ == '__main__':
    import cv2
    from pal.utilities.vision import Camera2D
    from pal.products.qcar import QCar

    qlabs, hqcar = setup_env()

    imageWidth = 1640
    imageHeight = 820
    
    # 恢复前摄像头的调用逻辑
    cameraID_front = "2@tcpip://localhost:18963"
    myCam_front = Camera2D(cameraId=cameraID_front, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=60)
    
    # 使用类似的逻辑调用后摄像头
    cameraID_rear = "1@tcpip://localhost:18962" # 也可以尝试3@...18964
    myCam_rear = Camera2D(cameraId=cameraID_rear, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=60)
    
    myCar = QCar(readMode=0, frequency=60)

    Kp = 0.0006
    Ki = 0.0000
    Kd = 0.000005

    error_sum = 0
    last_error = 0
    last_time = time.time()

    lane_width_front = 800
    lane_width_rear = 800

    base_throttle = 0.10
    max_steering = np.pi / 6
    
    print("开始获取双摄像头图像并自主循迹！")
    print("注意退出方法：请【点击弹出的图像窗口】使其获得焦点，然后按 'q' 或 'ESC' 键退出；或者在当前终端按 Ctrl+C 退出。")
    
    try:
        while True:
            # 分别读取两路摄像头图像
            myCam_front.read()
            myCam_rear.read()

            rawRGB_front = myCam_front.imageData
            rawRGB_rear = myCam_rear.imageData

            error_front = 0
            error_rear = 0

            if rawRGB_front is not None and rawRGB_front.size > 0:
                error_front, lane_width_front = get_lane_error(rawRGB_front, lane_width_front, 'Front Camera', draw_color=(0, 255, 0))                          
            
            if rawRGB_rear is not None and rawRGB_rear.size > 0:
                error_rear, lane_width_rear = get_lane_error(rawRGB_rear, lane_width_rear, 'Rear Camera', draw_color=(255, 165, 0))                             
            
            # --- 融合双摄像头的控制误差 ---
            # 如果之前单纯前摄像头可以跑，说明向前的误差方向无误（右偏为正，左偏为负）。
            # 由于后置摄像头也是向后看的视角，当车头偏右（即车在虚线右侧），后方画面里的主车道中心可能也是在画面右边。
            # 这会导致 error_rear 和 error_front 符号可能相同。原先 -0.3 会强行把 error 打折甚至打歪。
            # 我们先回归到只用前摄像头作为主要引导，后摄像头作极小权重的辅助：
            error_combined = -0.75 * error_front + 0.3 * error_rear

            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.001

            error_sum += error_combined * dt
            error_sum = np.clip(error_sum, -1000, 1000)

            error_diff = (error_combined - last_error) / dt

            steering_angle = Kp * error_combined + Ki * error_sum + Kd * error_diff                                                                                         
            steering_angle = np.clip(steering_angle, -max_steering, max_steering)                                                                               
            
            myCar.write(base_throttle, steering_angle)

            last_error = error_combined
            last_time = current_time

            # cv2.waitKey 需要图像窗口处于激活/选中状态才能捕获键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27: # 27 是 ESC 键
                print("检测到退出按键，正在关闭...")
                break
    except KeyboardInterrupt:
        print("检测到 Ctrl+C 中断，正在关闭...")
        pass
    finally:
        try:
            myCar.write(0.0, 0.0)
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
