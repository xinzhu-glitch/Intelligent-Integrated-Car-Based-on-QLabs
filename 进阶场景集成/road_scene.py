import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.spline_line import QLabsSplineLine


from qvl.basic_shape import QLabsBasicShape

Vector2 = Tuple[float, float]
CenterlinePoint = Tuple[float, float, float, Vector2, Vector2]


@dataclass
class RoadSceneConfig:
    """公路场景配置。"""

    lanes_per_direction: int = 3
    lane_width: float = 3.5
    y_start: float = 60.0
    flat1_length: float = 30.0
    up_length: float = 28.0
    flat2_length: float = 35.0
    slope_height: float = 5.0
    s_curve_length: float = 90.0
    s_curve_amplitude: float = 10.0
    flat3_length: float = 28.0
    down_length: float = 32.0
    flat_end_length: float = 20.0
    center_x: float = 0.0
    num_points: int = 600
    terminal_straight_length: float = 35.0
    terminal_step: float = 0.6
    terminal_marking_trim_end: float = 0.0
    line_height_offset: float = 0.02
    line_width: float = 0.15
    yellow_gap: float = 0.3
    road_color: Sequence[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    white_color: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    yellow_color: Sequence[float] = field(default_factory=lambda: [1.0, 0.8, 0.0])

    @property
    def total_width(self) -> float:
        return 2.0 * self.lanes_per_direction * self.lane_width

    def validate(self) -> None:
        if self.lanes_per_direction not in (1, 2, 3, 4):
            raise ValueError("lanes_per_direction 仅支持 1/2/3/4")
        if self.lane_width <= 0.0:
            raise ValueError("lane_width 必须大于 0")
        if self.num_points < 10:
            raise ValueError("num_points 不能过小")


@dataclass
class RoadSceneAnchor:
    """道路末端锚点，供十字路口场景拼接。"""

    x: float
    y: float
    z: float
    tx: float
    ty: float
    px: float
    py: float
    total_width: float
    lane_width: float
    lanes_per_direction: int


@dataclass
class RoadSceneResult:
    """道路场景构建结果。"""

    anchor: RoadSceneAnchor
    main_centerline: List[CenterlinePoint]
    terminal_centerline: List[CenterlinePoint]
    qlines: QLabsSplineLine


def safe_cleanup(qlabs: QuanserInteractiveLabs, timeout: float = 3.0) -> None:
    """安全清理当前场景。"""

    def cleanup_task() -> None:
        try:
            qlabs.destroy_all_spawned_actors()
            QLabsRealTime().terminate_all_real_time_models()
        except Exception as exc:
            print(f"[清理] 忽略异常: {exc}")

    cleanup_thread = threading.Thread(target=cleanup_task)
    cleanup_thread.start()
    cleanup_thread.join(timeout=timeout)
    if cleanup_thread.is_alive():
        print("[警告] 清理超时，继续执行...")
    else:
        print("[清理] 完成。")
    time.sleep(0.5)


def smootherstep(value: float) -> float:
    """五次缓动曲线，端点一阶和二阶导数均为 0。"""
    value = max(0.0, min(1.0, value))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def compute_z_profile(
    y: float,
    flat1_end_y: float,
    up_length: float,
    slope_height: float,
    flat3_end_y: float,
    down_length: float,
) -> float:
    """纵向剖面：前段缓上坡，后段缓下坡，中间保持平直。"""
    up_start_y = flat1_end_y
    up_end_y = up_start_y - up_length
    down_start_y = flat3_end_y
    down_end_y = down_start_y - down_length

    if y >= up_start_y:
        return 0.0
    if y >= up_end_y:
        progress = (up_start_y - y) / up_length
        return slope_height * smootherstep(progress)
    if y >= down_start_y:
        return slope_height
    if y >= down_end_y:
        progress = (down_start_y - y) / down_length
        return slope_height * (1.0 - smootherstep(progress))
    return 0.0


def smooth_s_curve_x(y: float, s_start_y: float, s_length: float, amplitude: float, center_x: float) -> float:
    """单次平滑变道，避免出现回头和自交。"""
    progress = (s_start_y - y) / s_length
    progress = max(0.0, min(1.0, progress))
    eased = smootherstep(progress)
    offset = amplitude * math.sin(math.pi * eased)
    return center_x + offset


def normalize2d(vx: float, vy: float) -> Vector2:
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return 0.0, -1.0
    return vx / length, vy / length


def precompute_centerline(config: RoadSceneConfig) -> List[CenterlinePoint]:
    """预计算道路中心线及其左右法向。"""
    flat1_end_y = config.y_start - config.flat1_length
    up_end_y = flat1_end_y - config.up_length
    flat2_end_y = up_end_y - config.flat2_length
    s_start_y = flat2_end_y
    s_end_y = s_start_y - config.s_curve_length
    flat3_end_y = s_end_y - config.flat3_length
    down_end_y = flat3_end_y - config.down_length
    y_end = down_end_y - config.flat_end_length

    raw_points: List[List[float]] = []
    step = (config.y_start - y_end) / config.num_points
    current_y = config.y_start
    while current_y >= y_end - 1e-6:
        current_z = compute_z_profile(
            current_y,
            flat1_end_y,
            config.up_length,
            config.slope_height,
            flat3_end_y,
            config.down_length,
        )
        if current_y < s_start_y:
            current_x = smooth_s_curve_x(
                current_y,
                s_start_y,
                config.s_curve_length,
                config.s_curve_amplitude,
                config.center_x,
            )
        else:
            current_x = config.center_x
        raw_points.append([current_x, current_y, current_z])
        current_y -= step

    centerline: List[CenterlinePoint] = []
    count = len(raw_points)
    for index in range(count):
        x0, y0, z0 = raw_points[index]
        if index == 0:
            dx = raw_points[1][0] - raw_points[0][0]
            dy = raw_points[1][1] - raw_points[0][1]
        elif index == count - 1:
            dx = raw_points[index][0] - raw_points[index - 1][0]
            dy = raw_points[index][1] - raw_points[index - 1][1]
        else:
            dx = raw_points[index + 1][0] - raw_points[index - 1][0]
            dy = raw_points[index + 1][1] - raw_points[index - 1][1]
        tx, ty = normalize2d(dx, dy)
        left_normal = (-ty, tx)
        right_normal = (ty, -tx)
        centerline.append((x0, y0, z0, left_normal, right_normal))
    return centerline


def extend_centerline_straight(centerline: Sequence[CenterlinePoint], length: float, step: float = 0.5) -> List[CenterlinePoint]:
    """从当前道路末端按切线方向延伸一段直路。"""
    if len(centerline) < 2:
        return []

    x0, y0, z0, _, _ = centerline[-1]
    x1, y1, _, _, _ = centerline[-2]
    tx, ty = normalize2d(x0 - x1, y0 - y1)
    left_normal = (-ty, tx)
    right_normal = (ty, -tx)

    step = max(step, 0.2)
    num_steps = max(2, int(length / step))
    result: List[CenterlinePoint] = []
    for index in range(1, num_steps + 1):
        distance = length * index / num_steps
        result.append((x0 + tx * distance, y0 + ty * distance, z0, left_normal, right_normal))
    return result


def gen_road_surface(centerline: Sequence[CenterlinePoint], total_width: float) -> List[List[float]]:
    return [[x, y, z, total_width] for x, y, z, _, _ in centerline]


def gen_solid_line(centerline: Sequence[CenterlinePoint], offset: float, height_offset: float, line_width: float) -> List[List[float]]:
    """生成沿道路法向贴合的实线。"""
    points: List[List[float]] = []
    for x, y, z, left_normal, right_normal in centerline:
        if offset < 0.0:
            offset_x = left_normal[0] * abs(offset)
            offset_y = left_normal[1] * abs(offset)
        else:
            offset_x = right_normal[0] * offset
            offset_y = right_normal[1] * offset
        points.append([x + offset_x, y + offset_y, z + height_offset, line_width])
    return points


def spawn_strip(qlines: QLabsSplineLine, points: Sequence[Sequence[float]], color: Sequence[float]) -> None:
    if len(points) < 2:
        return
    qlines.spawn([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1)
    qlines.set_points(color, list(points), False)


def spawn_road_colliders(qlabs: QuanserInteractiveLabs, centerline: Sequence[CenterlinePoint], total_width: float, thickness: float = 0.5, color: Sequence[float] = (0.1, 0.1, 0.1)) -> None:
    """沿中心线生成一系列长方体作为道路碰撞体实体，让小车可以上坡。"""
    if len(centerline) < 2:
        return
    
    for i in range(len(centerline) - 1):
        x0, y0, z0, _, _ = centerline[i]
        x1, y1, z1, _, _ = centerline[i + 1]
        
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        
        dist2d = math.hypot(dx, dy)
        dist3d = math.hypot(dist2d, dz)
        if dist3d < 1e-4:
            continue
            
        yaw = math.atan2(dy, dx)
        pitch = -math.atan2(dz, dist2d)
        
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        cz = (z0 + z1) / 2.0
        
        shape = QLabsBasicShape(qlabs)
        # 为了防止缝隙，将长方体长度稍微拉长一点点 (e.g. 1.01倍)
        # 必须 waitForConfirmation=True 才能获取到正确的 ID 来设置材质颜色
        shape.spawn(
            location=[cx, cy, cz - thickness / 2.0 - 0.005],
            rotation=[0.0, pitch, yaw],
            scale=[dist3d * 1.05, total_width, thickness],
            configuration=QLabsBasicShape.SHAPE_CUBE,
            waitForConfirmation=True
        )
        shape.set_material_properties(color, roughness=1.0, metallic=False, waitForConfirmation=False)


def draw_dashed_line_fixed(
    qlines: QLabsSplineLine,
    centerline: Sequence[CenterlinePoint],
    offset: float,
    height_offset: float,
    line_width: float,
    color: Sequence[float],
    dash: float = 2.0,
    gap: float = 2.0,
) -> None:
    """按弧长绘制虚线，避免末端插值时越界。"""
    if len(centerline) < 2:
        return

    arcs = [0.0]
    total_length = 0.0
    count = len(centerline)
    for index in range(1, count):
        x0, y0, _, _, _ = centerline[index - 1]
        x1, y1, _, _, _ = centerline[index]
        total_length += math.hypot(x1 - x0, y1 - y0)
        arcs.append(total_length)

    current_arc = 0.0
    segment_step = dash + gap
    while current_arc < total_length:
        dash_end = min(current_arc + dash, total_length)

        start_index = 0
        while start_index < count and arcs[start_index] < current_arc:
            start_index += 1
        start_index = min(start_index, count - 1)

        end_index = start_index
        while end_index < count and arcs[end_index] < dash_end:
            end_index += 1
        end_index = min(end_index, count - 1)

        dash_points: List[List[float]] = []
        for sample_index in range(9):
            progress = sample_index / 8.0
            target_arc = current_arc + progress * (dash_end - current_arc)

            interp_index = start_index
            while interp_index < end_index and interp_index < count - 1 and arcs[interp_index + 1] < target_arc:
                interp_index += 1
            interp_index = min(interp_index, count - 2)

            start_arc = arcs[interp_index]
            stop_arc = arcs[interp_index + 1]
            ratio = (target_arc - start_arc) / max(stop_arc - start_arc, 1e-6)

            x0, y0, z0, left_normal, right_normal = centerline[interp_index]
            x1, y1, z1, _, _ = centerline[interp_index + 1]
            x = x0 + ratio * (x1 - x0)
            y = y0 + ratio * (y1 - y0)
            z = z0 + ratio * (z1 - z0)

            if offset < 0.0:
                offset_x = left_normal[0] * abs(offset)
                offset_y = left_normal[1] * abs(offset)
            else:
                offset_x = right_normal[0] * offset
                offset_y = right_normal[1] * offset
            dash_points.append([x + offset_x, y + offset_y, z + height_offset, line_width])

        spawn_strip(qlines, dash_points, color)
        current_arc += segment_step


def trim_centerline_by_distance(
    centerline: Sequence[CenterlinePoint],
    trim_start: float = 0.0,
    trim_end: float = 0.0,
) -> List[CenterlinePoint]:
    """按距离裁剪中心线，常用于路口前后留白。"""
    if len(centerline) < 2:
        return list(centerline)

    total_length = 0.0
    for index in range(1, len(centerline)):
        x0, y0, _, _, _ = centerline[index - 1]
        x1, y1, _, _, _ = centerline[index]
        total_length += math.hypot(x1 - x0, y1 - y0)

    avg_step = max(total_length / (len(centerline) - 1), 1e-6)
    start_index = max(0, int(trim_start / avg_step))
    end_index = max(0, int(trim_end / avg_step))
    left = min(start_index, len(centerline) - 2)
    right = max(left + 2, len(centerline) - end_index)
    right = min(right, len(centerline))
    if right - left < 2:
        return []
    return list(centerline[left:right])


def draw_lane_markings(
    qlines: QLabsSplineLine,
    centerline: Sequence[CenterlinePoint],
    config: RoadSceneConfig,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    draw_edge: bool = True,
    draw_double_yellow: bool = True,
) -> None:
    """按配置绘制双向道路的边缘线、双黄线和分道虚线。"""
    segment = trim_centerline_by_distance(centerline, trim_start=trim_start, trim_end=trim_end)
    if len(segment) < 2:
        return

    total_width = config.total_width
    if draw_edge:
        spawn_strip(
            qlines,
            gen_solid_line(segment, -total_width / 2.0, config.line_height_offset, config.line_width),
            config.white_color,
        )
        spawn_strip(
            qlines,
            gen_solid_line(segment, total_width / 2.0, config.line_height_offset, config.line_width),
            config.white_color,
        )

    if draw_double_yellow:
        spawn_strip(
            qlines,
            gen_solid_line(segment, -config.yellow_gap / 2.0, config.line_height_offset, config.line_width),
            config.yellow_color,
        )
        spawn_strip(
            qlines,
            gen_solid_line(segment, config.yellow_gap / 2.0, config.line_height_offset, config.line_width),
            config.yellow_color,
        )

    for lane_index in range(1, config.lanes_per_direction):
        offset = lane_index * config.lane_width
        draw_dashed_line_fixed(
            qlines,
            segment,
            offset,
            config.line_height_offset,
            config.line_width,
            config.white_color,
        )
        draw_dashed_line_fixed(
            qlines,
            segment,
            -offset,
            config.line_height_offset,
            config.line_width,
            config.white_color,
        )


def build_road_scene(
    qlabs: QuanserInteractiveLabs,
    config: RoadSceneConfig | None = None,
    cleanup_first: bool = False,
) -> RoadSceneResult:
    """构建公路场景，不包含十字路口。"""
    config = config or RoadSceneConfig()
    config.validate()
    if cleanup_first:
        safe_cleanup(qlabs)

    qlines = QLabsSplineLine(qlabs)
    main_centerline = precompute_centerline(config)
    terminal_extension = extend_centerline_straight(
        main_centerline,
        config.terminal_straight_length,
        step=config.terminal_step,
    )
    terminal_centerline = [main_centerline[-1]] + terminal_extension

    spawn_strip(qlines, gen_road_surface(main_centerline, config.total_width), config.road_color)
    spawn_strip(qlines, gen_road_surface(terminal_centerline, config.total_width), config.road_color)
    
    # 建立具有物理碰撞厚度的实体车道
    spawn_road_colliders(qlabs, main_centerline, config.total_width, thickness=0.3, color=config.road_color)
    spawn_road_colliders(qlabs, terminal_centerline, config.total_width, thickness=0.3, color=config.road_color)
    
    draw_lane_markings(qlines, main_centerline, config)
    draw_lane_markings(qlines, terminal_centerline, config, trim_end=config.terminal_marking_trim_end)

    anchor_x, anchor_y, anchor_z, _, _ = terminal_centerline[-1]
    prev_x, prev_y, _, _, _ = terminal_centerline[-2]
    tx, ty = normalize2d(anchor_x - prev_x, anchor_y - prev_y)
    px, py = -ty, tx
    anchor = RoadSceneAnchor(
        x=anchor_x,
        y=anchor_y,
        z=anchor_z,
        tx=tx,
        ty=ty,
        px=px,
        py=py,
        total_width=config.total_width,
        lane_width=config.lane_width,
        lanes_per_direction=config.lanes_per_direction,
    )
    print(
        f"✅ 公路场景生成完成：双向每侧 {config.lanes_per_direction} 车道，"
        f"车道宽 {config.lane_width:.2f} m，总宽 {config.total_width:.2f} m"
    )
    return RoadSceneResult(
        anchor=anchor,
        main_centerline=main_centerline,
        terminal_centerline=terminal_centerline,
        qlines=qlines,
    )


def demo_build_road_scene() -> None:
    """道路库独立测试入口。"""
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("❌ QLabs 连接失败，请先打开 QLabs 软件")
        return

    try:
        config = RoadSceneConfig()
        build_road_scene(qlabs, config=config, cleanup_first=True)
        print("道路自测完成，可在 QLabs 中检查不同车道数和标线效果。")
    finally:
        qlabs.close()


if __name__ == "__main__":
    demo_build_road_scene()