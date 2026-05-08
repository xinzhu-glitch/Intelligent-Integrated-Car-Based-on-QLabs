import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from qvl.basic_shape import QLabsBasicShape
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.spline_line import QLabsSplineLine


Vector2 = Tuple[float, float]
CenterlinePoint = Tuple[float, float, float, Vector2, Vector2]

STATE_RED = 1
STATE_GREEN = 2
STATE_YELLOW = 3
DEFAULT_CYCLE_SEQUENCE = (STATE_RED, STATE_YELLOW, STATE_GREEN, STATE_YELLOW)


@dataclass
class TrafficLightPlacement:
    """单个方向红绿灯的相对放置修正。"""

    lateral_sign: float
    longitudinal_shift: float = 0.0
    lateral_shift: float = 0.0
    yaw_offset_deg: float = 0.0


@dataclass
class TrafficLightStyle:
    """红绿灯几何风格配置。"""

    scale: float = 0.65
    pole_height: float = 6.0
    pole_width: float = 0.16
    light_size: float = 0.52
    far_side_margin: float = 2.3
    pole_lateral_margin: float = 1.4
    horizontal_span_ratio: float = 0.8


@dataclass
class TrafficLightCycleConfig:
    """红绿灯循环时序配置。"""

    sequence: Sequence[int] = field(default_factory=lambda: list(DEFAULT_CYCLE_SEQUENCE))
    durations: Dict[int, float] = field(
        default_factory=lambda: {
            STATE_RED: 10.0,
            STATE_GREEN: 10.0,
            STATE_YELLOW: 10.0,
        }
    )
    start_state: int = STATE_RED
    base_actor_id: int = 1200
    actor_id_stride: int = 20


def default_traffic_light_offsets() -> Dict[str, TrafficLightPlacement]:
    return {
        "south": TrafficLightPlacement(lateral_sign=-1.0, yaw_offset_deg=0.0),
        "north": TrafficLightPlacement(lateral_sign=1.0, yaw_offset_deg=180.0),
        "west": TrafficLightPlacement(lateral_sign=1.0, yaw_offset_deg=180.0),
        "east": TrafficLightPlacement(lateral_sign=-1.0, yaw_offset_deg=0.0),
    }


@dataclass
class IntersectionConfig:
    """十字路口场景配置。"""

    lanes_per_direction: int = 3
    lane_width: float = 3.5
    intersection_length: Optional[float] = None
    main_exit_length: float = 36.0
    side_exit_length: float = 32.0
    include_incoming_arm: bool = False
    crosswalk_clear_length: float = 4.0
    crosswalk_outside_offset: float = 1.75
    crosswalk_stripe_length: float = 3.5
    crosswalk_stripe_thickness: float = 0.4
    crosswalk_stripe_gap: float = 0.5
    line_height_offset: float = 0.02
    line_width: float = 0.15
    yellow_gap: float = 0.3
    road_color: Sequence[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    white_color: Sequence[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    yellow_color: Sequence[float] = field(default_factory=lambda: [1.0, 0.8, 0.0])
    traffic_light_style: TrafficLightStyle = field(default_factory=TrafficLightStyle)
    traffic_light_cycle: TrafficLightCycleConfig = field(default_factory=TrafficLightCycleConfig)
    traffic_light_offsets: Dict[str, TrafficLightPlacement] = field(default_factory=default_traffic_light_offsets)

    @property
    def total_width(self) -> float:
        return 2.0 * self.lanes_per_direction * self.lane_width

    @property
    def resolved_intersection_length(self) -> float:
        return self.intersection_length if self.intersection_length is not None else self.total_width

    def validate(self) -> None:
        if self.lanes_per_direction not in (1, 2, 3, 4):
            raise ValueError("lanes_per_direction 仅支持 1/2/3/4")
        if self.lane_width <= 0.0:
            raise ValueError("lane_width 必须大于 0")
        if self.resolved_intersection_length <= 0.0:
            raise ValueError("intersection_length 必须大于 0")


@dataclass
class IntersectionAnchor:
    """十字路口拼接锚点。"""

    x: float
    y: float
    z: float
    tx: float
    ty: float
    px: float
    py: float


class CustomTrafficLight:
    """自定义红绿灯模型，内置非阻塞状态推进逻辑。"""

    def __init__(
        self,
        qlabs: QuanserInteractiveLabs,
        location: Sequence[float],
        start_actor_id: int,
        direction_name: str,
        scale: float,
        yaw: float,
        pole_height: float,
        pole_width: float,
        horizontal_length: float,
        light_size: float,
    ) -> None:
        self.qlabs = qlabs
        self.location = list(location)
        self.base_id = start_actor_id
        self.direction_name = direction_name
        self.scale = scale
        self.yaw = yaw
        self.pole_height = pole_height
        self.pole_width = pole_width
        self.horizontal_length = horizontal_length
        self.light_size = light_size

        self.vertical_pole: Optional[QLabsBasicShape] = None
        self.horizontal_pole: Optional[QLabsBasicShape] = None
        self.housing: Optional[QLabsBasicShape] = None
        self.red_lamp: Optional[QLabsBasicShape] = None
        self.yellow_lamp: Optional[QLabsBasicShape] = None
        self.green_lamp: Optional[QLabsBasicShape] = None
        self.red_glow_front: Optional[QLabsBasicShape] = None
        self.red_glow_back: Optional[QLabsBasicShape] = None
        self.yellow_glow_front: Optional[QLabsBasicShape] = None
        self.yellow_glow_back: Optional[QLabsBasicShape] = None
        self.green_glow_front: Optional[QLabsBasicShape] = None
        self.green_glow_back: Optional[QLabsBasicShape] = None

        self.state: Optional[int] = None
        self.cycle_sequence: List[int] = list(DEFAULT_CYCLE_SEQUENCE)
        self.cycle_durations: Dict[int, float] = {
            STATE_RED: 10.0,
            STATE_GREEN: 10.0,
            STATE_YELLOW: 10.0,
        }
        self.elapsed_in_state = 0.0

        self.color_pole = [0.15, 0.15, 0.15]
        self.color_housing = [0.05, 0.05, 0.05]
        self.color_red_on = [1.0, 0.0, 0.0]
        self.color_red_off = [0.1, 0.0, 0.0]
        self.color_yellow_on = [1.0, 1.0, 0.0]
        self.color_yellow_off = [0.1, 0.1, 0.0]
        self.color_green_on = [0.0, 1.0, 0.0]
        self.color_green_off = [0.0, 0.1, 0.0]
        self.hidden_scale = [0.001, 0.001, 0.001]
        self._lamp_z = 0.0
        self._glow_offset_y = 0.0
        self._glow_scale = [0.001, 0.001, 0.001]
        self._lamp_x_positions: Dict[int, float] = {}

    def _set_lamp_material(self, lamp: Optional[QLabsBasicShape], color: Sequence[float], active: bool) -> None:
        if lamp is None:
            return
        lamp.set_material_properties(
            color=list(color),
            roughness=0.1,
            metallic=active,
            waitForConfirmation=True,
        )

    def _set_shape_transform(
        self,
        shape: Optional[QLabsBasicShape],
        location: Sequence[float],
        scale: Sequence[float],
    ) -> None:
        if shape is None:
            return
        shape.set_transform(
            location=list(location),
            rotation=[0.0, 0.0, self.yaw],
            scale=list(scale),
            waitForConfirmation=True,
        )

    def _spawn_shape(
        self,
        actor_number: int,
        location: Sequence[float],
        scale: Sequence[float],
        configuration: int,
        color: Sequence[float],
        roughness: float = 0.8,
        metallic: bool = False,
    ) -> QLabsBasicShape:
        shape = QLabsBasicShape(self.qlabs)
        shape.spawn_id(
            actorNumber=actor_number,
            location=list(location),
            rotation=[0.0, 0.0, self.yaw],
            scale=list(scale),
            configuration=configuration,
            waitForConfirmation=True,
        )
        shape.set_material_properties(
            color=list(color),
            roughness=roughness,
            metallic=metallic,
            waitForConfirmation=True,
        )
        return shape

    def _local_to_world(self, dx: float, dy: float, dz: float) -> List[float]:
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        return [
            self.location[0] + dx * cosine - dy * sine,
            self.location[1] + dx * sine + dy * cosine,
            self.location[2] + dz,
        ]

    def _spawn_glow_pair(
        self,
        start_actor_number: int,
        local_x: float,
        local_z: float,
        glow_offset_y: float,
        glow_scale: Sequence[float],
        color: Sequence[float],
    ) -> Tuple[QLabsBasicShape, QLabsBasicShape]:
        front = self._spawn_shape(
            start_actor_number,
            self._local_to_world(local_x, glow_offset_y, local_z),
            self.hidden_scale,
            2,
            color,
            roughness=0.0,
            metallic=False,
        )
        back = self._spawn_shape(
            start_actor_number + 1,
            self._local_to_world(local_x, -glow_offset_y, local_z),
            self.hidden_scale,
            2,
            color,
            roughness=0.0,
            metallic=False,
        )
        self._set_shape_transform(front, self._local_to_world(local_x, glow_offset_y, local_z), self.hidden_scale)
        self._set_shape_transform(back, self._local_to_world(local_x, -glow_offset_y, local_z), self.hidden_scale)
        return front, back

    def _set_glow_visible(
        self,
        front: Optional[QLabsBasicShape],
        back: Optional[QLabsBasicShape],
        local_x: float,
        local_z: float,
        glow_offset_y: float,
        visible_scale: Sequence[float],
        visible: bool,
    ) -> None:
        target_scale = list(visible_scale) if visible else self.hidden_scale
        self._set_shape_transform(front, self._local_to_world(local_x, glow_offset_y, local_z), target_scale)
        self._set_shape_transform(back, self._local_to_world(local_x, -glow_offset_y, local_z), target_scale)

    def spawn(self) -> None:
        """创建红绿灯几何体。"""
        scale = self.scale
        pole_height = self.pole_height * scale
        pole_width = self.pole_width * scale
        horizontal_length = self.horizontal_length * scale
        light_size = self.light_size * scale

        self.vertical_pole = self._spawn_shape(
            self.base_id,
            self._local_to_world(0.0, 0.0, pole_height / 2.0),
            [pole_width, pole_width, pole_height],
            0,
            self.color_pole,
            roughness=0.8,
            metallic=True,
        )
        self.horizontal_pole = self._spawn_shape(
            self.base_id + 1,
            self._local_to_world(-horizontal_length / 2.0, 0.0, pole_height - pole_width),
            [horizontal_length, pole_width * 0.8, pole_width * 0.8],
            0,
            self.color_pole,
            roughness=0.8,
            metallic=True,
        )

        housing_width = light_size * 4.5
        housing_height = light_size * 1.5
        housing_depth = light_size * 1.5
        housing_x = -horizontal_length + housing_width / 2.0 + 0.2 * scale
        if housing_x > -pole_width:
            housing_x = -horizontal_length / 2.0

        self.housing = self._spawn_shape(
            self.base_id + 2,
            self._local_to_world(housing_x, 0.0, pole_height - pole_width - housing_height),
            [housing_width, housing_depth, housing_height],
            0,
            self.color_housing,
            roughness=1.0,
            metallic=False,
        )

        lamp_scale = [light_size * 0.85, light_size * 0.45, light_size * 0.85]
        lamp_z = pole_height - pole_width - housing_height
        glow_scale = [light_size * 1.2, light_size * 0.2, light_size * 1.2]
        glow_offset_y = housing_depth / 2.0 + light_size * 0.1
        green_lamp_x = housing_x - light_size * 1.5
        yellow_lamp_x = housing_x
        red_lamp_x = housing_x + light_size * 1.5
        self.green_lamp = self._spawn_shape(
            self.base_id + 3,
            self._local_to_world(green_lamp_x, 0.0, lamp_z),
            lamp_scale,
            2,
            self.color_green_off,
            roughness=0.1,
            metallic=False,
        )
        self.yellow_lamp = self._spawn_shape(
            self.base_id + 4,
            self._local_to_world(yellow_lamp_x, 0.0, lamp_z),
            lamp_scale,
            2,
            self.color_yellow_off,
            roughness=0.1,
            metallic=False,
        )
        self.red_lamp = self._spawn_shape(
            self.base_id + 5,
            self._local_to_world(red_lamp_x, 0.0, lamp_z),
            lamp_scale,
            2,
            self.color_red_off,
            roughness=0.1,
            metallic=False,
        )

        self.green_glow_front, self.green_glow_back = self._spawn_glow_pair(
            self.base_id + 6,
            green_lamp_x,
            lamp_z,
            glow_offset_y,
            glow_scale,
            self.color_green_on,
        )
        self.yellow_glow_front, self.yellow_glow_back = self._spawn_glow_pair(
            self.base_id + 8,
            yellow_lamp_x,
            lamp_z,
            glow_offset_y,
            glow_scale,
            self.color_yellow_on,
        )
        self.red_glow_front, self.red_glow_back = self._spawn_glow_pair(
            self.base_id + 10,
            red_lamp_x,
            lamp_z,
            glow_offset_y,
            glow_scale,
            self.color_red_on,
        )

        self._glow_offset_y = glow_offset_y
        self._glow_scale = glow_scale
        self._lamp_z = lamp_z
        self._lamp_x_positions = {
            STATE_GREEN: green_lamp_x,
            STATE_YELLOW: yellow_lamp_x,
            STATE_RED: red_lamp_x,
        }

        # 确保 Actor 完成注册后再设置初始灯态，降低首帧不亮的概率。
        time.sleep(0.15)
        self.set_state(STATE_RED, force=True)

    def set_state(self, state: int, force: bool = False, reset_elapsed: bool = True) -> None:
        """切换红绿灯状态。"""
        if self.red_lamp is None or self.yellow_lamp is None or self.green_lamp is None:
            return
        if not force and self.state == state:
            return

        self._set_lamp_material(self.red_lamp, self.color_red_off, active=False)
        self._set_lamp_material(self.yellow_lamp, self.color_yellow_off, active=False)
        self._set_lamp_material(self.green_lamp, self.color_green_off, active=False)
        self._set_glow_visible(
            self.red_glow_front,
            self.red_glow_back,
            self._lamp_x_positions[STATE_RED],
            self._lamp_z,
            self._glow_offset_y,
            self._glow_scale,
            visible=False,
        )
        self._set_glow_visible(
            self.yellow_glow_front,
            self.yellow_glow_back,
            self._lamp_x_positions[STATE_YELLOW],
            self._lamp_z,
            self._glow_offset_y,
            self._glow_scale,
            visible=False,
        )
        self._set_glow_visible(
            self.green_glow_front,
            self.green_glow_back,
            self._lamp_x_positions[STATE_GREEN],
            self._lamp_z,
            self._glow_offset_y,
            self._glow_scale,
            visible=False,
        )

        if state == STATE_RED:
            self._set_lamp_material(self.red_lamp, self.color_red_on, active=True)
            self._set_glow_visible(
                self.red_glow_front,
                self.red_glow_back,
                self._lamp_x_positions[STATE_RED],
                self._lamp_z,
                self._glow_offset_y,
                self._glow_scale,
                visible=True,
            )
        elif state == STATE_GREEN:
            self._set_lamp_material(self.green_lamp, self.color_green_on, active=True)
            self._set_glow_visible(
                self.green_glow_front,
                self.green_glow_back,
                self._lamp_x_positions[STATE_GREEN],
                self._lamp_z,
                self._glow_offset_y,
                self._glow_scale,
                visible=True,
            )
        elif state == STATE_YELLOW:
            self._set_lamp_material(self.yellow_lamp, self.color_yellow_on, active=True)
            self._set_glow_visible(
                self.yellow_glow_front,
                self.yellow_glow_back,
                self._lamp_x_positions[STATE_YELLOW],
                self._lamp_z,
                self._glow_offset_y,
                self._glow_scale,
                visible=True,
            )
        else:
            raise ValueError(f"未知红绿灯状态: {state}")

        self.state = state
        if reset_elapsed:
            self.elapsed_in_state = 0.0

    def configure_cycle(
        self,
        cycle_sequence: Optional[Sequence[int]] = None,
        cycle_durations: Optional[Dict[int, float]] = None,
        start_state: int = STATE_RED,
    ) -> None:
        """配置非阻塞切灯循环。"""
        if cycle_sequence is not None:
            self.cycle_sequence = list(cycle_sequence)
        if cycle_durations is not None:
            self.cycle_durations = dict(cycle_durations)
        self.set_state(start_state, force=True)

    def update(self, delta_time: float) -> int:
        """推进一次状态机，不阻塞主线程。"""
        if self.state is None or not self.cycle_sequence:
            return STATE_RED
        if delta_time <= 0.0:
            return self.state

        self.elapsed_in_state += delta_time
        duration = self.cycle_durations.get(self.state, 10.0)
        while self.elapsed_in_state >= duration and duration > 0.0:
            self.elapsed_in_state -= duration
            current_index = self.cycle_sequence.index(self.state)
            next_state = self.cycle_sequence[(current_index + 1) % len(self.cycle_sequence)]
            self.set_state(next_state, force=True, reset_elapsed=False)
            duration = self.cycle_durations.get(self.state, 10.0)
        return self.state


class TrafficLightController:
    """统一驱动一组红绿灯的非阻塞控制器。"""

    def __init__(self, lights: Sequence[CustomTrafficLight], cycle: TrafficLightCycleConfig) -> None:
        self.lights = list(lights)
        self.last_update_time: Optional[float] = None
        self.cycle_sequence: List[int] = list(cycle.sequence)
        self.cycle_durations: Dict[int, float] = dict(cycle.durations)
        self.state = cycle.start_state
        self.elapsed_in_state = 0.0
        self.primary_group = [light for light in self.lights if light.direction_name in ("south", "north")]
        self.secondary_group = [light for light in self.lights if light.direction_name in ("west", "east")]
        self._apply_group_states(force=True)

    def _paired_state(self, state: int) -> int:
        if state == STATE_GREEN:
            return STATE_RED
        if state == STATE_RED:
            return STATE_GREEN
        return STATE_YELLOW

    def _apply_group_states(self, force: bool = False) -> None:
        paired_state = self._paired_state(self.state)
        for light in self.primary_group:
            light.set_state(self.state, force=force)
        for light in self.secondary_group:
            light.set_state(paired_state, force=force)

    def update(self, current_time: Optional[float] = None, delta_time: Optional[float] = None) -> None:
        if not self.lights:
            return

        if delta_time is None:
            now = time.time() if current_time is None else current_time
            if self.last_update_time is None:
                self.last_update_time = now
                return
            delta_time = max(0.0, now - self.last_update_time)
            self.last_update_time = now
        elif current_time is not None:
            self.last_update_time = current_time

        if delta_time <= 0.0:
            return

        self.elapsed_in_state += delta_time
        duration = self.cycle_durations.get(self.state, 10.0)
        while self.elapsed_in_state >= duration and duration > 0.0:
            self.elapsed_in_state -= duration
            current_index = self.cycle_sequence.index(self.state)
            self.state = self.cycle_sequence[(current_index + 1) % len(self.cycle_sequence)]
            self._apply_group_states(force=True)
            duration = self.cycle_durations.get(self.state, 10.0)


@dataclass
class IntersectionSceneResult:
    """十字路口构建结果。"""

    center: Tuple[float, float, float]
    qlines: QLabsSplineLine
    traffic_lights: List[CustomTrafficLight]
    controller: TrafficLightController

    def update_traffic_lights(
        self,
        current_time: Optional[float] = None,
        delta_time: Optional[float] = None,
    ) -> None:
        self.controller.update(current_time=current_time, delta_time=delta_time)


def safe_cleanup(qlabs: QuanserInteractiveLabs, timeout: float = 3.0) -> None:
    """安全清理当前场景。"""

    def cleanup_task() -> None:
        try:
            qlabs.destroy_all_spawned_actors()
            QLabsRealTime(qlabs).terminate_all_real_time_models()
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


def normalize2d(vx: float, vy: float) -> Vector2:
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return 0.0, -1.0
    return vx / length, vy / length


def spawn_strip(qlines: QLabsSplineLine, points: Sequence[Sequence[float]], color: Sequence[float]) -> None:
    if len(points) < 2:
        return
    qlines.spawn([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1)
    qlines.set_points(color, list(points), False)


def gen_road_surface(centerline: Sequence[CenterlinePoint], total_width: float) -> List[List[float]]:
    return [[x, y, z, total_width] for x, y, z, _, _ in centerline]


def gen_straight_surface(
    x0: float,
    y0: float,
    z0: float,
    tx: float,
    ty: float,
    length: float,
    width: float,
    num: int = 20,
) -> List[List[float]]:
    return [[x0 + tx * length * index / num, y0 + ty * length * index / num, z0, width] for index in range(num + 1)]


def gen_centerline_segment(
    x0: float,
    y0: float,
    z0: float,
    tx: float,
    ty: float,
    length: float,
    num: int = 30,
) -> List[CenterlinePoint]:
    tx, ty = normalize2d(tx, ty)
    left_normal = (-ty, tx)
    right_normal = (ty, -tx)
    segment: List[CenterlinePoint] = []
    for index in range(num + 1):
        distance = length * index / num
        segment.append((x0 + tx * distance, y0 + ty * distance, z0, left_normal, right_normal))
    return segment


def gen_solid_line(centerline: Sequence[CenterlinePoint], offset: float, height_offset: float, line_width: float) -> List[List[float]]:
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
    step = dash + gap
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
        current_arc += step


def trim_centerline_by_distance(
    centerline: Sequence[CenterlinePoint],
    trim_start: float = 0.0,
    trim_end: float = 0.0,
) -> List[CenterlinePoint]:
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
    config: IntersectionConfig,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    draw_edge: bool = True,
    draw_double_yellow: bool = True,
) -> None:
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


def rotate_yaw(base_yaw: float, offset_deg: float) -> float:
    return base_yaw + math.radians(offset_deg)


def draw_zebra_crosswalk(
    qlines: QLabsSplineLine,
    cx: float,
    cy: float,
    cz: float,
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    span_width: float,
    stripe_length: float,
    stripe_thickness: float,
    stripe_gap: float,
    z_offset: float = 0.02,
) -> None:
    total_width = max(1.0, span_width - 1.0)
    stripe_count = max(1, int(total_width / (stripe_thickness + stripe_gap)))
    occupied_width = stripe_count * stripe_thickness + (stripe_count - 1) * stripe_gap
    start_offset = -occupied_width / 2.0 + stripe_thickness / 2.0
    half_length = stripe_length / 2.0

    for index in range(stripe_count):
        lateral_offset = start_offset + index * (stripe_thickness + stripe_gap)
        stripe_center_x = cx - nx * lateral_offset
        stripe_center_y = cy - ny * lateral_offset
        start_x = stripe_center_x - tx * half_length
        start_y = stripe_center_y - ty * half_length
        end_x = stripe_center_x + tx * half_length
        end_y = stripe_center_y + ty * half_length
        spawn_strip(
            qlines,
            [
                [start_x, start_y, cz + z_offset, stripe_thickness],
                [end_x, end_y, cz + z_offset, stripe_thickness],
            ],
            [1.0, 1.0, 1.0],
        )


def _resolve_anchor(anchor: object) -> IntersectionAnchor:
    """兼容 dataclass、普通对象或字典形式的锚点输入。"""
    if isinstance(anchor, dict):
        return IntersectionAnchor(
            x=anchor["x"],
            y=anchor["y"],
            z=anchor["z"],
            tx=anchor["tx"],
            ty=anchor["ty"],
            px=anchor["px"],
            py=anchor["py"],
        )

    required = ("x", "y", "z", "tx", "ty", "px", "py")
    if not all(hasattr(anchor, key) for key in required):
        raise ValueError("anchor 必须包含 x/y/z/tx/ty/px/py")
    return IntersectionAnchor(
        x=float(getattr(anchor, "x")),
        y=float(getattr(anchor, "y")),
        z=float(getattr(anchor, "z")),
        tx=float(getattr(anchor, "tx")),
        ty=float(getattr(anchor, "ty")),
        px=float(getattr(anchor, "px")),
        py=float(getattr(anchor, "py")),
    )


def spawn_custom_traffic_lights(
    qlabs: QuanserInteractiveLabs,
    cx: float,
    cy: float,
    cz: float,
    tx: float,
    ty: float,
    px: float,
    py: float,
    intersection_half: float,
    total_width: float,
    style: TrafficLightStyle,
    cycle: TrafficLightCycleConfig,
    placements: Dict[str, TrafficLightPlacement],
) -> List[CustomTrafficLight]:
    """生成四向红绿灯。"""
    one_way_width = total_width / 2.0
    side_offset = one_way_width - style.pole_lateral_margin
    horizontal_span = max(one_way_width * style.horizontal_span_ratio, style.light_size * 4.5)

    configs = [
        {
            "name": "south",
            "base": (cx + tx * (intersection_half - style.far_side_margin), cy + ty * (intersection_half - style.far_side_margin)),
            "cross": (px, py),
            "yaw": math.atan2(py, px),
        },
        {
            "name": "north",
            "base": (cx - tx * (intersection_half - style.far_side_margin), cy - ty * (intersection_half - style.far_side_margin)),
            "cross": (px, py),
            "yaw": math.atan2(py, px),
        },
        {
            "name": "west",
            "base": (cx + px * (intersection_half - style.far_side_margin), cy + py * (intersection_half - style.far_side_margin)),
            "cross": (tx, ty),
            "yaw": math.atan2(ty, tx),
        },
        {
            "name": "east",
            "base": (cx - px * (intersection_half - style.far_side_margin), cy - py * (intersection_half - style.far_side_margin)),
            "cross": (tx, ty),
            "yaw": math.atan2(ty, tx),
        },
    ]

    lights: List[CustomTrafficLight] = []
    for index, light_config in enumerate(configs):
        placement = placements.get(light_config["name"], default_traffic_light_offsets()[light_config["name"]])
        cross_x, cross_y = light_config["cross"]
        base_x, base_y = light_config["base"]
        longitudinal_x = tx if light_config["name"] in ("south", "north") else px
        longitudinal_y = ty if light_config["name"] in ("south", "north") else py
        light_x = base_x + cross_x * (side_offset * placement.lateral_sign + placement.lateral_shift)
        light_y = base_y + cross_y * (side_offset * placement.lateral_sign + placement.lateral_shift)
        light_x += longitudinal_x * placement.longitudinal_shift
        light_y += longitudinal_y * placement.longitudinal_shift
        light_yaw = rotate_yaw(light_config["yaw"], placement.yaw_offset_deg + 180.0)

        try:
            light = CustomTrafficLight(
                qlabs,
                location=[light_x, light_y, cz],
                start_actor_id=cycle.base_actor_id + index * cycle.actor_id_stride,
                direction_name=light_config["name"],
                scale=style.scale,
                yaw=light_yaw,
                pole_height=style.pole_height,
                pole_width=style.pole_width,
                horizontal_length=horizontal_span,
                light_size=style.light_size,
            )
            light.spawn()
            lights.append(light)
        except Exception as exc:
            print(f"[警告] {light_config['name']} 方向红绿灯创建失败: {exc}")
    return lights


def build_intersection_scene(
    qlabs: QuanserInteractiveLabs,
    config: IntersectionConfig | None = None,
    anchor: object | None = None,
    center: Sequence[float] | None = None,
    forward: Sequence[float] | None = None,
    cleanup_first: bool = False,
) -> IntersectionSceneResult:
    """构建十字路口、斑马线和红绿灯。"""
    config = config or IntersectionConfig()
    config.validate()
    if cleanup_first:
        safe_cleanup(qlabs)

    if anchor is not None:
        resolved_anchor = _resolve_anchor(anchor)
        tx, ty = normalize2d(resolved_anchor.tx, resolved_anchor.ty)
        px, py = normalize2d(resolved_anchor.px, resolved_anchor.py)
        center_x = resolved_anchor.x + tx * (config.resolved_intersection_length / 2.0)
        center_y = resolved_anchor.y + ty * (config.resolved_intersection_length / 2.0)
        center_z = resolved_anchor.z
    else:
        if center is None or forward is None:
            raise ValueError("未提供 anchor 时，必须同时提供 center 和 forward")
        tx, ty = normalize2d(float(forward[0]), float(forward[1]))
        px, py = -ty, tx
        center_x = float(center[0])
        center_y = float(center[1])
        center_z = float(center[2])

    qlines = QLabsSplineLine(qlabs)
    total_width = config.total_width
    intersection_half = config.resolved_intersection_length / 2.0

    main_intersection = gen_straight_surface(
        center_x - tx * intersection_half,
        center_y - ty * intersection_half,
        center_z,
        tx,
        ty,
        config.resolved_intersection_length,
        total_width,
        num=24,
    )
    side_intersection = gen_straight_surface(
        center_x - px * intersection_half,
        center_y - py * intersection_half,
        center_z,
        px,
        py,
        config.resolved_intersection_length,
        total_width,
        num=24,
    )
    spawn_strip(qlines, main_intersection, config.road_color)
    spawn_strip(qlines, side_intersection, config.road_color)

    north_arm = gen_centerline_segment(
        center_x + tx * intersection_half,
        center_y + ty * intersection_half,
        center_z,
        tx,
        ty,
        config.main_exit_length,
        num=64,
    )
    west_arm = gen_centerline_segment(
        center_x - px * intersection_half,
        center_y - py * intersection_half,
        center_z,
        -px,
        -py,
        config.side_exit_length,
        num=58,
    )
    east_arm = gen_centerline_segment(
        center_x + px * intersection_half,
        center_y + py * intersection_half,
        center_z,
        px,
        py,
        config.side_exit_length,
        num=58,
    )
    spawn_strip(qlines, gen_road_surface(north_arm, total_width), config.road_color)
    spawn_strip(qlines, gen_road_surface(west_arm, total_width), config.road_color)
    spawn_strip(qlines, gen_road_surface(east_arm, total_width), config.road_color)
    draw_lane_markings(qlines, north_arm, config, trim_start=config.crosswalk_clear_length)
    draw_lane_markings(qlines, west_arm, config, trim_start=config.crosswalk_clear_length)
    draw_lane_markings(qlines, east_arm, config, trim_start=config.crosswalk_clear_length)

    if config.include_incoming_arm:
        south_arm = gen_centerline_segment(
            center_x - tx * intersection_half,
            center_y - ty * intersection_half,
            center_z,
            -tx,
            -ty,
            config.main_exit_length,
            num=64,
        )
        spawn_strip(qlines, gen_road_surface(south_arm, total_width), config.road_color)
        draw_lane_markings(qlines, south_arm, config, trim_start=config.crosswalk_clear_length)

    south_walk_center = (
        center_x - tx * (intersection_half + config.crosswalk_outside_offset),
        center_y - ty * (intersection_half + config.crosswalk_outside_offset),
    )
    north_walk_center = (
        center_x + tx * (intersection_half + config.crosswalk_outside_offset),
        center_y + ty * (intersection_half + config.crosswalk_outside_offset),
    )
    west_walk_center = (
        center_x - px * (intersection_half + config.crosswalk_outside_offset),
        center_y - py * (intersection_half + config.crosswalk_outside_offset),
    )
    east_walk_center = (
        center_x + px * (intersection_half + config.crosswalk_outside_offset),
        center_y + py * (intersection_half + config.crosswalk_outside_offset),
    )
    draw_zebra_crosswalk(
        qlines,
        south_walk_center[0],
        south_walk_center[1],
        center_z,
        tx,
        ty,
        px,
        py,
        total_width,
        config.crosswalk_stripe_length,
        config.crosswalk_stripe_thickness,
        config.crosswalk_stripe_gap,
        z_offset=config.line_height_offset,
    )
    draw_zebra_crosswalk(
        qlines,
        north_walk_center[0],
        north_walk_center[1],
        center_z,
        tx,
        ty,
        px,
        py,
        total_width,
        config.crosswalk_stripe_length,
        config.crosswalk_stripe_thickness,
        config.crosswalk_stripe_gap,
        z_offset=config.line_height_offset,
    )
    draw_zebra_crosswalk(
        qlines,
        west_walk_center[0],
        west_walk_center[1],
        center_z,
        px,
        py,
        tx,
        ty,
        total_width,
        config.crosswalk_stripe_length,
        config.crosswalk_stripe_thickness,
        config.crosswalk_stripe_gap,
        z_offset=config.line_height_offset,
    )
    draw_zebra_crosswalk(
        qlines,
        east_walk_center[0],
        east_walk_center[1],
        center_z,
        px,
        py,
        tx,
        ty,
        total_width,
        config.crosswalk_stripe_length,
        config.crosswalk_stripe_thickness,
        config.crosswalk_stripe_gap,
        z_offset=config.line_height_offset,
    )

    traffic_lights = spawn_custom_traffic_lights(
        qlabs,
        center_x,
        center_y,
        center_z,
        tx,
        ty,
        px,
        py,
        intersection_half,
        total_width,
        config.traffic_light_style,
        config.traffic_light_cycle,
        config.traffic_light_offsets,
    )
    controller = TrafficLightController(traffic_lights, config.traffic_light_cycle)
    print(
        f"✅ 十字路口生成完成：双向每侧 {config.lanes_per_direction} 车道，"
        f"路口直线长度 {config.resolved_intersection_length:.2f} m，红绿灯数量 {len(traffic_lights)}"
    )
    return IntersectionSceneResult(
        center=(center_x, center_y, center_z),
        qlines=qlines,
        traffic_lights=traffic_lights,
        controller=controller,
    )


def run_intersection_demo(duration: Optional[float] = 20.0, step: float = 0.1) -> None:
    """十字路口库独立测试入口。"""
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("❌ QLabs 连接失败，请先打开 QLabs 软件")
        return

    try:
        config = IntersectionConfig(include_incoming_arm=True)
        result = build_intersection_scene(
            qlabs,
            config=config,
            center=[0.0, 0.0, 0.0],
            forward=[0.0, 1.0],
            cleanup_first=True,
        )
        print("红绿灯循环已启动，按 Ctrl+C 手动停止。")
        start_time = time.time()
        while duration is None or time.time() - start_time < duration:
            result.update_traffic_lights(current_time=time.time())
            time.sleep(step)
    except KeyboardInterrupt:
        print("\n已手动停止红绿灯演示。")
    finally:
        qlabs.close()


if __name__ == "__main__":
    run_intersection_demo(duration=None)