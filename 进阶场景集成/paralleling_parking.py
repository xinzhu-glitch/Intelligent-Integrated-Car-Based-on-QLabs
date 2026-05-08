import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from qvl.basic_shape import QLabsBasicShape
from qvl.real_time import QLabsRealTime
from qvl.spline_line import QLabsSplineLine


@dataclass
class ParkingSpot:
    top_y: float
    bottom_y: float


@dataclass
class ParkingSign:
    x: float
    y: float
    yaw: float
    pole_height: float
    board_width: float
    board_height: float


@dataclass
class ParkingRoadAnchor:
    x: float
    y: float
    z: float
    tx: float
    ty: float
    px: float
    py: float
    total_width: float
    lane_width: float
    total_lanes: int


@dataclass
class ParkingRoadSceneGeometry:
    road_left_x: float
    road_right_x: float
    road_start_y: float
    road_end_y: float
    parking_lane_edge_x: float
    spot_right_x: float
    direction_divider_x: float
    parking_entry_start_y: float
    parking_entry_end_y: float
    lane_boundary_xs: List[float] = field(default_factory=list)
    parking_spots: List[ParkingSpot] = field(default_factory=list)
    parking_sign: Optional[ParkingSign] = None
    entry_anchor: Optional[ParkingRoadAnchor] = None
    exit_anchor: Optional[ParkingRoadAnchor] = None


@dataclass
class ParkingRoadSceneConfig:
    """可拼接的直路侧方位停车场景配置。"""

    total_lanes: Optional[int] = None
    lanes_per_direction: int = 3
    forward_lanes: Optional[int] = None
    lane_width: float = 3.5
    road_center_x: float = 0.0
    road_start_y: float = 20.0
    total_road_length: float = 135.0

    parking_spot_count: int = 3
    parking_spot_length: float = 6.0
    parking_depth: float = 3.0
    parking_zone_start_y: Optional[float] = None
    parking_side_sign: int = -1

    background_width: float = 16.0
    line_height: float = 0.02
    line_width: float = 0.15
    parking_entry_dash_length: float = 0.4
    parking_entry_gap_length: float = 0.4
    lane_dash_length: float = 1.2
    lane_gap_length: float = 1.0
    divider_gap: float = 0.18

    road_color: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    white_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    yellow_color: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.0])
    sign_pole_color: List[float] = field(default_factory=lambda: [0.55, 0.55, 0.55])
    sign_board_color: List[float] = field(default_factory=lambda: [0.05, 0.35, 0.85])
    sign_letter_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    roadside_sign_offset_x: float = 0.75
    roadside_sign_offset_y: float = 1.8
    sign_face_oncoming_yaw: float = 0.0
    sign_pole_height: float = 2.2
    sign_pole_width: float = 0.08
    sign_board_width: float = 0.72
    sign_board_height: float = 0.9
    sign_board_thickness: float = 0.08
    sign_letter_thickness: float = 0.03

    def validate(self) -> None:
        if self.lane_width <= 0.0:
            raise ValueError("lane_width 必须大于 0")
        if self.total_road_length <= 0.0:
            raise ValueError("total_road_length 必须大于 0")
        if self.parking_spot_count < 1:
            raise ValueError("parking_spot_count 至少为 1")
        if self.parking_spot_length <= 0.0 or self.parking_depth <= 0.0:
            raise ValueError("停车位长度和深度必须大于 0")
        if self.parking_side_sign not in (-1, 1):
            raise ValueError("parking_side_sign 仅支持 -1 或 1")

        resolved_total_lanes = self.resolved_total_lanes
        if resolved_total_lanes < 2:
            raise ValueError("双向道路总车道数至少为 2")

        resolved_forward_lanes = self.resolved_forward_lanes
        if not 1 <= resolved_forward_lanes < resolved_total_lanes:
            raise ValueError("forward_lanes 必须在 1 到总车道数减 1 之间")

        if self.background_width < self.total_width + self.parking_depth + 2.0:
            self.background_width = self.total_width + self.parking_depth + 4.0
        if self.parking_zone_end_y < self.road_end_y:
            raise ValueError("停车区超出道路长度，请增大 total_road_length 或调整停车位参数")

    @property
    def resolved_total_lanes(self) -> int:
        if self.lanes_per_direction < 1:
            raise ValueError("lanes_per_direction 必须大于 0")
        if self.total_lanes is None:
            return 2 * self.lanes_per_direction
        if self.total_lanes != 2 * self.lanes_per_direction:
            raise ValueError("total_lanes 必须等于 2 * lanes_per_direction")
        return self.total_lanes

    @property
    def resolved_forward_lanes(self) -> int:
        if self.forward_lanes is not None and self.forward_lanes != self.lanes_per_direction:
            raise ValueError("当前双向对称道路要求 forward_lanes 与 lanes_per_direction 一致")
        if self.forward_lanes is not None:
            return self.forward_lanes
        return self.lanes_per_direction

    @property
    def resolved_backward_lanes(self) -> int:
        return self.resolved_total_lanes - self.resolved_forward_lanes

    @property
    def total_width(self) -> float:
        return self.resolved_total_lanes * self.lane_width

    @property
    def road_end_y(self) -> float:
        return self.road_start_y - self.total_road_length

    @property
    def road_left_x(self) -> float:
        return self.road_center_x - self.total_width / 2.0

    @property
    def road_right_x(self) -> float:
        return self.road_center_x + self.total_width / 2.0

    @property
    def spot_right_x(self) -> float:
        return self.parking_lane_edge_x + self.parking_side_sign * self.parking_depth

    @property
    def parking_lane_edge_x(self) -> float:
        return self.road_right_x if self.parking_side_sign > 0 else self.road_left_x

    @property
    def parking_surface_center_x(self) -> float:
        return (self.parking_lane_edge_x + self.spot_right_x) / 2.0

    @property
    def parking_zone_end_y(self) -> float:
        return self.resolved_parking_zone_start_y - self.parking_spot_count * self.parking_spot_length

    @property
    def parking_zone_total_length(self) -> float:
        return self.parking_spot_count * self.parking_spot_length

    @property
    def resolved_parking_zone_start_y(self) -> float:
        if self.parking_zone_start_y is not None:
            return self.parking_zone_start_y
        road_mid_y = (self.road_start_y + self.road_end_y) / 2.0
        return road_mid_y + self.parking_zone_total_length / 2.0


def _normalize2d(vx: float, vy: float) -> tuple[float, float]:
    length = math.hypot(vx, vy)
    if length < 1e-6:
        raise ValueError("方向向量长度不能为 0")
    return vx / length, vy / length


def _resolve_anchor(anchor: object) -> ParkingRoadAnchor:
    if isinstance(anchor, dict):
        return ParkingRoadAnchor(
            x=float(anchor["x"]),
            y=float(anchor["y"]),
            z=float(anchor["z"]),
            tx=float(anchor["tx"]),
            ty=float(anchor["ty"]),
            px=float(anchor["px"]),
            py=float(anchor["py"]),
            total_width=float(anchor.get("total_width", 0.0)),
            lane_width=float(anchor.get("lane_width", 0.0)),
            total_lanes=int(anchor.get("total_lanes", 0)),
        )
    required = ("x", "y", "z", "tx", "ty", "px", "py")
    if not all(hasattr(anchor, key) for key in required):
        raise ValueError("anchor 必须包含 x/y/z/tx/ty/px/py")
    return ParkingRoadAnchor(
        x=float(getattr(anchor, "x")),
        y=float(getattr(anchor, "y")),
        z=float(getattr(anchor, "z")),
        tx=float(getattr(anchor, "tx")),
        ty=float(getattr(anchor, "ty")),
        px=float(getattr(anchor, "px")),
        py=float(getattr(anchor, "py")),
        total_width=float(getattr(anchor, "total_width", 0.0)),
        lane_width=float(getattr(anchor, "lane_width", 0.0)),
        total_lanes=int(getattr(anchor, "total_lanes", 0)),
    )


def resolve_scene_config(config: Optional[ParkingRoadSceneConfig] = None, **overrides) -> ParkingRoadSceneConfig:
    scene_config = config if config is not None else ParkingRoadSceneConfig()
    if overrides:
        config_values = dict(scene_config.__dict__)
        config_values.update(overrides)
        scene_config = ParkingRoadSceneConfig(**config_values)
    scene_config.validate()
    return scene_config


def build_scene_geometry(config: Optional[ParkingRoadSceneConfig] = None, **overrides) -> ParkingRoadSceneGeometry:
    scene_config = resolve_scene_config(config, **overrides)

    parking_spots: List[ParkingSpot] = []
    current_top = scene_config.resolved_parking_zone_start_y
    for _ in range(scene_config.parking_spot_count):
        current_bottom = current_top - scene_config.parking_spot_length
        parking_spots.append(ParkingSpot(top_y=current_top, bottom_y=current_bottom))
        current_top = current_bottom

    direction_divider_x = scene_config.road_left_x + scene_config.resolved_backward_lanes * scene_config.lane_width
    lane_boundary_xs = []
    for lane_index in range(1, scene_config.resolved_total_lanes):
        boundary_x = scene_config.road_left_x + lane_index * scene_config.lane_width
        if abs(boundary_x - direction_divider_x) > 1e-6:
            lane_boundary_xs.append(boundary_x)

    parking_sign = ParkingSign(
        x=scene_config.spot_right_x + scene_config.parking_side_sign * scene_config.roadside_sign_offset_x,
        y=scene_config.resolved_parking_zone_start_y + scene_config.roadside_sign_offset_y,
        yaw=scene_config.sign_face_oncoming_yaw,
        pole_height=scene_config.sign_pole_height,
        board_width=scene_config.sign_board_width,
        board_height=scene_config.sign_board_height,
    )

    entry_anchor = ParkingRoadAnchor(
        x=scene_config.road_center_x,
        y=scene_config.road_start_y,
        z=0.0,
        tx=0.0,
        ty=-1.0,
        px=1.0,
        py=0.0,
        total_width=scene_config.total_width,
        lane_width=scene_config.lane_width,
        total_lanes=scene_config.resolved_total_lanes,
    )
    exit_anchor = ParkingRoadAnchor(
        x=scene_config.road_center_x,
        y=scene_config.road_end_y,
        z=0.0,
        tx=0.0,
        ty=-1.0,
        px=1.0,
        py=0.0,
        total_width=scene_config.total_width,
        lane_width=scene_config.lane_width,
        total_lanes=scene_config.resolved_total_lanes,
    )

    return ParkingRoadSceneGeometry(
        road_left_x=scene_config.road_left_x,
        road_right_x=scene_config.road_right_x,
        road_start_y=scene_config.road_start_y,
        road_end_y=scene_config.road_end_y,
        parking_lane_edge_x=scene_config.parking_lane_edge_x,
        spot_right_x=scene_config.spot_right_x,
        direction_divider_x=direction_divider_x,
        parking_entry_start_y=scene_config.resolved_parking_zone_start_y,
        parking_entry_end_y=scene_config.parking_zone_end_y,
        lane_boundary_xs=lane_boundary_xs,
        parking_spots=parking_spots,
        parking_sign=parking_sign,
        entry_anchor=entry_anchor,
        exit_anchor=exit_anchor,
    )


def _spawn_line_segment(qlines: QLabsSplineLine, start, end, color) -> None:
    qlines.spawn(location=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0], configuration=1)
    qlines.set_points(color=color, pointList=[list(start), list(end)], alignEndPointTangents=False)


def _spawn_box(qlabs, location, scale, color, yaw: float = 0.0) -> None:
    shape = QLabsBasicShape(qlabs)
    shape.spawn(
        location=location,
        rotation=[0.0, 0.0, yaw],
        scale=scale,
        configuration=QLabsBasicShape.SHAPE_CUBE,
        waitForConfirmation=True,
    )
    shape.set_material_properties(color, roughness=0.8, metallic=False, waitForConfirmation=False)


def _local_to_world(origin_x: float, origin_y: float, yaw: float, dx: float, dy: float, z: float) -> List[float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        origin_x + dx * cos_yaw - dy * sin_yaw,
        origin_y + dx * sin_yaw + dy * cos_yaw,
        z,
    ]


def _resolve_scene_pose(scene_config: ParkingRoadSceneConfig, anchor: object | None) -> tuple[float, float, float, float, float, float, float]:
    if anchor is None:
        return scene_config.road_center_x, scene_config.road_start_y, 0.0, 0.0, -1.0, 1.0, 0.0
    resolved_anchor = _resolve_anchor(anchor)
    tx, ty = _normalize2d(resolved_anchor.tx, resolved_anchor.ty)
    lateral_x, lateral_y = -ty, tx
    return resolved_anchor.x, resolved_anchor.y, resolved_anchor.z, tx, ty, lateral_x, lateral_y


def _transform_local_point(
    scene_config: ParkingRoadSceneConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tx: float,
    ty: float,
    lateral_x: float,
    lateral_y: float,
    x: float,
    y: float,
    z: float = 0.0,
) -> list[float]:
    lateral_offset = float(x - scene_config.road_center_x)
    longitudinal_offset = float(scene_config.road_start_y - y)
    return [
        origin_x + lateral_x * lateral_offset + tx * longitudinal_offset,
        origin_y + lateral_y * lateral_offset + ty * longitudinal_offset,
        origin_z + z,
    ]


def _spawn_transformed_line_segment(
    qlines: QLabsSplineLine,
    scene_config: ParkingRoadSceneConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tx: float,
    ty: float,
    lateral_x: float,
    lateral_y: float,
    start,
    end,
    color,
) -> None:
    start_xyz = _transform_local_point(scene_config, origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y, start[0], start[1], start[2])
    end_xyz = _transform_local_point(scene_config, origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y, end[0], end[1], end[2])
    _spawn_line_segment(
        qlines,
        [start_xyz[0], start_xyz[1], start_xyz[2], float(start[3])],
        [end_xyz[0], end_xyz[1], end_xyz[2], float(end[3])],
        color,
    )


def _spawn_transformed_surface_strip(
    qlines: QLabsSplineLine,
    scene_config: ParkingRoadSceneConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tx: float,
    ty: float,
    lateral_x: float,
    lateral_y: float,
    center_x: float,
    start_y: float,
    end_y: float,
    width: float,
    color,
) -> None:
    start_xyz = _transform_local_point(scene_config, origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y, center_x, start_y, 0.0)
    end_xyz = _transform_local_point(scene_config, origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y, center_x, end_y, 0.0)
    qlines.spawn(location=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0], configuration=1)
    qlines.set_points(
        color=color,
        pointList=[
            [start_xyz[0], start_xyz[1], start_xyz[2], float(width)],
            [end_xyz[0], end_xyz[1], end_xyz[2], float(width)],
        ],
        alignEndPointTangents=False,
    )


def _spawn_vertical_dashed_line(
    qlines: QLabsSplineLine,
    scene_config: ParkingRoadSceneConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tx: float,
    ty: float,
    lateral_x: float,
    lateral_y: float,
    x: float,
    start_y: float,
    end_y: float,
    line_h: float,
    line_w: float,
    color,
    dash_length: float,
    gap_length: float,
) -> None:
    current_y = start_y
    while current_y > end_y + 1e-6:
        next_y = max(current_y - dash_length, end_y)
        _spawn_transformed_line_segment(
            qlines,
            scene_config,
            origin_x,
            origin_y,
            origin_z,
            tx,
            ty,
            lateral_x,
            lateral_y,
            [x, current_y, line_h, line_w],
            [x, next_y, line_h, line_w],
            color,
        )
        current_y -= dash_length + gap_length


def _spawn_parking_sign(qlabs, geometry: ParkingRoadSceneGeometry, config: ParkingRoadSceneConfig) -> None:
    sign = geometry.parking_sign
    if sign is None:
        return

    _spawn_box(
        qlabs,
        [sign.x, sign.y, sign.pole_height / 2.0],
        [config.sign_pole_width, config.sign_pole_width, sign.pole_height],
        config.sign_pole_color,
        yaw=sign.yaw,
    )

    board_center_z = sign.pole_height
    _spawn_box(
        qlabs,
        [sign.x, sign.y, board_center_z],
        [sign.board_width, config.sign_board_thickness, sign.board_height],
        config.sign_board_color,
        yaw=sign.yaw,
    )

    face_y = -config.sign_board_thickness / 2.0 - config.sign_letter_thickness / 2.0
    stem_height = sign.board_height * 0.6
    bar_width = sign.board_width * 0.34
    bar_height = sign.board_height * 0.16
    leg_height = sign.board_height * 0.24

    _spawn_box(
        qlabs,
        _local_to_world(sign.x, sign.y, sign.yaw, -sign.board_width * 0.18, face_y, board_center_z),
        [sign.board_width * 0.1, config.sign_letter_thickness, stem_height],
        config.sign_letter_color,
        yaw=sign.yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign.x, sign.y, sign.yaw, 0.0, face_y, board_center_z + sign.board_height * 0.18),
        [bar_width, config.sign_letter_thickness, bar_height],
        config.sign_letter_color,
        yaw=sign.yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign.x, sign.y, sign.yaw, 0.0, face_y, board_center_z),
        [bar_width, config.sign_letter_thickness, bar_height],
        config.sign_letter_color,
        yaw=sign.yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign.x, sign.y, sign.yaw, sign.board_width * 0.13, face_y, board_center_z + sign.board_height * 0.09),
        [sign.board_width * 0.1, config.sign_letter_thickness, leg_height],
        config.sign_letter_color,
        yaw=sign.yaw,
    )


def _spawn_parking_sign_transformed(
    qlabs,
    geometry: ParkingRoadSceneGeometry,
    config: ParkingRoadSceneConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tx: float,
    ty: float,
    lateral_x: float,
    lateral_y: float,
) -> None:
    sign = geometry.parking_sign
    if sign is None:
        return

    sign_center = _transform_local_point(
        config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        sign.x,
        sign.y,
        0.0,
    )
    world_yaw = math.atan2(lateral_y, lateral_x) + sign.yaw

    _spawn_box(
        qlabs,
        [sign_center[0], sign_center[1], sign_center[2] + sign.pole_height / 2.0],
        [config.sign_pole_width, config.sign_pole_width, sign.pole_height],
        config.sign_pole_color,
        yaw=world_yaw,
    )

    board_center_z = sign_center[2] + sign.pole_height
    _spawn_box(
        qlabs,
        [sign_center[0], sign_center[1], board_center_z],
        [sign.board_width, config.sign_board_thickness, sign.board_height],
        config.sign_board_color,
        yaw=world_yaw,
    )

    face_y = -config.sign_board_thickness / 2.0 - config.sign_letter_thickness / 2.0
    stem_height = sign.board_height * 0.6
    bar_width = sign.board_width * 0.34
    bar_height = sign.board_height * 0.16
    leg_height = sign.board_height * 0.24

    _spawn_box(
        qlabs,
        _local_to_world(sign_center[0], sign_center[1], world_yaw, -sign.board_width * 0.18, face_y, board_center_z),
        [sign.board_width * 0.1, config.sign_letter_thickness, stem_height],
        config.sign_letter_color,
        yaw=world_yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign_center[0], sign_center[1], world_yaw, 0.0, face_y, board_center_z + sign.board_height * 0.18),
        [bar_width, config.sign_letter_thickness, bar_height],
        config.sign_letter_color,
        yaw=world_yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign_center[0], sign_center[1], world_yaw, 0.0, face_y, board_center_z),
        [bar_width, config.sign_letter_thickness, bar_height],
        config.sign_letter_color,
        yaw=world_yaw,
    )
    _spawn_box(
        qlabs,
        _local_to_world(sign_center[0], sign_center[1], world_yaw, sign.board_width * 0.13, face_y, board_center_z + sign.board_height * 0.09),
        [sign.board_width * 0.1, config.sign_letter_thickness, leg_height],
        config.sign_letter_color,
        yaw=world_yaw,
    )


def setup_map(
    qlabs,
    config: Optional[ParkingRoadSceneConfig] = None,
    clear_existing: bool = True,
    anchor: object | None = None,
    **overrides,
) -> ParkingRoadSceneGeometry:
    """构建一段直行主路，并在右侧附加三个侧方停车位。"""
    scene_config = resolve_scene_config(config, **overrides)
    geometry = build_scene_geometry(scene_config)

    if clear_existing:
        qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(0.5)

    spline_lines = QLabsSplineLine(qlabs)
    origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y = _resolve_scene_pose(scene_config, anchor)

    _spawn_transformed_surface_strip(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        scene_config.road_center_x,
        geometry.road_start_y,
        geometry.road_end_y,
        scene_config.total_width,
        scene_config.road_color,
    )
    _spawn_transformed_surface_strip(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        scene_config.parking_surface_center_x,
        geometry.parking_entry_start_y,
        geometry.parking_entry_end_y,
        scene_config.parking_depth,
        scene_config.road_color,
    )

    line_h = scene_config.line_height
    line_w = scene_config.line_width

    _spawn_transformed_line_segment(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        [geometry.road_left_x, geometry.road_start_y, line_h, line_w],
        [geometry.road_left_x, geometry.road_end_y, line_h, line_w],
        scene_config.white_color,
    )
    _spawn_transformed_line_segment(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        [geometry.parking_lane_edge_x, geometry.road_start_y, line_h, line_w],
        [geometry.parking_lane_edge_x, geometry.parking_entry_start_y, line_h, line_w],
        scene_config.white_color,
    )
    _spawn_transformed_line_segment(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        [geometry.parking_lane_edge_x, geometry.parking_entry_end_y, line_h, line_w],
        [geometry.parking_lane_edge_x, geometry.road_end_y, line_h, line_w],
        scene_config.white_color,
    )
    _spawn_vertical_dashed_line(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        geometry.parking_lane_edge_x,
        geometry.parking_entry_start_y,
        geometry.parking_entry_end_y,
        line_h,
        line_w,
        scene_config.white_color,
        scene_config.parking_entry_dash_length,
        scene_config.parking_entry_gap_length,
    )

    _spawn_transformed_line_segment(
        spline_lines,
        scene_config,
        origin_x,
        origin_y,
        origin_z,
        tx,
        ty,
        lateral_x,
        lateral_y,
        [geometry.spot_right_x, geometry.parking_entry_start_y, line_h, line_w],
        [geometry.spot_right_x, geometry.parking_entry_end_y, line_h, line_w],
        scene_config.white_color,
    )

    divider_ys = [geometry.parking_entry_start_y]
    divider_ys.extend(spot.bottom_y for spot in geometry.parking_spots)
    for divider_y in divider_ys:
        _spawn_transformed_line_segment(
            spline_lines,
            scene_config,
            origin_x,
            origin_y,
            origin_z,
            tx,
            ty,
            lateral_x,
            lateral_y,
            [geometry.parking_lane_edge_x, divider_y, line_h, line_w],
            [geometry.spot_right_x, divider_y, line_h, line_w],
            scene_config.white_color,
        )

    for divider_x in geometry.lane_boundary_xs:
        _spawn_vertical_dashed_line(
            spline_lines,
            scene_config,
            origin_x,
            origin_y,
            origin_z,
            tx,
            ty,
            lateral_x,
            lateral_y,
            divider_x,
            geometry.road_start_y,
            geometry.road_end_y,
            line_h,
            line_w,
            scene_config.white_color,
            scene_config.lane_dash_length,
            scene_config.lane_gap_length,
        )

    for divider_x in (
        geometry.direction_divider_x - scene_config.divider_gap / 2.0,
        geometry.direction_divider_x + scene_config.divider_gap / 2.0,
    ):
        _spawn_transformed_line_segment(
            spline_lines,
            scene_config,
            origin_x,
            origin_y,
            origin_z,
            tx,
            ty,
            lateral_x,
            lateral_y,
            [divider_x, geometry.road_start_y, line_h, line_w],
            [divider_x, geometry.road_end_y, line_h, line_w],
            scene_config.yellow_color,
        )

    if anchor is None:
        _spawn_parking_sign(qlabs, geometry, scene_config)
    else:
        _spawn_parking_sign_transformed(qlabs, geometry, scene_config, origin_x, origin_y, origin_z, tx, ty, lateral_x, lateral_y)
    return geometry


def draw_static_2d_map(ax, config: Optional[ParkingRoadSceneConfig] = None, **overrides) -> ParkingRoadSceneGeometry:
    """绘制直路加右侧侧方停车区的静态俯视图。"""
    scene_config = resolve_scene_config(config, **overrides)
    geometry = build_scene_geometry(scene_config)

    ax.set_aspect("equal")
    ax.add_patch(
        patches.Rectangle(
            (geometry.road_left_x, geometry.road_end_y),
            scene_config.total_width,
            scene_config.total_road_length,
            facecolor=(0.18, 0.18, 0.18),
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (min(geometry.parking_lane_edge_x, geometry.spot_right_x), geometry.parking_entry_end_y),
            abs(scene_config.parking_depth),
            geometry.parking_entry_start_y - geometry.parking_entry_end_y,
            facecolor=(0.2, 0.2, 0.2),
            edgecolor="none",
            zorder=0,
        )
    )

    ax.plot([geometry.road_left_x, geometry.road_left_x], [geometry.road_start_y, geometry.road_end_y], "w-", lw=2)
    ax.plot([geometry.parking_lane_edge_x, geometry.parking_lane_edge_x], [geometry.road_start_y, geometry.parking_entry_start_y], "w-", lw=2)
    ax.plot([geometry.parking_lane_edge_x, geometry.parking_lane_edge_x], [geometry.parking_entry_end_y, geometry.road_end_y], "w-", lw=2)
    ax.plot(
        [geometry.parking_lane_edge_x, geometry.parking_lane_edge_x],
        [geometry.parking_entry_start_y, geometry.parking_entry_end_y],
        linestyle="--",
        color="w",
        lw=2,
    )
    ax.plot([geometry.spot_right_x, geometry.spot_right_x], [geometry.parking_entry_start_y, geometry.parking_entry_end_y], "w-", lw=2)

    divider_ys = [geometry.parking_entry_start_y]
    divider_ys.extend(spot.bottom_y for spot in geometry.parking_spots)
    for divider_y in divider_ys:
        ax.plot([geometry.parking_lane_edge_x, geometry.spot_right_x], [divider_y, divider_y], "w-", lw=2)

    for divider_x in geometry.lane_boundary_xs:
        ax.plot([divider_x, divider_x], [geometry.road_start_y, geometry.road_end_y], "w--", lw=1.5)

    for divider_x in (
        geometry.direction_divider_x - scene_config.divider_gap / 2.0,
        geometry.direction_divider_x + scene_config.divider_gap / 2.0,
    ):
        ax.plot([divider_x, divider_x], [geometry.road_start_y, geometry.road_end_y], color="gold", lw=1.8)

    sign = geometry.parking_sign
    if sign is not None:
        ax.add_patch(patches.Circle((sign.x, sign.y), radius=0.06, facecolor="gray", edgecolor="none", zorder=4))
        ax.add_patch(
            patches.Rectangle(
                (sign.x - sign.board_width / 2.0, sign.y - scene_config.sign_board_thickness / 2.0),
                sign.board_width,
                scene_config.sign_board_thickness,
                facecolor="dodgerblue",
                edgecolor="white",
                lw=1.0,
                zorder=5,
            )
        )
        ax.text(sign.x, sign.y - 0.12, "P", color="white", ha="center", va="center", fontsize=12, fontweight="bold", zorder=6)

    padding_x = max(3.0, scene_config.parking_depth + 2.0)
    x_min = min(geometry.road_left_x, geometry.spot_right_x) - padding_x
    x_max = max(geometry.road_right_x, geometry.spot_right_x) + padding_x
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(geometry.road_end_y - 2.0, geometry.road_start_y + 2.0)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.grid(False)
    return geometry


def demo_scene_config() -> ParkingRoadSceneConfig:
    return ParkingRoadSceneConfig()


__all__ = [
    "ParkingRoadAnchor",
    "ParkingRoadSceneConfig",
    "ParkingRoadSceneGeometry",
    "ParkingSign",
    "ParkingSpot",
    "build_scene_geometry",
    "demo_scene_config",
    "draw_static_2d_map",
    "resolve_scene_config",
    "setup_map",
]


if __name__ == "__main__":
    from qvl.qlabs import QuanserInteractiveLabs
    from qvl.qcar2 import QLabsQCar2
    import pal.resources.rtmodels as rtmodels

    scene_config = demo_scene_config()
    print("正在连接 QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("【错误】无法连接到 QLabs！请确保 QLabs 已经启动。")
    else:
        print("连接成功！开始构建侧方位停车场景...")
        geometry = setup_map(qlabs, scene_config)
        print("场景物理模型构建完成！")
        print(f"道路末端锚点: {geometry.exit_anchor}")

        print("开始创建并初始化小车...")
        init_x = geometry.road_right_x - scene_config.lane_width / 2.0
        init_y = geometry.parking_entry_start_y + 1.5
        init_yaw = math.pi / 2.0

        hqcar = QLabsQCar2(qlabs)
        hqcar.spawn_id(
            actorNumber=0,
            location=[init_x, init_y, 0.0],
            rotation=[0.0, 0.0, init_yaw],
            waitForConfirmation=True,
        )
        hqcar.possess()
        QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
        print("小车生成完毕！(仅初始化状态，未开启控制回路)")

        print("正在拉起2D俯视图...（关闭 Matplotlib 绘图窗口即可退出运行）")
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 9))
        draw_static_2d_map(ax, scene_config)

        car_length = 3.6
        wheelbase = 2.4
        car_width = 1.8
        car_body_scale = 1.0

        line_car, = ax.plot([], [], "b-", lw=3, zorder=7, alpha=0.55)
        line_front, = ax.plot([], [], "r-", lw=2, zorder=8)
        wheel_lf, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_rf, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_lr, = ax.plot([], [], "go", markersize=8, zorder=10)
        wheel_rr, = ax.plot([], [], "go", markersize=8, zorder=10)
        car_center_dot, = ax.plot([], [], "ro", markersize=6, zorder=12, label="Car Center")
        ax.legend(loc="upper right")

        cos_yaw = math.cos(init_yaw)
        sin_yaw = math.sin(init_yaw)
        rear_overhang = (car_length - wheelbase) / 2.0
        front_overhang = rear_overhang
        back_length = -rear_overhang * car_body_scale
        front_length = (wheelbase + front_overhang) * car_body_scale
        half_width = (car_width * car_body_scale) / 2.0

        rect_pts = np.array([[front_length, half_width], [front_length, -half_width], [back_length, -half_width], [back_length, half_width], [front_length, half_width]])
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_pts = rect_pts @ rot_matrix.T + [init_x, init_y]
        line_car.set_data(rotated_pts[:, 0], rotated_pts[:, 1])

        front_center = np.array([[front_length, 0.0]]) @ rot_matrix.T + [init_x, init_y]
        line_front.set_data([init_x, front_center[0][0]], [init_y, front_center[0][1]])

        wheels_pts = np.array([[wheelbase, car_width / 2.0], [wheelbase, -car_width / 2.0], [0.0, car_width / 2.0], [0.0, -car_width / 2.0]])
        rot_wheels = wheels_pts @ rot_matrix.T + [init_x, init_y]
        wheel_lf.set_data([rot_wheels[0, 0]], [rot_wheels[0, 1]])
        wheel_rf.set_data([rot_wheels[1, 0]], [rot_wheels[1, 1]])
        wheel_lr.set_data([rot_wheels[2, 0]], [rot_wheels[2, 1]])
        wheel_rr.set_data([rot_wheels[3, 0]], [rot_wheels[3, 1]])

        car_geo_center = np.array([[(front_length + back_length) / 2.0, 0.0]]) @ rot_matrix.T + [init_x, init_y]
        car_center_dot.set_data([car_geo_center[0][0]], [car_geo_center[0][1]])

        ax.set_title(f"Parking Scene Preview ({scene_config.lanes_per_direction} lanes/dir, total {scene_config.resolved_total_lanes} lanes, {scene_config.total_road_length:.1f} m)")
        plt.show(block=True)
        qlabs.close()
