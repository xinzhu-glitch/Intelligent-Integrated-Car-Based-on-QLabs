import time

from qvl.qlabs import QuanserInteractiveLabs

from intersection_scene import IntersectionConfig, build_intersection_scene
from paralleling_parking import ParkingRoadSceneConfig, setup_map as build_parking_scene
from road_scene import RoadSceneConfig, build_road_scene, safe_cleanup


LANES_PER_DIRECTION = 3
LANE_WIDTH = 3.5


def _build_right_turn_exit_anchor(road_anchor, intersection_result, intersection_config):
    center_x, center_y, center_z = intersection_result.center
    intersection_half = intersection_config.resolved_intersection_length / 2.0
    exit_tx = -road_anchor.px
    exit_ty = -road_anchor.py
    exit_px = road_anchor.tx
    exit_py = road_anchor.ty
    exit_distance = intersection_half + intersection_config.side_exit_length
    return {
        "x": center_x + exit_tx * exit_distance,
        "y": center_y + exit_ty * exit_distance,
        "z": center_z,
        "tx": exit_tx,
        "ty": exit_ty,
        "px": exit_px,
        "py": exit_py,
    }


def run_combined_scene() -> None:
    """总入口：组合公路场景与十字路口场景。"""
    road_config = RoadSceneConfig(
        lanes_per_direction=LANES_PER_DIRECTION,
        lane_width=LANE_WIDTH,
        terminal_marking_trim_end=4.0,
    )
    intersection_config = IntersectionConfig(
        lanes_per_direction=LANES_PER_DIRECTION,
        lane_width=LANE_WIDTH,
        intersection_length=None,
        main_exit_length=36.0,
        side_exit_length=32.0,
        include_incoming_arm=False,
        crosswalk_clear_length=4.0,
        traffic_light_cycle=IntersectionConfig().traffic_light_cycle,
    )
    parking_config = ParkingRoadSceneConfig(
        lanes_per_direction=LANES_PER_DIRECTION,
        lane_width=LANE_WIDTH,
    )
    update_step = 0.1

    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("❌ QLabs 连接失败，请先打开 QLabs 软件")
        return

    try:
        safe_cleanup(qlabs)
        road_result = build_road_scene(qlabs, config=road_config, cleanup_first=False)
        intersection_result = build_intersection_scene(
            qlabs,
            config=intersection_config,
            anchor=road_result.anchor,
            cleanup_first=False,
        )
        right_turn_exit_anchor = _build_right_turn_exit_anchor(
            road_result.anchor,
            intersection_result,
            intersection_config,
        )
        build_parking_scene(
            qlabs,
            config=parking_config,
            clear_existing=False,
            anchor=right_turn_exit_anchor,
        )

        print("组合场景已启动，红绿灯将持续切换，按 Ctrl+C 结束。")
        while True:
            intersection_result.update_traffic_lights(current_time=time.time())
            time.sleep(update_step)
    except KeyboardInterrupt:
        print("\n已手动停止组合场景。")
    finally:
        qlabs.close()


if __name__ == "__main__":
    run_combined_scene()