from dataclasses import dataclass


@dataclass
class pose_request:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    timestamp: float = 0.0

@dataclass
class joy_buttons:
    button_a: int
    button_b: int
    button_x: int
    button_y: int
    trigger_left: float
    trigger_right: float