"""Idealized pro-swimmer pose definitions for SwimVision ghost overlay."""

import numpy as np

# Idealized Track Start Pose (normalized relative to hip midpoint)
# These are heuristic offsets for a 'perfect' block setup
PRO_BLOCK_POSE = {
    11: [-0.15, -0.25], # Left Shoulder
    12: [-0.15, -0.25], # Right Shoulder
    23: [0.0, 0.0],     # Left Hip
    24: [0.0, 0.0],     # Right Hip
    25: [0.05, 0.35],   # Left Knee (Front)
    26: [-0.1, 0.2],    # Right Knee (Rear)
    27: [0.1, 0.6],     # Left Ankle
    28: [-0.2, 0.5],    # Right Ankle
    15: [0.2, 0.4],     # Left Wrist (Grabbing block)
    16: [0.2, 0.4],     # Right Wrist
}

# Idealized Flight Pose (Streamline)
PRO_FLIGHT_POSE = {
    11: [0.4, 0.0],     # Shoulder
    12: [0.4, 0.0],
    23: [0.0, 0.0],     # Hip
    24: [0.0, 0.0],
    27: [-0.5, 0.0],    # Ankle
    28: [-0.5, 0.0],
    15: [0.8, 0.0],     # Wrist
    16: [0.8, 0.0],
}
