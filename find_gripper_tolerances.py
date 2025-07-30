import numpy as np
from scipy.spatial.transform import Rotation as R
from arm_ik_model import RobotKinematics, Ring

def sample_tolerance(robot, ring, axis_vec, mode, max_range=300, angle_range_deg=90, n_search=100):
    """
    Samples translation or rotation along/about a given axis and returns the tolerance range.
    mode: 'trans' or 'rot'
    axis_vec: axis in world coordinates
    """
    # Store the original pose to always start from it
    orig_pos = ring.position.copy()
    orig_approach = ring.approach_vec.copy()
    orig_tangential = ring.tangential_vec.copy()
    orig_radial = ring.radial_vec.copy()

    def test_value(v):
        if mode == 'trans':
            new_pos = orig_pos + v * axis_vec
            test_ring = Ring(ring.outer_radius, ring.middle_radius, ring.inner_radius, ring.outer_thickness, ring.inner_thickness,
                             new_pos, orig_approach, orig_tangential, orig_radial)
        else:
            rot = R.from_rotvec(np.deg2rad(v) * axis_vec)
            new_approach = rot.apply(orig_approach)
            new_tangential = rot.apply(orig_tangential)
            new_radial = rot.apply(orig_radial)
            test_ring = Ring(ring.outer_radius, ring.middle_radius, ring.inner_radius, ring.outer_thickness, ring.inner_thickness,
                             orig_pos, new_approach, new_tangential, new_radial)
        return robot.evaluate_grip(test_ring)

    # Binary search for positive limit
    low, high = 0.0, max_range if mode == 'trans' else angle_range_deg
    pos_limit = 0.0
    for _ in range(n_search):
        mid = (low + high) / 2
        if test_value(mid):
            pos_limit = mid
            low = mid
        else:
            high = mid
    # Binary search for negative limit
    low, high = -max_range if mode == 'trans' else -angle_range_deg, 0.0
    neg_limit = 0.0
    for _ in range(n_search):
        mid = (low + high) / 2
        if test_value(mid):
            neg_limit = mid
            high = mid
        else:
            low = mid
    return neg_limit, pos_limit

def main():
    robot = RobotKinematics(verbosity=0)
    # Solve for a valid pose
    target_e1_pos = np.array([0, 350, 0])
    target_rx_deg = 10
    target_rz_deg = -15
    success = robot.update_from_e1_pose(target_e1_pos, target_rx_deg, target_rz_deg)
    if not success:
        print("Could not solve robot pose.")
        return
    ring = robot.create_ring()
    print("\nSampling gripper tolerances...")
    axes = [
        (ring.approach_vec, 'approach'),
        (ring.radial_vec, 'radial'),
        (ring.tangential_vec, 'tangential')
    ]
    results = {}
    for axis, name in axes:
        neg, pos = sample_tolerance(robot, ring, axis, 'trans')
        results[f'trans_{name}'] = (neg, pos)
        print(f"Translation along {name}: {neg:.2f} mm to {pos:.2f} mm")
    for axis, name in axes:
        neg, pos = sample_tolerance(robot, ring, axis, 'rot', angle_range_deg=10.0)
        results[f'rot_{name}'] = (neg, pos)
        print(f"Rotation about {name}: {neg:.2f} deg to {pos:.2f} deg")
    print("\nSummary of gripper tolerances:")
    for k, (neg, pos) in results.items():
        print(f"{k}: {neg:.2f} to {pos:.2f}")

if __name__ == "__main__":
    main()
