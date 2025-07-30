import numpy as np

def _get_circle_intersections(p0, r0, p1, r1):
    """
    Calculates the intersection points of two circles in a 2D plane.

    This function determines where two circles intersect based on their center points
    and radii. It handles cases where the circles do not intersect, are tangent,
    or intersect at two distinct points.

    Args:
        p0 (np.ndarray): A NumPy array of shape (2,) representing the [x, y]
                         coordinates of the center of the first circle.
        r0 (float): The radius of the first circle.
        p1 (np.ndarray): A NumPy array of shape (2,) representing the [x, y]
                         coordinates of the center of the second circle.
        r1 (float): The radius of the second circle.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: A tuple containing two NumPy
        arrays, where each array is an [x, y] coordinate of an intersection
        point. Returns None if the circles do not intersect, if one is
        contained within the other, or if they are concentric.
    """
    # Calculate the distance between the centers of the two circles.
    d = np.linalg.norm(p0 - p1)

    # Check for non-intersection cases:
    # 1. Circles are too far apart.
    # 2. One circle is contained within the other.
    # 3. Circles are concentric (d=0).
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return None

    # 'a' is the distance from the center of the first circle (p0) to the
    # point 'p2', which lies on the line connecting the centers (p0, p1) and
    # is also on the line connecting the two intersection points.
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    
    # 'h' is half the distance between the two intersection points.
    # This forms a right-angled triangle with 'a' and 'r0'.
    h_sq = r0**2 - a**2
    if h_sq < 0:
        # This case can occur due to floating point inaccuracies when circles
        # are tangent.
        return None
    h = np.sqrt(h_sq)

    # 'p2' is the midpoint of the line segment connecting the intersection points.
    p2 = p0 + a * (p1 - p0) / d

    # Calculate the coordinates of the two intersection points by moving
    # from 'p2' by a distance 'h' perpendicular to the line connecting the centers.
    pt1 = np.array([p2[0] + h * (p1[1] - p0[1]) / d, p2[1] - h * (p1[0] - p0[0]) / d])
    pt2 = np.array([p2[0] - h * (p1[1] - p0[1]) / d, p2[1] + h * (p1[0] - p0[0]) / d])

    return (pt1, pt2)

def _get_sphere_intersections(p1, r1, p2, r2, p3, r3):
    """
    Performs trilateration to find the intersection points of three spheres.

    This function calculates the common intersection points of three spheres in 3D
    space, given their center points and radii. The method establishes a new
    coordinate system based on the sphere centers and solves for the
    intersection coordinates.

    Args:
        p1 (np.ndarray): A NumPy array of shape (3,) for the [x, y, z] center of the first sphere.
        r1 (float): The radius of the first sphere.
        p2 (np.ndarray): A NumPy array of shape (3,) for the [x, y, z] center of the second sphere.
        r2 (float): The radius of the second sphere.
        p3 (np.ndarray): A NumPy array of shape (3,) for the [x, y, z] center of the third sphere.
        r3 (float): The radius of the third sphere.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: A tuple containing two NumPy
        arrays, where each array is an [x, y, z] coordinate of an intersection
        point. These two points are reflections of each other across the plane
        defined by the three sphere centers. Returns None if a unique solution
        cannot be found (e.g., sphere centers are collinear, no intersection).
    """
    # --- Establish a new coordinate system ---
    # e_x is the unit vector from p1 to p2.
    temp1 = p2 - p1
    norm_temp1 = np.linalg.norm(temp1)
    if norm_temp1 == 0: return None # p1 and p2 are the same point
    e_x = temp1 / norm_temp1

    # 'i' is the component of the vector (p3 - p1) along e_x.
    temp2 = p3 - p1
    i = np.dot(e_x, temp2)

    # e_y is the unit vector in the direction of (p3 - p1) that is
    # perpendicular to e_x.
    temp3 = temp2 - i * e_x
    norm_temp3 = np.linalg.norm(temp3)
    if norm_temp3 == 0: return None # p1, p2, p3 are collinear
    e_y = temp3 / norm_temp3

    # e_z is the cross product of e_x and e_y, completing the orthonormal basis.
    e_z = np.cross(e_x, e_y)

    # --- Solve for the coordinates in the new system ---
    # 'd' is the distance between the centers of sphere 1 and sphere 2.
    d = np.linalg.norm(p2 - p1)
    # 'j' is the component of the vector (p3 - p1) along e_y.
    j = np.dot(e_y, temp2)
    if j == 0: return None # This shouldn't happen if they are not collinear.

    # 'x', 'y', and 'z' are the coordinates of the intersection point in the
    # new coordinate system (e_x, e_y, e_z) relative to p1.
    x = (r1**2 - r2**2 + d**2) / (2 * d)
    y = (r1**2 - r3**2 + i**2 + j**2) / (2 * j) - (i / j) * x
    
    # The z-coordinate can be positive or negative, giving two possible points.
    z_sq = r1**2 - x**2 - y**2
    if z_sq < 0: return None # No real intersection exists.

    z = np.sqrt(z_sq)

    # --- Convert coordinates back to the original system ---
    # The two intersection points are found by adding the components along the
    # basis vectors (e_x, e_y, e_z) to the origin of the new system (p1).
    pt_a = p1 + x * e_x + y * e_y + z * e_z
    pt_b = p1 + x * e_x + y * e_y - z * e_z

    return (pt_a, pt_b)

def _get_line_segment_sphere_intersections(line_p1, line_p2, sphere_center, sphere_radius):
    """
    Finds the intersection points of a line segment and a sphere.

    Args:
        line_p1 (np.ndarray): The starting point of the line segment.
        line_p2 (np.ndarray): The ending point of the line segment.
        sphere_center (np.ndarray): The center of the sphere.
        sphere_radius (float): The radius of the sphere.

    Returns:
        list: A list of 3D intersection points that lie on the line segment.
              Returns an empty list if there are no intersections on the segment.
    """
    v = line_p2 - line_p1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return [] # Line segment is a point

    v_hat = v / v_norm

    # Solve the quadratic equation for the intersection parameter 't'
    a = v_hat.dot(v_hat)  # This is 1.0
    b = 2 * v_hat.dot(line_p1 - sphere_center)
    c = (line_p1 - sphere_center).dot(line_p1 - sphere_center) - sphere_radius**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return [] # No real intersection

    # Calculate the two potential solutions for 't'
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    
    solutions = []
    # Check if the intersection points lie on the physical line segment
    if 0 <= t1 <= v_norm:
        solutions.append(line_p1 + t1 * v_hat)
    if 0 <= t2 <= v_norm:
        solutions.append(line_p1 + t2 * v_hat)
        
    return solutions