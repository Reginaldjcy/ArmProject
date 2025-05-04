

def point3_to_robot(points, x_prop=1, y_prop=1, z_prop=1):
    x, y, z = points
    robot_point = [x*x_prop, y*y_prop, z*z_prop]

    return robot_point
