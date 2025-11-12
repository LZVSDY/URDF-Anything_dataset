import numpy as np

def rotation_matrix_to_axis(point, target_axis='x+'):
    """
    Compute the rotation matrix that rotates the given point to the specified axis.
    
    Parameters:
    point (array-like): A 3D point [x, y, z]
    target_axis (str): The target axis, one of 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'
    
    Returns:
    numpy.ndarray: The 3x3 rotation matrix.
    """
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = point / r
    if r == 0:
        return np.eye(3)  # Identity matrix if the point is the origin
    
    if target_axis in ['x+', 'x-']:
        # Rotate around z-axis to remove y component
        theta = - np.arctan2(y, x)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        Rz = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # Rotate around y-axis to remove z component
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Ry = np.array([
            [cos_phi, 0, sin_phi],
            [0, 1, 0],
            [-sin_phi, 0, cos_phi]
        ])
        
        # Combined rotation matrix
        R = Ry @ Rz
        
        if target_axis == 'x-':
            # Additional rotation to flip to negative x-axis
            R_flip = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            R = R_flip @ R
    
    elif target_axis in ['y+', 'y-']:
        # Rotate around x-axis to remove z component
        theta = np.arctan2(z, y)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ])
        
        # Rotate around z-axis to remove x component
        phi = np.arctan2(x, np.sqrt(y**2 + z**2))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Rz = np.array([
            [cos_phi, -sin_phi, 0],
            [sin_phi, cos_phi, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Rx
        
        if target_axis == 'y-':
            # Additional rotation to flip to negative y-axis
            R_flip = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            R = R_flip @ R
    
    elif target_axis in ['z+', 'z-']:
        # Rotate around y-axis to remove x component
        theta = np.arctan2(x, z)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        Ry = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        
        # Rotate around x-axis to remove y component
        phi = np.arctan2(y, np.sqrt(x**2 + z**2))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_phi, -sin_phi],
            [0, sin_phi, cos_phi]
        ])
        
        # Combined rotation matrix
        R = Rx @ Ry
        
        if target_axis == 'z-':
            # Additional rotation to flip to negative z-axis
            R_flip = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
            R = R_flip @ R
    
    else:
        raise ValueError("Invalid target_axis. Must be one of 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'")
    
    return R

def eliminate_z_component(point):
    """
    生成将点投影到XY平面的旋转矩阵（消除z分量）
    
    参数:
    point (array-like): 3D点 [x, y, z]
    
    返回:
    numpy.ndarray: 3x3旋转矩阵，旋转后z分量为0
    """
    x, y, z = point
    # 处理特殊情况：点已经在XY平面或原點
    if x == 0 and z == 0:
        return np.eye(3)
    
    # 计算绕y轴的旋转角度
    theta = np.arctan2(z, x)
    cos_theta = x / np.hypot(x, z)
    sin_theta = z / np.hypot(x, z)
    
    # 构造绕y轴的旋转矩阵
    Ry = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    return Ry

def eliminate_y_component(point):
    """
    生成将点投影到XZ平面的旋转矩阵（消除y分量）
    
    参数:
    point (array-like): 3D点 [x, y, z]
    
    返回:
    numpy.ndarray: 3x3旋转矩阵，旋转后y分量为0
    """
    x, y, z = point
    # 处理特殊情况：点已经在XZ平面或原點
    if y == 0 and z == 0:
        return np.eye(3)
    
    # 计算绕x轴的旋转角度
    theta = np.arctan2(y, z)
    cos_theta = z / np.hypot(y, z)
    sin_theta = y / np.hypot(y, z)
    
    # 构造绕x轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    return Rx

def eliminate_x_component(point):
    """
    生成将点投影到YZ平面的旋转矩阵（消除x分量）
    
    参数:
    point (array-like): 3D点 [x, y, z]
    
    返回:
    numpy.ndarray: 3x3旋转矩阵，旋转后x分量为0
    """
    x, y, z = point
    # 处理特殊情况：点已经在YZ平面或原點
    if x == 0 and y == 0:
        return np.eye(3)
    
    # 计算绕z轴的旋转角度
    theta = np.arctan2(x, y)
    cos_theta = y / np.hypot(x, y)
    sin_theta = x / np.hypot(x, y)
    
    # 构造绕z轴的旋转矩阵
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return Rz