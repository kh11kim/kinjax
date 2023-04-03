import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import numpy as np
import sympy as sym
import jax
import jax.numpy as jnp
from typing import List, Tuple

def get_link_joint_dict(urdf_file):
    """Extracts link and joint information from a URDF file and returns them as dictionaries.

    Args:
        urdf_file (str): The file path of the URDF file.

    Returns:
        tuple: A tuple of two dictionaries. The first dictionary contains link information with
        each link's name as the key and a dictionary of its parent joint's name (if exists) as the
        value. The second dictionary contains joint information with each joint's name as the key
        and a dictionary of its type, parent link, child link, origin XYZ, origin RPY, and axis (if
        the joint is not fixed) as the value.

    Example:
        >>> link_dict, joint_dict = get_link_joint_dict("robot.urdf")
        >>> print(link_dict["link_name"])
        {'parent_joint': 'joint_name'}
        >>> print(joint_dict["joint_name"])
        {'joint_type': 'revolute', 'parent': 'parent_link', 'child': 'child_link', 'origin_xyz':
        [0.0, 0.0, 0.1], 'origin_rpy': [0.0, 0.0, 0.0], 'axis': [array([1.]), array([0.]), array([0.])]}
    """
    def str2arr(string):
        return np.array(string.split(" ")).astype(float)
    
    # Parse the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    # Extract the robot name from the URDF file
    robot_name = root.attrib["name"]
    print("Robot name:", robot_name)

    # Extract the joint information from the URDF file
    link_dict = {}
    for link in root.iter("link"):
        link_name = link.attrib["name"]
        link_dict[link_name] = {}

    joint_dict = {}
    for joint in root.iter("joint"):
        joint_name = joint.attrib["name"]
        joint_origin_xyz = str2arr(joint.find("origin").attrib["xyz"])
        joint_origin_rpy = str2arr(joint.find("origin").attrib["rpy"])
        joint_type = joint.attrib["type"]
        joint_parent = joint.find("parent").attrib["link"]
        joint_child = joint.find("child").attrib["link"]
        joint_dict[joint_name] = {
            "joint_type": joint_type,
            "parent": joint_parent,
            "child": joint_child,
            "origin_xyz": joint_origin_xyz,
            "origin_rpy": joint_origin_rpy
        }
        link_dict[joint_parent]["joint"] = joint_name
        link_dict[joint_child]["parent_joint"] = joint_name
        if joint_type != "fixed":
            joint_axis = [np.array(i.split(" ")).astype(float) for i in joint.find("axis").attrib.values()]
            joint_dict[joint_name]["axis"] = joint_axis
    return link_dict, joint_dict

def get_sym_matrix(rpy=[0,0,0], trans=[0,0,0]):
    """Creates a 4x4 homogeneous transformation matrix using the given RPY angles and translation.

    Args:
        rpy (list, optional): A list of three floats representing roll, pitch, and yaw angles in radians,
        in the order of zyx. Defaults to [0,0,0].
        trans (list, optional): A list of three floats representing the translation in x, y, and z
        directions. Defaults to [0,0,0].

    Returns:
        sympy.Matrix: A 4x4 homogeneous transformation matrix as a SymPy Matrix object.
    """
    rot_mat = Rotation.from_euler("zyx", rpy[::-1]).as_matrix()
    rot_mat[np.abs(rot_mat) < 1e-9] = 0.
    T = np.block([[rot_mat, np.array(trans)[:,None]], [0,0,0,1]])
    return sym.Matrix(T)

def yaw_rot_mat(q):
    """Creates a 4x4 rotation matrix for a rotation about the z-axis by the given angle.

    Args:
        q (float or sympy.Symbol): The angle of rotation about the z-axis in radians.

    Returns:
        sympy.Matrix: A 4x4 rotation matrix as a SymPy Matrix object.
    """
    s, c = sym.sin(q), sym.cos(q)
    R = sym.Matrix([[c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return R

def get_T_offset_lists(
    link_dict: dict, joint_dict: dict, ee_link_name: str
) -> Tuple[List[sym.Matrix], List[sym.Matrix]]:
    """Computes the list of transformation matrices between the base and end effector links.

    Args:
        link_dict (dict): A dictionary containing information about each link in the robot, with the name of the link as the key.
        joint_dict (dict): A dictionary containing information about each joint in the robot, with the name of the joint as the key.
        ee_link_name (str): The name of the end effector link.

    Returns:
        Tuple[List[sympy.Matrix], List[sympy.Matrix]]: A tuple containing a list of the transformation matrices between the base link and each joint's
        parent link, and an empty list.
    """

    T_offset_list = []
    
    # Compute the transformation matrices between the base link and the end effector link
    link_name = ee_link_name 
    while True:
        if "parent_joint" not in link_dict[link_name]:  
            break
        parent_joint_name = link_dict[link_name]["parent_joint"]
        parent_joint = joint_dict[parent_joint_name]
        T = get_sym_matrix(parent_joint["origin_rpy"], parent_joint["origin_xyz"])
        T_offset_list.append(T)
        link_name = joint_dict[parent_joint_name]["parent"]
    T_offset_list = T_offset_list[::-1]
   
    return T_offset_list

def get_tf_between_links(T_offset_list, qs, link_from, link_to):
    """Computes the transformation matrix between two links given a list of offset matrices and joint angles.

    Args:
        T_offset_list (list): A list of SymPy matrices representing the offset transformation matrices between each joint and its parent link.
        qs (list): A list of SymPy symbols representing the joint variables.
        link_from (int): The index of the starting link in the list of offset matrices and joint angles.
        link_to (int): The index of the ending link in the list of offset matrices and joint angles.

    Returns:
        sympy.Matrix: A 4x4 homogeneous transformation matrix as a SymPy Matrix object.
    """
    assert link_to <= len(T_offset_list)
    T = sym.Matrix(np.eye(4))
    
    # Compute the transformation matrix between the starting and ending links
    for i in range(link_from, link_to):
        q = 0.
        if len(qs) > i:
            q = qs[i]
        T = T @ T_offset_list[i] @ yaw_rot_mat(q)
    return T

def get_FK_fn(
    link_dict:dict, 
    joint_dict:dict, 
    dof: int, 
    ee_link_name: str, 
    batch=False
):
    """Returns a function for computing the forward kinematics of a robot.

    Args:
        link_dict (dict): A dictionary containing information about each link in the robot, with the name of the link as the key.
        joint_dict (dict): A dictionary containing information about each joint in the robot, with the name of the joint as the key.
        dof (int): The number of degrees of freedom of the robot.
        ee_link_name (str): The name of the end effector link.
        batch (bool): Whether to use JAX's vmap to allow for batched inputs. Default is False.

    Returns:
        callable: A function that takes a tuple of joint angles as input and returns the 4x4 homogeneous transformation matrix from the base link to the end effector link.

    Example:
        >>> link_dict = {"base_link": {"joint": "joint_1"},
                         "link_1": {"parent_joint": "joint_1"},
                         "link_2": {"parent_joint": "joint_2"},
                         "end_effector_link": {"parent_joint": "joint_3"}}
        >>> joint_dict = {"joint_1": {"origin_xyz": [1,0,0], "origin_rpy": [0,0,0], "child": "link_1", "parent": "base_link"},
                          "joint_2": {"origin_xyz": [0,1,0], "origin_rpy": [0,0,0], "child": "link_2", "parent": "link_1"},
                          "joint_3": {"origin_xyz": [0,0,1], "origin_rpy": [0,0,0], "child": "end_effector_link", "parent": "link_2"}}
        >>> fk_fn = get_fk_fn(link_dict, joint_dict, 3, "end_effector_link", batch=False)
        >>> q = (0.1, 0.2, 0.3)
        >>> T = fk_fn(q)
        >>> print(T)
        [[ 1.  0.  0.  1.]
         [ 0.  1.  0.  1.]
         [ 0.  0.  1.  1.]
         [ 0.  0.  0.  1.]]
    """
    # Define symbolic joint angle variables and compute list of offset matrices
    qs = sym.symbols(" ".join([f"q{num}" for num in range(1, dof+1)]))
    T_offset_list = get_T_offset_lists(link_dict, joint_dict, ee_link_name)
    # Compute the full transformation matrix from the base link to the end effector link
    T_fk = get_tf_between_links(T_offset_list, qs, 0, len(T_offset_list))
    
    # Create a JAX-compiled function that takes joint angle inputs and returns the transformation matrix
    _fk_fn = sym.lambdify(qs, T_fk, "jax", cse=True)
    fk_fn = lambda q: _fk_fn(*q)
    if not batch:
        return jax.jit(fk_fn)
    else:
        return jax.jit(jax.vmap(fk_fn))

def get_jacobian_fn(
    link_dict:dict, 
    joint_dict:dict, 
    dof: int, 
    ee_link_name: str, 
    batch=False
):
    """Returns a function for computing the geometric Jacobian of a robot.

    Args:
        link_dict (dict): A dictionary containing information about each link in the robot, with the name of the link as the key.
        joint_dict (dict): A dictionary containing information about each joint in the robot, with the name of the joint as the key.
        dof (int): The number of degrees of freedom of the robot.
        ee_link_name (str): The name of the end effector link.
        batch (bool): Whether to use JAX's vmap to allow for batched inputs. Default is False.

    Returns:
        callable: A function that takes a tuple of joint angles as input and returns the geometric Jacobian matrix of the robot.
    """
    # Define symbolic joint angle variables and compute list of offset matrices
    qs = sym.symbols(" ".join([f"q{num}" for num in range(1, dof+1)]))
    T_offset_list = get_T_offset_lists(link_dict, joint_dict, ee_link_name)
    ee_idx = len(T_offset_list)

    pos_jac = []
    rot_jac = []
    for joint in range(1, 8):
        T_link_wrt_base = get_tf_between_links(T_offset_list, qs, 0, joint)
        T_ee_wrt_link = get_tf_between_links(T_offset_list, qs, joint, ee_idx)
        vec_ee_from_joint_wrt_world = T_link_wrt_base[:3, :3] @ T_ee_wrt_link[:3, -1]
        rot_axis = T_link_wrt_base[:3, 2]
        lin_vel_by_joint = rot_axis.cross(vec_ee_from_joint_wrt_world)
        pos_jac.append(lin_vel_by_joint)
        rot_jac.append(rot_axis)
    pos_jac = sym.Matrix(sym.BlockMatrix([pos_jac]))
    rot_jac = sym.Matrix(sym.BlockMatrix([rot_jac]))
    jac = sym.Matrix(sym.BlockMatrix([[pos_jac],[rot_jac]]))
    _jac_fn = sym.lambdify(qs, jac, "jax", cse=True)
    jac_fn = lambda q: _jac_fn(*q)
    if not batch:
        return jax.jit(jac_fn)
    else:
        return jax.jit(jax.vmap(jac_fn))