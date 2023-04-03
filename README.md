# kinjax
---------------
**kinjax** is a Python package that provides functions for computing the forward kinematics and Jacobian of a robot. It uses JAX to enable fast computations on GPUs.

### Installation
To install the module, run:
```pip install kinjax```

### Usage
To use kinjax, you need to provide it with a URDF file that describes the robot you want to compute the forward kinematics and Jacobian for. You can then call `get_fk_fn` and `get_jacobian_fn` to create functions that can compute the forward kinematics and Jacobian of the robot, respectively.

Here's an example of how to use `kinjax` to compute the forward kinematics and Jacobian of a robot:

```python
import jax
import jax.numpy as jnp
import kinjax

end_effector_link_name = "hand"
dof = 6
# Load the URDF file
link_dict, joint_dict = kinjax.get_link_joint_dict("robot.urdf")

# Compute the forward kinematics function for the robot
fk_fn = kinjax.get_fk_fn(link_dict, joint_dict, dof, end_effector_link_name)

# Compute the Jacobian function for the robot
jac_fn = kinjax.get_jacobian_fn(link_dict, joint_dict, dof, end_effector_link_name)

# Compute the forward kinematics and Jacobian for a given joint configuration
q = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
T_fk = fk_fn(q)
J = jac_fn(q) 
```

### License
This project is licensed under the MIT License