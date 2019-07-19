import numpy as np
import hebi


class ArmContainer(object):

  def __init__(self, group, robot_model):
    self._group = group
    self._robot = robot_model
    self._masses = robot_model.masses

  @property
  def dof_count(self):
    return self._robot.dof_count

  @property
  def group(self):
    return self._group

  @property
  def robot(self):
    return self._robot
  
  def get_jog(self, cur_pose, positions, cmd_vel, dt):

    robot = self._robot
    
    cur_pose[0] = cur_pose[0] + cmd_vel[0]*dt;
    cur_pose[1] = cur_pose[1] + cmd_vel[1]*dt;
    cur_pose[2] = cur_pose[2] + cmd_vel[2]*dt;
    xyz_objective = hebi.robot_model.endeffector_position_objective(cur_pose)
    new_arm_joint_angs = robot.solve_inverse_kinematics(positions, xyz_objective)
    # Find the determinant of the jacobian at the endeffector of the solution
    # to the IK. If below a set threshold, set the joint velocities to zero
    # in an attempt to avoid nearing the kinematic singularity. 
    jacobian_new = robot.get_jacobian_end_effector(new_arm_joint_angs)[0:3, 0:3]
    det_J_new = abs(np.linalg.det(jacobian_new))

    joint_velocities = np.empty(3, np.float64)
    joint_velocities = [0.0, 0.0, 0.0]

    if (det_J_new < 0.01):
      # Near singularity - don't command arm towards it
      joint_velocities = [0.0, 0.0, 0.0]
    else:
      try:
        joint_velocities = np.linalg.solve(jacobian_new, cmd_vel)
    #    self._joint_angles[0:3, 0] = new_arm_joint_angs[0:3].reshape((3, 1))
    #    np.copyto(self._grip_pos, self._new_grip_pos)
      except np.linalg.LinAlgError as lin:
    #    # This may happen still sometimes
        joint_velocities = [0.0, 0.0, 0.0]

    # wrist_vel = self._direction*self._user_commanded_wrist_velocity
    # self._joint_velocities[3, 0] = self._joint_velocities[1, 0]+self._joint_velocities[2, 0]+wrist_vel
    # self._joint_angles[3, 0] = self._joint_angles[3, 0]+(self._joint_velocities[3, 0]*dt)
    return new_arm_joint_angs[0:3] , joint_velocities

  def get_FK(self, positions):
    robot = self._robot
    return robot.get_end_effector(positions)

  def get_grav_comp_efforts(self, feedback, output=None):
    """
    Gets the torques which approximately balance out the effect
    of gravity on the arm
    """
    #print('accelerometer', feedback[0].accelerometer, '\nnext_angles')
    gravity = -1.0*feedback[0].accelerometer
    #print('gravity', gravity, '\n')

    #gravity = np.array([0,0,-1]) # Assume the gravity points down from the first module

    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm > 0.0:
      gravity  = gravity / gravity_norm * 9.81

    num_dof = self._robot.dof_count
    num_frames = self._robot.get_frame_count('CoM')
    jacobians = self._robot.get_jacobians('CoM', feedback.position)
    masses = self._masses

    comp_torque = output or np.asmatrix(np.zeros((num_dof, 1), dtype=np.float64))
    wrench_vec = np.zeros((6, 1), dtype=np.float64)

    for i in range(num_frames):
      # Set translational part
      for j in range(3):
        wrench_vec[j, 0] = - gravity[j] * masses[i]
      comp_torque += jacobians[i].T * wrench_vec

    return comp_torque.A1


def create_3_dof():
  lookup = hebi.Lookup()

  # You can modify the names here to match modules found on your network
  module_family = 'HEBI'
  module_names = ['base', 'shoulder', 'elbow']

  from time import sleep
  sleep(2)
  arm = lookup.get_group_from_names([module_family], module_names)

  if arm is None:
    print('\nCould not find arm group: Did you forget to set the module family and names?')
    print('Searched for modules named:')
    print("{0} with family '{1}'".format(
      ', '.join(["'{0}'".format(entry) for entry in module_names]), module_family))

    print('Modules on the network:')
    for entry in lookup.entrylist:
      print(entry)
    else:
      print('[No Modules Found]')
    exit(1)

  model = hebi.robot_model.RobotModel()
  model.add_actuator('X5-4')
  model.add_bracket('X5-LightBracket', 'right')
  model.add_actuator('X5-4')
  model.add_link('X5', extension=0.18, twist=np.pi)
  model.add_actuator('X5-4')
  model.add_link('X5', extension=0.18, twist=0)

  assert arm.size == model.dof_count
  return ArmContainer(arm, model)


def create_5_dof(hrdf_filename):
  lookup = hebi.Lookup()

  # You can modify the names here to match modules found on your network
  module_family = 'feeding'
  module_names = ['0.base', '1.shoulder', '2.elbow','3.wrist1','4.wrist2']

  from time import sleep
  sleep(2)
  arm = lookup.get_group_from_names([module_family], module_names)

  if arm is None:
    print('\nCould not find arm group: Did you forget to set the module family and names?')
    print('Searched for modules named:')
    print("{0} with family '{1}'".format(
      ', '.join(["'{0}'".format(entry) for entry in module_names]), module_family))

    print('Modules on the network:')
    for entry in lookup.entrylist:
      print(entry)
    else:
      print('[No Modules Found]')
    exit(1)

  model = hebi.robot_model.import_from_hrdf(hrdf_filename)

  assert arm.size == model.dof_count
  return ArmContainer(arm, model)

