import numpy as np
import hebi


class ArmContainer(object):

  def __init__(self, group, model_full, model_ee):
    self._group = group
    self._robot_full = model_full
    self._robot_ee = model_ee
    self._masses = model_full.masses

  @property
  def dof_count(self):
    return self._robot_full.dof_count

  @property
  def dof_count_ee(self):
    return self._robot_ee.dof_count

  @property
  def group(self):
    return self._group

  @property
  def robot_full(self):
    return self._robot_full

  @property
  def robot_ee(self):
    return self._robot_ee

  def get_jog_xyz(self, positions, cmd_pose, cmd_vel, dt):
    robot = self._robot_ee
    dof = robot.dof_count
    cmd_pose_xyz = np.array([cmd_pose[0, 3], cmd_pose[1, 3], cmd_pose[2, 3]])
    xyz_objective = hebi.robot_model.endeffector_position_objective(cmd_pose_xyz)
    print(xyz_objective, positions[:dof])
    new_arm_joint_angs = robot.solve_inverse_kinematics(positions[:dof], xyz_objective)
    # Find the determinant of the jacobian at the endeffector of the solution
    # to the IK. If below a set threshold, set the joint velocities to zero
    # in an attempt to avoid nearing the kinematic singularity. 
    jacobian_new = robot.get_jacobian_end_effector(new_arm_joint_angs)[0:3, 0:3]
    det_J_new = abs(np.linalg.det(jacobian_new))
    joint_velocities = [0.0, 0.0, 0.0]

    if (det_J_new >= 0.01):
      try:
        cmd_vel_xyz = cmd_vel[0:3]
        joint_velocities = np.linalg.solve(jacobian_new, cmd_vel_xyz)
      except np.linalg.LinAlgError as lin:
        # This may happen still sometimes
        print('No solution found in IK. \n\n\n')

    return new_arm_joint_angs[0:3] , joint_velocities
  
  def get_jog(self, cmd_pose, positions, cmd_vel, dt):

    robot = self._robot_ee
    dof = robot.dof_count

    cmd_pose_xyz = np.array([cmd_pose[0, 3], cmd_pose[1, 3], cmd_pose[2, 3]])
    xyz_objective = hebi.robot_model.endeffector_position_objective(cmd_pose_xyz)
    orientation = cmd_pose[0:3, 0:3]
    theta_objective = hebi.robot_model.endeffector_so3_objective(orientation, weight = 0.6)
    new_arm_joint_angs = robot.solve_inverse_kinematics(positions[:dof], xyz_objective, theta_objective)

    jacobian_new = robot.get_jacobian_end_effector(new_arm_joint_angs)

    try:
      joint_velocities = np.linalg.pinv(jacobian_new) * np.array(cmd_vel).reshape(6,1);
  #    self._joint_angles[0:3, 0] = new_arm_joint_angs[0:3].reshape((3, 1))
  #    np.copyto(self._grip_pos, self._new_grip_pos)
    except np.linalg.LinAlgError as lin:
  #    # This may happen still sometimes
      print('No solution found. \n\n\n')
      new_arm_joint_angs = positions
      joint_velocities = np.zeros(dof, np.float64)

    # wrist_vel = self._direction*self._user_commanded_wrist_velocity
    # self._joint_velocities[3, 0] = self._joint_velocities[1, 0]+self._joint_velocities[2, 0]+wrist_vel
    # self._joint_angles[3, 0] = self._joint_angles[3, 0]+(self._joint_velocities[3, 0]*dt)
    return new_arm_joint_angs[:dof] , joint_velocities

  def get_FK_ee(self, positions):
    return self._robot_ee.get_end_effector(positions[:self._robot_ee.dof_count])

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

    num_dof = self._robot_full.dof_count
    num_frames = self._robot_full.get_frame_count('CoM')
    jacobians = self._robot_full.get_jacobians('CoM', feedback.position)
    masses = self._masses

    comp_torque = output or np.asmatrix(np.zeros((num_dof, 1), dtype=np.float64))
    wrench_vec = np.zeros((6, 1), dtype=np.float64)

    for i in range(num_frames):
      # Set translational part
      for j in range(3):
        wrench_vec[j, 0] = - gravity[j] * masses[i]
      comp_torque += jacobians[i].T * wrench_vec

    return comp_torque.A1

def create_robot(hrdf_filename):
  lookup = hebi.Lookup()

  # You can modify the names here to match modules found on your network
  module_family = 'feeding'
  module_names = ['0.base', '1.shoulder', '2.elbow', '3.wrist1', '4.wrist2', '9.chop']

  from time import sleep
  sleep(1)
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

  # For grav comp. use model_full. For computing IK, use model_ee which exclues chopstick actuator
  model_ee = hebi.robot_model.import_from_hrdf(hrdf_filename)
  model_full  = hebi.robot_model.import_from_hrdf(hrdf_filename)
  model_full.add_bracket('X5-LightBracket', 'left')
  model_full.add_actuator('X5-1')

  assert arm.size == model_full.dof_count
  return ArmContainer(arm, model_full, model_ee)

