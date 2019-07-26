import os
import sys

# Add the root folder of the repository to the search path for modules
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path = [root_path] + sys.path


import numpy as np
import hebi

from time import time

from components import arm_container
from components import trajectory_time_heuristic

from util.input.keyboard import getch


class Waypoint(object):

  def __init__(self, num_modules):
    self._position = np.empty(num_modules, np.float64)
    self._velocity = np.empty(num_modules, np.float64)
    self._acceleration = np.empty(num_modules, np.float64)

  @property
  def position(self):
    return self._position
  
  @property
  def velocity(self):
    return self._velocity

  @property
  def acceleration(self):
    return self._acceleration
  

class State(object):

  def __init__(self, arm):
    self._waypoints = list()
    self._quit = False
    self._mode = 'training'
    self._arm = arm
    self._jog_direction = '0'
    self.unlock_joints()
    self._current_position = np.empty(arm.dof_count, dtype=np.float64)
    self._current_velocity = np.empty(arm.dof_count, dtype=np.float64)
    from threading import Lock
    self._mutex = Lock()
    self._cmd_pose = arm.get_FK(self._current_position)
    self._update_cmd_pose = True
    self._speed_base = 0.0
    self._teleop_target_zero = -1
    self._teleop_latest_target = np.zeros(3) # TODO to 6D pose
    self._robot_zero_pose = np.zeros((arm.dof_count, 4)) # TODO why 4

  @property
  def quit(self):
    return self._quit
  
  @property
  def mode(self):
    return self._mode
  
  @property
  def arm(self):
    return self._arm

  @property
  def current_position(self):
    return self._current_position

  @property
  def current_velocity(self):
    return self._current_velocity

  @property
  def number_of_waypoints(self):
    return len(self._waypoints)

  @property
  def locked_joints(self):
    return self._locked_joints
  
  @property
  def jog_direction(self):
    return self._jog_direction

  @property
  def teleop_target_zero(self):
    return self._teleop_target_zero

  @property
  def teleop_latest_target(self):
    return self._teleop_latest_target
  
  @property
  def robot_zero_pose(self):
    return self._robot_zero_pose
  

  def lock(self):
    self._mutex.acquire()

  def unlock(self):
    self._mutex.release()

  def unlock_joints(self):
    self._locked_joints = [None for _ in range(self._arm.dof_count)]

  def lock_joints(self, joint_numbers):
    for joint_number in joint_numbers:
      self._locked_joints[joint_number] = self._current_position[joint_number]

import rospy
from geometry_msgs.msg import PointStamped
import tf

def get_current_pose(arm, feedback):
  current_position = np.empty(arm.dof_count, dtype=np.float64)
  feedback.get_position(current_position)
  print("Got current position", current_position, '\r\n')
  return arm.get_FK(current_position)

def init_teleop(state):
  print_and_cr('Initializing teleoperation')
  rospy.init_node('hebiteleop')

  # tf
  transformListener = tf.TransformListener()
  transformListener.waitForTransform("/optitrack_natnet", "/map", rospy.Time(0),rospy.Duration(1.0))

  # Set the current robot pose to correspond to the first published teleop target (both are considered "zero")
  #state.lock()
  print_and_cr('setting zero state with lock')
  state._robot_zero_pose = state._cmd_pose
  #state.unlock()

  def callback(data):
    #rospy.loginfo('Received chobi-teleop info %f %f %f', data.point.x, data.point.y, data.point.z)
    #cur_point = transformListener.transformPoint("map", data).point
    cur_point = data.point
    #state.lock()
    if state.teleop_target_zero == -1:
      state._teleop_target_zero = cur_point
      state._mode = 'teleop'
      rospy.loginfo('Initialization of teleoperation is completed')

    state._teleop_latest_target =  cur_point
    #state.unlock()
  
  rospy.Subscriber('/M1/point', PointStamped, callback, queue_size=1)
  #rospy.spin() # does nothing but block? so we don't need it?

def construct_command(arm, feedback, velocity, dt):
  command = hebi.GroupCommand(arm.group.size)
  current_position = np.empty(arm.dof_count, dtype=np.float64)
  current_velocity = np.empty(arm.dof_count, dtype=np.float64)
  feedback.get_position(current_position)
  feedback.get_velocity(current_velocity)

  # position and velocity
  current_pose = arm.get_FK(current_position)
  velocity = velocity.reshape(-1)
  cmd_pose_xyz = current_pose[0:3,3].reshape(-1)
  cmd_pose_xyz += velocity * dt
  cmd_vel_xyz = velocity
  cmd_pose_xyz = cmd_pose_xyz.tolist()
  cmd_vel_xyz = cmd_vel_xyz.tolist()
  rospy.loginfo(cmd_pose_xyz)
  rospy.loginfo(cmd_vel_xyz)
  
  jog_cmd = arm.get_jog(cmd_pose_xyz, current_position, cmd_vel_xyz, dt)
  command.position = jog_cmd[0]
  command.velocity = jog_cmd[1]

  # effort: grav comp and spring offset
  grav_comp_effort = arm.get_grav_comp_efforts(feedback).copy()
  spring_effort = np.zeros(arm.dof_count)
  spring_offset = 4.0 - 5.0*(current_position[1] - 1.4)
  spring_effort[1] = -spring_offset
  effort = grav_comp_effort + spring_effort
  command.effort = effort

  return command

def construct_velocity(state, feedback):
  a = state.teleop_latest_target
  b = state.teleop_target_zero
  target_pose = np.zeros_like(state.robot_zero_pose)
  target_pose[0:3,3] = np.array([a.x-b.x, a.y-b.y, a.z-b.z]).reshape(3,1)
  delta_pose = target_pose + state.robot_zero_pose - get_current_pose(state.arm, feedback)
  delta_pose = delta_pose[0:3,3].reshape(-1)
  np.clip(delta_pose, -0.05, 0.05, out=delta_pose)
  rospy.loginfo(delta_pose)
  return delta_pose


def command_proc(state):
  group = state.arm.group
  group.feedback_frequency = 100.0

  num_modules = group.size

  command = hebi.GroupCommand(num_modules)
  feedback = hebi.GroupFeedback(num_modules)
  prev_mode = state.mode
  start_time = time()
  last_time = start_time
  trajectory = None


  while True:
    if group.get_next_feedback(reuse_fbk=feedback) is None:
      print_and_cr('Did not receive feedback')
      continue

    state.lock()
    if state.quit:
      state.unlock()
      break

    feedback.get_position(state.current_position)
    feedback.get_velocity(state.current_velocity)
    grav_comp_effort = state.arm.get_grav_comp_efforts(feedback).copy()
    total_effort = np.empty(5, np.float64)
    
    spring_effort = 4.0 - 5.0*(state.current_position[1] - 1.4)
    np.add(grav_comp_effort, [0.0, -spring_effort, 0.0, 0.0, 0.0], total_effort)
    current_mode = state.mode
    
    cur_time = time()
    dt = cur_time - last_time
    last_time = cur_time

    x_speed = 0.0
    y_speed = 0.0
    z_speed = 0.0
    
    if(state._speed_base < 0.1):
        state._speed_base += 0.001
    if current_mode == 'teleop':
      rospy.loginfo('Confirm: in teleop mode')
      vel = construct_velocity(state, feedback)
      #rospy.loginfo('about to send velocity %f, %f, %f', vel[0], vel[1], vel[2])
      construct_command(state.arm, feedback, vel, dt)
      #group.send_command(command)
    if current_mode == 'operational':
      if state.jog_direction == 'x_plus':
        x_speed = state._speed_base
      elif state.jog_direction == 'y_plus':
        y_speed = state._speed_base
      elif state.jog_direction == 'z_plus':
        z_speed = state._speed_base
      elif state.jog_direction == 'x_minus':
        x_speed = -state._speed_base
      elif state.jog_direction == 'y_minus':
        y_speed = -state._speed_base
      elif state.jog_direction == 'z_minus':
        z_speed = -state._speed_base
      else:
        x_speed = 0.0
        y_speed = 0.0
        z_speed = 0.0

      next_angles = state.current_position
      next_speed = [0.0, 0.0, 0.0, 0.0, 0.0]
      state._cmd_pose[0] = state._cmd_pose[0] + x_speed*dt;
      state._cmd_pose[1] = state._cmd_pose[1] + y_speed*dt;
      state._cmd_pose[2] = state._cmd_pose[2] + z_speed*dt;
      print('cmd_pose', state._cmd_pose, '\r\n' )
      
      cmd_pose_xyz = [state._cmd_pose[0, 3], state._cmd_pose[1, 3], state._cmd_pose[2, 3]]
      cmd_vel_xyz = [x_speed, y_speed, z_speed]

      jog_cmd = state.arm.get_jog(cmd_pose_xyz, state.current_position, cmd_vel_xyz, dt)
      next_angles[0:3] = jog_cmd[0]
      next_speed[0:3] = jog_cmd[1]
      command.position = next_angles
      command.velocity = next_speed

      current_pose = state.arm.get_FK(state.current_position)
      #current_pose_xyz = np.empty(3, np.float64)
      current_pose_xyz = [current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]]

      xyz_pos_error = np.zeros(3, np.float64)
      xyz_pos_effort = np.zeros(3, np.float64)
      xyz_pos_joint_effort = np.zeros(5, np.float64)
      xyz_pos_gains = 0.0*np.ones(3, np.float64)

      np.subtract(cmd_pose_xyz, current_pose_xyz, xyz_pos_error)
      np.multiply(xyz_pos_error, xyz_pos_gains, xyz_pos_effort)

      xyz_vel_error = np.zeros(3, np.float64)
      xyz_vel_effort = np.zeros(3, np.float64)
      xyz_vel_joint_effort = np.zeros(5, np.float64)
      xyz_vel_gains = 0.0*np.ones(3, np.float64)
      xyz_vel_current = np.zeros(3, np.float64)


      np.dot(state.arm._robot.get_jacobian_end_effector(state.current_position)[0:3, 0:3], state.current_velocity[0:3], xyz_vel_current)
      np.subtract(cmd_vel_xyz, xyz_vel_current, xyz_vel_error)
      np.multiply(xyz_vel_error, xyz_vel_gains, xyz_vel_effort)
      
      xyz_effort = np.zeros(3, np.float64)
      xyz_joint_effort = np.zeros(5, np.float64)

      #np.add(xyz_pos_effort, xyz_vel_effort, xyz_effort)
      np.dot(state.arm._robot.get_jacobian_end_effector(state.current_position)[0:3, 0:3].T, xyz_effort, xyz_joint_effort[0:3])
      #np.add(total_effort, xyz_joint_effort, total_effort)
      command.effort = total_effort


    group.send_command(command)
    state.unlock()
    prev_mode = current_mode

def print_and_cr(msg):
  sys.stdout.write(msg + '\r\n')

def load_gain(group, gain_xml_fn):
  group_cmd = hebi.GroupCommand(group.size)
  group_cmd.read_gains(gain_xml_fn)
  group.send_command_with_acknowledgement(group_cmd)
  print('loaded gain from ', gain_xml_fn)

def save_gain(group, gain_xml_fn):
  group_info = group.request_info()
  if group_info is not None:
    group_info.write_gains(gain_xml_fn)
    print_and_cr('saved gain')

def run():
  arm = arm_container.create_5_dof('chopstick.hrdf')
  state = State(arm)
  load_gain(state.arm.group, 'chopstick-gains.xml')

  from threading import Thread
  cmd_thread = Thread(target=command_proc,
                      name='Command Thread',
                      args=(state,))
  cmd_thread.start()

  print_and_cr("Press 'p' to enable cartesian control (wsadjl for moving, k for stopping). Press 'g' to save gains, 't' to teleop and 'q' to quit.")

  res = getch()

  while res != 'q':
    print_and_cr('')
    state.lock()

    current_mode = state.mode

    start_speed = 0.00

    if res == 'g':
      print_and_cr('saving gains')
      save_gain(state.arm.group, 'chopstick-gains.xml')
    elif res == 'p':
      state._mode = 'operational'
    elif res == 't':
      print_and_cr('Entering teleoperation mode')
      init_teleop(state)

    if state._update_cmd_pose == True:
        state._cmd_pose = state.arm.get_FK(state.current_position)

    if state._mode == 'operational':
      if res == 'w':
        state._jog_direction = 'x_plus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 'a':
        state._jog_direction = 'y_plus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 'j':
        state._jog_direction = 'z_plus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 's':
        state._jog_direction = 'x_minus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 'd':
        state._jog_direction = 'y_minus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 'l':
        state._jog_direction = 'z_minus'
        state._speed_base = start_speed
        state._update_cmd_pose = False
      if res == 'k':
        state._jog_direction = '0'
        state._update_cmd_pose = True

    state.unlock()
    res = getch()

  state._quit = True
  print_and_cr('')


run()
