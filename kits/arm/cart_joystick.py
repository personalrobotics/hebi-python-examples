from __future__ import print_function

import os
import sys
import numpy as np
import math
from time import time, sleep

# Ros
import rospy
from geometry_msgs.msg import PointStamped

# HEBI
import hebi
from threading import Lock
from threading import Thread

# Local
# Add the root folder of the repository to the search path for modules
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path = [root_path] + sys.path
from components import arm_container
from util.input.keyboard import getch

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if np.isclose(R[2,0], -1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif np.isclose(R[2,0], 1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return 180*psi/math.pi, 180*theta/math.pi, 180*phi/math.pi

class State(object):

  def __init__(self, arm):
    self._quit = False
    self._mode = 'training'
    self._arm = arm

    self._current_position = np.empty(arm.dof_count, dtype=np.float64)
    self._current_velocity = np.empty(arm.dof_count, dtype=np.float64)
    self._cmd_pose = arm.get_FK_ee(self._current_position) # The target pose
    self._update_cmd_pose = False

    # For jogging-like cartesian control
    self._speed_base = 0.0
    self._jog_direction = '0'

    # For teleop-based cartesian control
    self._teleop_target_zero = None
    self._teleop_latest_target = None # TODO d$ np.zeros(6) # 6D target pose
    self._robot_zero_pose = None # TODO d$ arm.get_FK_ee(self._current_position)

    # For threading safety
    self._mutex = Lock()
    self._print = False;

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

#######################################################################
# Utility scripts
#######################################################################

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

def parse_jog_xyz_speed(state):
  x_speed = 0.0
  y_speed = 0.0
  z_speed = 0.0

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

  return x_speed, y_speed, z_speed

def parse_jog_open_chopstick(state):
  o_speed = 0.0
  if state.jog_direction == 'open':
    o_speed = 0.2
  elif state.jog_direction == 'close':
    o_speed = -0.2

  return o_speed

#######################################################################
# Teleop
#######################################################################

def init_teleop(state):
  # Initialize teleoperation
  # - set the robot pose as zero
  # - set the first chopstick pose as zero
  # - continuely stream chopstick pose as target
  # ensures you have locked state outside this function scope

  print_and_cr('Initializing teleoperation')
  rospy.init_node('hebiteleop')

  # Set the current robot pose to correspond to the first published teleop target (both are considered "zero")
  print_and_cr('setting robot\'s zero state')
  state._cmd_pose = state.arm.get_FK_ee(state.current_position)
  state._robot_zero_pose = state._cmd_pose.copy()

  def callback(data):
    #rospy.loginfo('Received chobi-teleop info %f %f %f', data.point.x, data.point.y, data.point.z)
    cur_point = data.point
    state.lock()
    if state.teleop_target_zero is None:
      state._teleop_target_zero = np.array([cur_point.x, cur_point.y, cur_point.z])
      state._mode = 'teleop'
      rospy.loginfo('Initialization of teleoperation is completed')
    state._teleop_latest_target =  np.array([cur_point.x, cur_point.y, cur_point.z])
    state.unlock()
  
  rospy.Subscriber('/M1/point', PointStamped, callback, queue_size=1)

def construct_teleop_target(state, feedback, dt):
  current_pose = state.arm.get_FK_ee(state.current_position)

  target_pose = np.zeros_like(state.robot_zero_pose)
  target_pose[0:3, 3] = np.array(state.teleop_latest_target
                                 - state.teleop_target_zero).reshape(3,1)

  delta_pose = target_pose + state.robot_zero_pose - current_pose
  velocity = np.array(delta_pose[0:3,3])
  tol = 2 * 1e-4
  velocity[velocity < tol] = 0.0
  np.clip(velocity, -0.05, 0.05, out=velocity)

  target_pose = current_pose + velocity * dt
  #print(target_pose[0:3,3], state.robot_zero_pose[0:3,3], delta_pose[0:3,3], velocity)

  return target_pose, velocity


def construct_jog_target(state, velocity, dt):
  x_speed, y_speed, z_speed = velocity

  # Update the target cmd_pose
  state._cmd_pose[0,3] = state._cmd_pose[0,3] + x_speed*dt;
  state._cmd_pose[1,3] = state._cmd_pose[1,3] + y_speed*dt;
  state._cmd_pose[2,3] = state._cmd_pose[2,3] + z_speed*dt;

  target_vel_xyz = [x_speed, y_speed, z_speed, 0.0, 0.0, 0.0]

  return state._cmd_pose, target_vel_xyz

  #current_pose = state._cmd_pose
  #xyz_pose = current_pose[0:3,3] + np.array(velocity).reshape(3,1) * dt
  #cmd_pose_xyz = [xyz_pose[0,0], xyz_pose[1,0], xyz_pose[2,0]]


def construct_command(state, feedback, cmd_pose, cmd_vel, dt,
                      chopstick_angle_target=None, chopstick_angle_speed=None):
  command = hebi.GroupCommand(state.arm.group.size)

  # position and velocity
  next_angles = state.current_position
  next_speed = [0.0] * state.arm.group.size

  # XYZ
  jog_cmd = state.arm.get_jog_xyz(state.current_position, cmd_pose, cmd_vel, dt)
  dof = jog_cmd[0].shape[0]
  next_angles[:dof] = jog_cmd[0]
  next_speed[:dof] = jog_cmd[1]

  # chopstick open and close
  if chopstick_angle_speed: # For jogging
    chopstick_angle_target = next_angles[-1] + chopstick_angle_speed * dt
    if ((chopstick_angle_speed > 0 and chopstick_angle_target > -0.06) or
        (chopstick_angle_speed < 0 and chopstick_angle_target < -0.72)):
      print(chopstick_angle_speed, next_angles[-1], chopstick_angle_target, 'illegit')
      chopstick_angle_speed = chopstick_angle_target = None
    elif chopstick_angle_speed > 0 and chopstick_angle_target <= -0.72:
      chopstick_angle_target = -0.72
    elif chopstick_angle_speed < 0 and chopstick_angle_target >= -0.06:
      chopstick_angle_target = -0.06

  if chopstick_angle_target:
    #chopstick_angle_target -= 0.37 # TODO turn into a param?
    delta_angle = chopstick_angle_target - next_angles[-1]
    chop_speed = delta_angle / dt
    chopstick_angle_target = np.clip(chopstick_angle_target, -0.72, -0.06)
    if abs(chop_speed) > 0.05:
      next_angles[-1] = chopstick_angle_target
      next_speed[-1] = np.clip(chop_speed, -0.2, 0.2)
      print('open/close chopstick', next_angles[-1], delta_angle, next_speed[-1])

  command.position = next_angles
  command.velocity = next_speed
  command.effort = state.arm.get_grav_comp_efforts(feedback).copy()

  return command

def grav_comp_command(state, feedback):
  command = hebi.GroupCommand(state.arm.group.size)
  command.effort = state.arm.get_grav_comp_efforts(feedback).copy()
  return command

def command_proc(state):
  group = state.arm.group
  group.feedback_frequency = 100.0

  num_modules = group.size

  command = hebi.GroupCommand(num_modules)
  feedback = hebi.GroupFeedback(num_modules)
  start_time = time()
  last_time = start_time

  while True:
    if group.get_next_feedback(reuse_fbk=feedback) is None:
      print_and_cr('Did not receive feedback')
      state.lock()
      state._quit = True
      state.unlock()

    state.lock()
    if state.quit:
      state.unlock()
      break

    feedback.get_position(state.current_position)
    feedback.get_velocity(state.current_velocity)

    current_mode = state.mode

    cur_time = time()
    dt = cur_time - last_time
    last_time = cur_time

    if state._print :
      pose_mat = state.arm.get_FK_ee(state.current_position)
      pose_xyz_rpy = np.empty(6)
      pose_xyz_rpy[0:3] = pose_mat[0:3,3].reshape(3)
      pose_xyz_rpy[3:6] = euler_angles_from_rotation_matrix(pose_mat[0:3,0:3])
      print(pose_xyz_rpy)
      print('\r\n----\r\n')
      state._print = False

    if current_mode == 'teleop':
      cmd_pose, cmd_vel = construct_teleop_target(state, feedback, dt)
      command = construct_command(state, feedback, cmd_pose, cmd_vel, dt)

    if current_mode == 'operational':

      # xyz jog
      if(state._speed_base < 0.1):
        state._speed_base += 0.001
      cmd_pose, cmd_vel = construct_jog_target(state, parse_jog_xyz_speed(state), dt)
      command = construct_command(state, feedback, cmd_pose, cmd_vel, dt,
                                  chopstick_angle_speed=parse_jog_open_chopstick(state))

    if current_mode == 'teleop' or current_mode == 'operational':
      #sys.stdout.write("{}, {}, {}; \n\n\n".format(command.position, command.velocity, command.effort))
      #sys.stdout.flush()
      group.send_command(command)
    else:
      group.send_command(grav_comp_command(state, feedback))
    
    state.unlock()


#######################################################################
# Cartesian control of the end effector.
#######################################################################

def run():
  # initialize the arm
  arm = arm_container.create_robot('chopstick.hrdf')
  state = State(arm)
  load_gain(state.arm.group, 'chopstick-gains.xml')

  # command is sent out via a separate thread
  cmd_thread = Thread(target=command_proc,
                      name='Command Thread',
                      args=(state,))
  cmd_thread.start()

  # this script is preserved for switching modes
  np.set_printoptions(suppress=True)
  print_and_cr("Press 'p' to enable cartesian control (wsadjl for moving, k for stopping)." +
               "\r\nPress 'g' to save gains, 't' to teleop and 'q' to quit.")
  res = getch()

  while res != 'q' and not state.quit:
    print_and_cr('')
    state.lock()

    current_mode = state.mode

    if res == 'g':
      print_and_cr('saving gains')
      save_gain(state.arm.group, 'chopstick-gains.xml')
    elif res == 'p':
      print_and_cr('Entering jogging mode')
      state._mode = 'operational'
      state._cmd_pose = state.arm.get_FK_ee(state.current_position)
      state._update_cmd_pose = True
      state._jog_direction = '0'
    elif res == 't':
      print_and_cr('Entering teleoperation mode')
      init_teleop(state)
    elif res =='v':
      state._print = True

    if current_mode == 'operational':
      if state._update_cmd_pose == True:
        state._cmd_pose = state.arm.get_FK_ee(state.current_position)

      jog_dict = {
        'w': 'x_plus', 'a': 'y_plus', 'j': 'z_plus', 'i': 'open',
        's': 'x_minus','d': 'y_minus','l': 'z_minus', 'o': 'close',
        'k': '0'
      }
      if res in jog_dict.keys():
        state._jog_direction = jog_dict[res]
        state._speed_base = 0.0 # default start speed
        state._update_cmd_pose = (jog_dict[res] == '0')

    state.unlock()
    res = getch()

  state._quit = True
  print_and_cr('')


if __name__ == '__main__':
  run()
