from __future__ import print_function

import os
import sys
import numpy as np
from time import time, sleep

# Ros
import rospy
from geometry_msgs.msg import PointStamped
import tf

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


class State(object):

  def __init__(self, arm):
    self._quit = False
    self._mode = 'training'
    self._arm = arm

    self._current_position = np.empty(arm.dof_count, dtype=np.float64)
    self._current_velocity = np.empty(arm.dof_count, dtype=np.float64)
    self._cmd_pose = arm.get_FK(self._current_position)
    self._update_cmd_pose = False

    # For jogging-like cartesian control
    self._speed_base = 0.0
    self._jog_direction = '0'

    # For teleop-based cartesian control
    self._teleop_target_zero = None
    self._teleop_latest_target = np.zeros(3) # TODO to 6D pose
    self._robot_zero_pose = np.zeros((arm.dof_count, 4))

    # For threading safety
    self._mutex = Lock()

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

def get_current_pose(arm, feedback): # TODO deprecated?
  current_position = np.empty(arm.dof_count, dtype=np.float64)
  feedback.get_position(current_position)
  return arm.get_FK(current_position)


def init_teleop(state):
  # !! ensures you have locked state outside this function scope
  print_and_cr('Initializing teleoperation')
  rospy.init_node('hebiteleop')

  # tf
  transformListener = tf.TransformListener()
  transformListener.waitForTransform("/optitrack_natnet", "/map", rospy.Time(0),rospy.Duration(1.0))

  # Set the current robot pose to correspond to the first published teleop target (both are considered "zero")
  print_and_cr('setting zero state')
  state._cmd_pose = state.arm.get_FK(state.current_position)
  state._robot_zero_pose = state._cmd_pose.copy()

  def callback(data):
    #rospy.loginfo('Received chobi-teleop info %f %f %f', data.point.x, data.point.y, data.point.z)
    #cur_point = transformListener.transformPoint("map", data).point
    cur_point = data.point
    state.lock()
    if state.teleop_target_zero is None:
      state._teleop_target_zero = np.array([cur_point.x, cur_point.y, cur_point.z])
      state._mode = 'teleop'
      rospy.loginfo('Initialization of teleoperation is completed')
    state._teleop_latest_target =  np.array([cur_point.x, cur_point.y, cur_point.z])
    state.unlock()
  
  rospy.Subscriber('/M1/point', PointStamped, callback, queue_size=1)


def construct_velocity(state, feedback):
  target_pose = np.zeros_like(state.robot_zero_pose)
  a,b,c = np.array(state.teleop_latest_target-state.teleop_target_zero).reshape(3,1)
  d = b
  b = -c
  c = d
  # without tf, manually transform
  target_pose[0:3, 3] = np.array([a,b,c]).reshape(3,1)
  #target_pose[0:3,3] = np.array([[1.],[1.0],[0.0001]])

  delta_pose = target_pose + state.robot_zero_pose - state._cmd_pose # get_current_pose(state.arm, feedback)
  velocity = np.array(delta_pose[0:3,3])
  tol = 2 * 1e-4
  velocity[velocity < tol] = 0.0
  np.clip(velocity, -0.03, 0.03, out=velocity)
  #print(target_pose[0:3,3], state.robot_zero_pose[0:3,3], delta_pose[0:3,3], velocity)
  return velocity


def construct_command(state, feedback, velocity, dt):
  command = hebi.GroupCommand(state.arm.group.size)

  # position and velocity
  next_angles = state.current_position
  next_speed = [0.0, 0.0, 0.0, 0.0, 0.0]
  
  x_speed, y_speed, z_speed = velocity
  state._cmd_pose[0,3] = state._cmd_pose[0,3] + x_speed*dt;
  state._cmd_pose[1,3] = state._cmd_pose[1,3] + y_speed*dt;
  state._cmd_pose[2,3] = state._cmd_pose[2,3] + z_speed*dt;

  #current_pose = state._cmd_pose # np.array(state.arm.get_FK(state.current_position)) #TODO get rid of it
  #xyz_pose = current_pose[0:3,3] + np.array(velocity).reshape(3,1) * dt
  #cmd_pose_xyz = [xyz_pose[0,0], xyz_pose[1,0], xyz_pose[2,0]]
  cmd_pose_xyz = [state._cmd_pose[0,3], state._cmd_pose[1,3], state._cmd_pose[2,3]]
  cmd_vel_xyz = [x_speed, y_speed, z_speed]
  jog_cmd = state.arm.get_jog_xyz(cmd_pose_xyz, state.current_position, cmd_vel_xyz, dt)

  next_angles[0:3] = jog_cmd[0]
  next_speed[0:3] = jog_cmd[1]
  command.position = next_angles
  command.velocity = next_speed

  # effort: grav comp and spring offset
  grav_comp_effort = state.arm.get_grav_comp_efforts(feedback).copy()
  total_effort = np.empty(5, np.float64)
  spring_effort = 4.0 - 5.0*(state.current_position[1] - 1.4)
  np.add(grav_comp_effort, [0.0, -spring_effort, 0.0, 0.0, 0.0], total_effort)
  command.effort = total_effort

  return command

def parse_jog_speed(state):
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

    if current_mode == 'teleop':
      state._cmd_pose = state.arm.get_FK(state.current_position)
      vel = construct_velocity(state, feedback)
      print('vel', vel)
      command = construct_command(state, feedback, vel, dt)

    if current_mode == 'operational':
      if(state._speed_base < 0.1):
        state._speed_base += 0.001
      x_speed, y_speed, z_speed = parse_jog_speed(state)
      command = construct_command(state, feedback, np.array([x_speed, y_speed, z_speed]).reshape(3,1), dt)

      '''
      # Try to get rotation cartesian control out

      next_angles = state.current_position
      next_speed = [0.0, 0.0, 0.0, 0.0, 0.0]
      state._cmd_pose[0, 3] = state._cmd_pose[0, 3] + x_speed*dt;
      state._cmd_pose[1, 3] = state._cmd_pose[1, 3] + y_speed*dt;
      state._cmd_pose[2, 3] = state._cmd_pose[2, 3] + z_speed*dt;
      #base_angle = np.arctan2(state._cmd_pose[1, 3], state._cmd_pose[0, 3])
      base_angle = np.arctan2(y_speed, x_speed)
      base_angle_compenstation = np.array([[np.cos(base_angle), -np.sin(base_angle), 0.0],
                                           [np.sin(base_angle), np.cos(base_angle), 0.0],
                                           [0.0, 0.0, 1.0]])

      print('compenstaion')
      print(base_angle_compenstation)
      print('before')
      print(state._cmd_pose[0:3, 0:3])
      state._cmd_pose[0:3, 0:3] = base_angle_compenstation*state._cmd_pose[0:3, 0:3]
      print('After')
      print(state._cmd_pose[0:3, 0:3])

      cmd_vel_xyz = [x_speed, y_speed, z_speed]

      jog_cmd = state.arm.get_jog(state._cmd_pose, state.current_position, cmd_vel_xyz, dt)
      next_angles[0:3] = jog_cmd[0]
      next_speed[0:3] = jog_cmd[1]
      command.position = next_angles
      command.velocity = next_speed

      current_pose = state.arm.get_FK(state.current_position)
    
      command.effort = total_effort
      '''

    if current_mode == 'teleop' or current_mode == 'operational':
      #sys.stdout.write("{}, {}, {}; \n\n\n".format(command.position, command.velocity, command.effort))
      #sys.stdout.flush()
      group.send_command(command)
    
    state.unlock()

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

#######################################################################
# Cartesian control of the end effector.
#######################################################################

def run():
  # initialize the arm
  arm = arm_container.create_5_dof('chopstick.hrdf')
  state = State(arm)
  load_gain(state.arm.group, 'chopstick-gains.xml')

  # command is sent out via a separate thread
  cmd_thread = Thread(target=command_proc,
                      name='Command Thread',
                      args=(state,))
  cmd_thread.start()

  # this script is preserved for switching modes
  print_and_cr("Press 'p' to enable cartesian control (wsadjl for moving, k for stopping)." +
               "\nPress 'g' to save gains, 't' to teleop and 'q' to quit.")
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
      state._update_cmd_pose = True
    elif res == 't':
      print_and_cr('Entering teleoperation mode')
      init_teleop(state)

    if state._update_cmd_pose == True:
        state._cmd_pose = state.arm.get_FK(state.current_position)  # TODO: cmd_pose, does it make sense to "keep updating it" and storing it?

    if current_mode == 'operational':
      jog_dict = {
        'w': 'x_plus', 'a': 'y_plus', 'j': 'z_plus',
        's': 'x_minus','d': 'y_minus','l': 'z_minus',
        'k': '0'
      }
      if res in jog_dict.keys():
        state._jog_direction = jog_dict[res]
        state._speed_base = 0.0 # default start speed
        state._update_cmd_pose = False if jog_dict[res] != '0' else True

    state.unlock()
    res = getch()

  state._quit = True
  print_and_cr('')


if __name__ == '__main__':
  run()
