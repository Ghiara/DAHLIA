"""Motion primitives."""

import numpy as np
import cliport.tasks.grippers as grippers
from cliport.utils import utils
import time


class PickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pick and place primitive.
    
        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          pose0: SE(3) picking pose.
          pose1: SE(3) placing pose.
    
        Returns:
          timeout: robot movement timed out if True.
        """

        pick_pose, place_pose = pose0, pose1

        if isinstance(ee, grippers.Suction):

            # Execute picking primitive.
            prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
            postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
            prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
            postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
            timeout = movep(prepick_pose)

            # Move towards pick pose until contact is detected.
            delta = (np.float32([0, 0, -0.001]),
                     utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
            targ_pose = prepick_pose
            while not ee.detect_contact():  # and target_pose[2] > 0:
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose)
                if timeout:
                    return True

            # Activate end effector, move up, and check picking success.
            ee.activate()
            timeout |= movep(postpick_pose, self.speed)
            pick_success = ee.check_grasp()

            # Execute placing primitive if pick is successful.
            if pick_success:
                preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
                postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
                preplace_pose = utils.multiply(place_pose, preplace_to_place)
                postplace_pose = utils.multiply(place_pose, postplace_to_place)
                targ_pose = preplace_pose
                while not ee.detect_contact():
                    targ_pose = utils.multiply(targ_pose, delta)
                    timeout |= movep(targ_pose, self.speed)
                    if timeout:
                        return True
                ee.release()
                timeout |= movep(postplace_pose)

            # Move to prepick pose if pick is not successful.
            else:
                ee.release()
                timeout |= movep(prepick_pose)

            return timeout
        
        elif isinstance(ee, grippers.Grip):

            # Execute picking primitive.
            prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
            postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
            ready_to_pick = ((0, 0, 0.09), (0, 0, 0, 1))
            prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
            postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
            readypick_pose = utils.multiply(pick_pose, ready_to_pick)
            timeout = movep(prepick_pose)
            
            ee.release()  
            movej(targj=ee.open_gripper(), speed=10, effector=ee)

            # Move towards pick pose until contact is detected.
            delta = (np.float32([0, 0, -0.001]),
                     utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
            targ_pose = prepick_pose
            while not ee.at_target(readypick_pose[0][2]):  # and target_pose[2] > 0:
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose)
                if timeout:
                    return True   
            
            # Activate end effector, move up, and check picking success.
            start_time = time.time()
            while not ee.activated:
                timeout |= movej(targj=ee.close_gripper(), speed=5, effector=ee)
                ee.activate()
                if time.time() > start_time + 5:
                    print('activate time out')
                    break
                pass
            
            pick_success = ee.check_grasp() or ee.activated
            #print(f'pick_success: {pick_success}')
            #print(f'force: {ee.gripper_force()}')

            # Execute placing primitive if pick is successful.
            if pick_success:
                timeout |= movep(postpick_pose, self.speed)
                preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
                postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
                ready_to_place = ((0, 0, 0.11), (0, 0, 0, 1))
                preplace_pose = utils.multiply(place_pose, preplace_to_place)
                postplace_pose = utils.multiply(place_pose, postplace_to_place)
                readyplace_pose = utils.multiply(place_pose, ready_to_place)
                targ_pose = preplace_pose
                if not ee.check_grasp():
                    while not ee.detect_contact(): # not ee.at_target(readyplace_pose[0][2]):
                        targ_pose = utils.multiply(targ_pose, delta)
                        timeout |= movep(targ_pose, self.speed)
                        if timeout:
                            return True
                else:
                    while not ee.at_target(readyplace_pose[0][2]): # not ee.at_target(readyplace_pose[0][2]):
                        targ_pose = utils.multiply(targ_pose, delta)
                        timeout |= movep(targ_pose, self.speed)
                        if timeout:
                            return True
                ee.release()
                movej(targj=ee.open_gripper(), speed=10, effector=ee)
                timeout |= movep(postplace_pose)

            # Move to prepick pose if pick is not successful.
            else:
                ee.release()
                movej(targj=ee.open_gripper(), speed=10, effector=ee)
                timeout |= movep(prepick_pose)

            return timeout
        
        else:
            raise ValueError('Unknown End Effector')


def push(movej, movep, ee, pose0, pose1):  # pylint: disable=unused-argument
    """Execute pushing primitive.
  
    Args:
      movej: function to move robot joints.
      movep: function to move robot end effector pose.
      ee: robot end effector.
      pose0: SE(3) starting pose.
      pose1: SE(3) ending pose.
  
    Returns:
      timeout: robot movement timed out if True.
    """

    # Adjust push start and end positions.
    pos0 = np.float32((pose0[0][0], pose0[0][1], 0.005))
    pos1 = np.float32((pose1[0][0], pose1[0][1], 0.005))
    vec = np.float32(pos1) - np.float32(pos0)
    length = np.linalg.norm(vec)
    vec = vec / length
    pos0 -= vec * 0.02
    pos1 -= vec * 0.05

    # Align spatula against push direction.
    theta = np.arctan2(vec[1], vec[0])
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

    over0 = (pos0[0], pos0[1], 0.31)
    over1 = (pos1[0], pos1[1], 0.31)

    # Execute push.
    timeout = movep((over0, rot))
    timeout |= movep((pos0, rot))
    n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
    for _ in range(n_push):
        target = pos0 + vec * n_push * 0.01
        timeout |= movep((target, rot), speed=0.003)
    timeout |= movep((pos1, rot), speed=0.003)
    timeout |= movep((over1, rot))
    return timeout
