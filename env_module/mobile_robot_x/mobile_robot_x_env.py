import os
import numpy as np
import cv2
from gym import spaces
from env_module.base_env import BaseEnv
#  Number of steps before termination
MAX_STEPS = 250  
REWARD_DIST_THRESHOLD = 0.4  # Min distance to target before finishing an episode
RENDER_HEIGHT = 128  # output image shape (height, width, channels)
RENDER_WIDTH = 128
N_DISCRETE_ACTIONS = 4

DELTA_POS = 0.1  # DELTA_POS
NOISE_STD = 0.0

## TODO
ROBOT_WIDTH = 0.2
ROBOT_LENGTH = 0.325 * 2


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class MobileRobotXEnv(BaseEnv):
    """


    """

    def __init__(self, renders=False, is_discrete=True,
                 name="MobileX", shape_reward=False, record_data=False, srl_model="raw_pixels",
                 random_target=False, state_dim=-1, verbose=False,
                 save_path='./data/', env_rank=0, srl_pipe=None, img_shape=None,  **_):
        super(MobileRobotXEnv, self).__init__(srl_model=srl_model,
                                                env_rank=env_rank,
                                                srl_pipe=srl_pipe)

        self._renders = renders # open an GUI, display the behavior of robot (evaluation mode).
        self.img_shape = img_shape  # channel first !!
        if self.img_shape is None:
            self._width = RENDER_WIDTH
            self._height = RENDER_HEIGHT
        else:
            self._height, self._width = self.img_shape[1:]

        self._shape_reward = shape_reward
        self._random_target = random_target

        self._is_discrete = is_discrete
        self.state_dim = state_dim
        self.saver = None
        self.verbose = verbose
        self.max_steps = MAX_STEPS
        self.robot_pos = np.zeros(2)
        self.target_pos = np.zeros(2)
        # Boundaries of the square env
        self._min_x, self._max_x = 0, 4
        self._min_y, self._max_y = 0, 4
        self.collision_margin = 0.1

        self._observation = []
        self._env_step_counter = 0
        self.has_bumped = False  # Used for handling collisions
        self.imgs_folder = "mobilerobot" ## 'mobilerobot', 'omnirobot'
        self.robot_img = cv2.imread("./env_module/mobile_robot_x/images/{}/robot_224.png".format(self.imgs_folder))
        self.target_img = cv2.imread("./env_module/mobile_robot_x/images/{}/target_224.png".format(self.imgs_folder))
        self.background_img = cv2.imread("./env_module/mobile_robot_x/images/{}/background_224.png".format(self.imgs_folder))

        robot_img_shape = self.robot_img.shape
        self.robot_img = cv2.resize(self.robot_img, (int(
            robot_img_shape[1]*self._width/224), int(robot_img_shape[0]*self._height/224)))
        target_img_shape = self.target_img.shape
        self.target_img = cv2.resize(self.target_img, (int(
            target_img_shape[1]*self._width/224), int(target_img_shape[0]*self._height/224)))
        self.background_img = cv2.resize(self.background_img, (self._width, self._height))

        self.srl_model = srl_model

        if record_data:
            self.saver = EpisodeSaver(name, max_distance, state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=False, path=save_path)

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        if self.srl_model == "ground_truth":
            self.state_dim = self.getGroundTruthDim()

        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    # @staticmethod
    def getGroundTruthDim(self):
        """
        The new convention for the ground truth (GT): GT should include target position under random target
        setting. 
        """
        if not self._random_target:
            return 2
        else:
            return 4

    def getGroundTruth(self):
        """
        The new convention for the ground truth (GT): GT should include target position under random target
        setting. A better solution would be "change all the environment files, especially srl_env.py" !!! 

        """
        robot_pos = self.robot_pos
        if self._random_target:
            target_pos = self.target_pos
            return np.concatenate([robot_pos, target_pos], axis=0)
        else:
            return robot_pos

    def reset(self):
        self._env_step_counter = 0
        self.has_bumped = False
        # Init the robot randomly
        x_start = self._max_x / 2 + self.np_random.uniform(- self._max_x / 3, self._max_x / 3)
        y_start = self._max_y / 2 + self.np_random.uniform(- self._max_y / 3, self._max_y / 3)

        self.robot_pos = np.array([x_start, y_start])

        # Initialize target position
        x_pos = 0.8 * self._max_x
        y_pos = 0.7 * self._max_y
        if self._random_target:
            margin = self.collision_margin * self._max_x ## 0.4
            x_pos = self.np_random.uniform(self._min_x + margin, self._max_x - margin)
            y_pos = self.np_random.uniform(self._min_y + margin, self._max_y - margin)

        self.target_pos = np.array([x_pos, y_pos])

        # self.walls = [wall_left, wall_bottom, wall_right, wall_top]

        self._observation = self.getObservation()

        if self.saver is not None:
            self.saver.reset(self._observation, self.target_pos, self.robot_pos)

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation)

        return self._observation

    def getObservation(self):
        """
        :return: (numpy array)
        """
        self._observation = self.render("rgb_array")
        return self._observation

    def step(self, action):
        # True if it has bumped against a wall
        self._env_step_counter += 1
        self.has_bumped = False
        if self._is_discrete:
            dv = DELTA_POS
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0][action]
            dy = [0, 0, -dv, dv][action]
            real_action = np.array([dx, dy])
        else:
            dv = DELTA_POS
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            # scale action amplitude between -dv and dv
            real_action = np.maximum(np.minimum(action, 1), -1) * dv

        previous_pos = self.robot_pos.copy()
        self.robot_pos += real_action
        # Handle collisions
        for i, (limit, robot_dim) in enumerate(zip([self._max_y, self._max_x], [ROBOT_WIDTH, ROBOT_LENGTH])):
            margin = self.collision_margin * limit + robot_dim / 2
            # If it has bumped against a wall, stay at the previous position
            if self.robot_pos[i] < margin or self.robot_pos[i] > limit - margin:
                self.has_bumped = True
                self.robot_pos = previous_pos
                break
        # Update mobile robot position on image

        self._observation = self.getObservation()

        # calculate the reward of this step
        distance = np.linalg.norm(self.target_pos - self.robot_pos, 2)
        reward = 0
        if distance <= REWARD_DIST_THRESHOLD:
            reward = 1
        # Negative reward when it bumps into a wall
        if self.has_bumped:
            reward = -1
        if self._shape_reward:
            reward = -distance
        
        done = self._env_step_counter > self.max_steps

        if self.saver is not None:
            self.saver.step(self._observation, action, reward, done, self.robot_pos)

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation), reward, done, {}

        return np.array(self._observation), reward, done, {}

    def render(self, mode='rgb_array'):
        """
        mode is useless
        """

        image = self.background_img.copy()
        for obj_pos, obj_img in zip([self.target_pos, self.robot_pos], [self.target_img, self.robot_img]):
            obj_img_pos = np.array([(obj_pos[1]/self._max_y) * self._height,
                                    (obj_pos[0]/self._max_x) * self._width], dtype=int)
            h_st = obj_img_pos[0]-obj_img.shape[0]//2
            h_ed = obj_img.shape[0]+h_st
            w_st = obj_img_pos[1]-obj_img.shape[1]//2
            w_ed = obj_img.shape[1]+w_st
            mask = (np.mean(obj_img, axis=-1) < 240)
            image[h_st:h_ed, w_st:w_ed, :][mask] = obj_img[mask]

        if self._renders:
            cv2.imshow("Keep pressing (any key) to display or 'q'/'Esc' to quit.", image)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:  # 'q' or 'Esc'
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
        return image[..., ::-1]
    def interactive(self, show_map=False):
        image = self.getObservation()
        cv2.imshow("image", image[..., ::-1])
        while True:
            k = cv2.waitKey(0)
            if k == 27 or k == ord("q"):  # press 'Esc' or 'q' to quit
                break
            elif k == ord("2") or k == 84:
                # down
                action = 3
                image, reward, done, _ = self.step(action)
                print("Action: {}, Reward: {}, Done: {}".format(action, reward, done))
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("6") or k == 83:
                # right
                action = 1
                image, reward, done, _ = self.step(action)
                print("Action: {}, Reward: {}, Done: {}".format(action, reward, done))
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("8") or k == 82:
                # up
                action = 2
                image, reward, done, _ = self.step(action)
                print("Action: {}, Reward: {}, Done: {}".format(action, reward, done))
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("4") or k == 81:
                # left
                action = 0
                image, reward, done, _ = self.step(action)
                print("Action: {}, Reward: {}, Done: {}".format(action, reward, done))
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            else:
                print("You are pressing the key: {}".format(k))

if __name__ == "__main__":
    print("Start")
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Labyrinth environment (debug purpose)")
    parser.add_argument('--seed', default=0, type=int, help='random seed for initial robot position')
    parser.add_argument('--img-shape', type=str, default="(3,224,224)", help="Image shape of environment.")
    args, unknown = parser.parse_known_args()

    if args.img_shape is None:
        img_shape = (3, 224, 224)  # (3,224,224)
    else:
        img_shape = tuple(map(int, args.img_shape[1:-1].split(",")))
    _, RENDER_HEIGHT, RENDER_WIDTH = img_shape
    
    np.random.seed(args.seed)
    Env = MobileRobotXEnv()
    img = Env.reset()
    Env.interactive()

