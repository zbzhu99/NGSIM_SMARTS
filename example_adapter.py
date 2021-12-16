import gym
import utils


def get_observation_adapter(obs_stack_size):
    stack_size = obs_stack_size
    # look_ahead = 10
    closest_neighbor_num = 6
    img_resolution = 40
    observe_lane_num = 3

    subscribed_features = dict(
        # distance_to_center=(stack_size, 1),
        ego_pos=(stack_size, 2),
        heading=(stack_size, 1),
        speed=(stack_size, 1),
        neighbor=(stack_size, closest_neighbor_num * 4),  # dist, speed, ttc
        # heading_errors=(stack_size, look_ahead),
        # steering=(stack_size, 1),
        # ego_lane_dist_and_speed=(stack_size, observe_lane_num + 1),
        # img_gray=(stack_size, img_resolution, img_resolution) if use_rgb else False,
    )

    observation_space = gym.spaces.Dict(
        utils.subscribe_features(**subscribed_features)
    )

    observation_adapter = utils.get_observation_adapter(
        observation_space,
        # look_ahead=look_ahead,
        observe_lane_num=observe_lane_num,
        resize=(img_resolution, img_resolution),
        closest_neighbor_num=closest_neighbor_num,
    )

    return observation_adapter
