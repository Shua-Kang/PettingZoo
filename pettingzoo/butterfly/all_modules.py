from pettingzoo.butterfly import (
    cooperative_pong_v6,
    knights_archers_zombies_v10,
    pistonball_v6,
)
from pettingzoo.movingout import moving_out
butterfly_environments = {
    "butterfly/knights_archers_zombies_v10": knights_archers_zombies_v10,
    "butterfly/pistonball_v6": pistonball_v6,
    "butterfly/cooperative_pong_v6": cooperative_pong_v6,
    "movingout/moving_out_v0": moving_out,
}
