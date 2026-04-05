"""Physics for Street Crosser — car movement, collision detection."""

import math
import numpy as np
from config import Config
from env.state import Car, Player, GameState, UP, DOWN, LEFT, RIGHT, NOOP


def create_initial_state(config: Config, rng: np.random.RandomState) -> GameState:
    """Create fresh game state with player at bottom and random cars in lanes."""
    # Player starts at bottom center
    player = Player(col=float(config.grid_cols // 2), row=float(config.grid_rows - 1))

    # Create cars in each lane
    cars = _create_cars(config, rng)

    return GameState(player=player, cars=cars)


def _create_cars(config: Config, rng: np.random.RandomState) -> list:
    """Generate random cars across all lanes."""
    cars = []
    lane_start = config.safe_rows  # first lane row
    colors = config.car_colors

    if config.free_roam:
        # Cars on random rows in the road area
        road_rows = list(range(config.safe_rows,
                               config.grid_rows - config.safe_rows))
        total_cars = config.num_lanes * config.max_cars_per_lane
        for i in range(total_cars):
            row = rng.choice(road_rows)
            # Make sure tall cars don't overflow into safe zones
            if row + config.car_height > config.grid_rows - config.safe_rows:
                row = config.grid_rows - config.safe_rows - config.car_height
            direction = 1 if rng.random() < 0.5 else -1
            speed = rng.uniform(config.min_car_speed, config.max_car_speed) * direction
            color = colors[i % len(colors)]
            x = rng.uniform(0, config.grid_cols)
            car = Car(x=x, row=row, speed=speed, width=config.car_width,
                      color=color, height=config.car_height)
            cars.append(car)
    else:
        for lane_idx in range(config.num_lanes):
            row = lane_start + lane_idx
            num_cars = rng.randint(config.min_cars_per_lane,
                                   config.max_cars_per_lane + 1)

            # Alternate direction per lane
            direction = 1 if lane_idx % 2 == 0 else -1
            speed = rng.uniform(config.min_car_speed,
                                config.max_car_speed) * direction

            color = colors[lane_idx % len(colors)]

            for _ in range(num_cars):
                x = rng.uniform(0, config.grid_cols)
                car = Car(x=x, row=row, speed=speed, width=config.car_width,
                          color=color, height=config.car_height)
                cars.append(car)

    return cars


def step_physics(state: GameState, action: int, config: Config,
                 rng: np.random.RandomState) -> tuple:
    """Advance one step. Returns (reward, scored, hit)."""
    player = state.player
    state.hit = False

    # Move player
    _move_player(player, action, config)

    # Move cars — smooth wrapping so cars are always visible approaching
    for car in state.cars:
        car.x += car.speed / config.fps  # normalize by fps for consistent speed
        # Wrap around with seamless transition (car slides in from off-screen)
        if car.speed > 0 and car.x > config.grid_cols:
            car.x = -car.width
        elif car.speed < 0 and car.x + car.width < 0:
            car.x = config.grid_cols

    # Check collision
    hit = _check_collision(player, state.cars, config)

    reward = 0.0
    scored = False

    if hit:
        state.hit = True
        reward = -1.0
        # Reset player to bottom
        player.col = float(config.grid_cols // 2)
        player.row = float(config.grid_rows - 1)

    # Check if player reached the goal (top safe zone)
    if player.row < config.safe_rows:
        state.score += 1
        reward = 1.0
        scored = True
        # Reset to bottom
        player.col = float(config.grid_cols // 2)
        player.row = float(config.grid_rows - 1)
        # Respawn cars for variety
        state.cars = _create_cars(config, rng)

    state.step_count += 1

    # Episode end
    if config.max_steps > 0 and state.step_count >= config.max_steps:
        state.done = True
    if state.score >= config.target_score:
        state.done = True

    return reward, scored, hit


def _move_player(player: Player, action: int, config: Config):
    """Move player based on action, clamped to grid."""
    step = config.player_speed
    if action == UP:
        player.row = max(0.0, player.row - step)
    elif action == DOWN:
        player.row = min(float(config.grid_rows - 1), player.row + step)
    elif action == LEFT:
        player.col = max(0.0, player.col - step)
    elif action == RIGHT:
        player.col = min(float(config.grid_cols - 1), player.col + step)


def _check_collision(player: Player, cars: list, config: Config) -> bool:
    """Check if player overlaps with any car (supports fractional positions)."""
    px = player.col
    py = player.row
    # Player occupies roughly [px, px+1) x [py, py+1) with some margin
    p_left = px + 0.2
    p_right = px + 0.8
    p_top = py + 0.2
    p_bottom = py + 0.8

    for car in cars:
        # Car occupies rows [car.row, car.row + car.height)
        if p_bottom <= car.row or p_top >= car.row + car.height:
            continue
        # Car occupies [car.x, car.x + car.width) horizontally
        if p_right <= car.x or p_left >= car.x + car.width:
            continue
        return True

    return False
