from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import (
    TrainFreq,
    TrainFrequencyUnit,
)


@dataclass
class Config:
    additional_episodes: int = 5_000
    max_steps: int = 500
    console_log_interval: int = 100
    save_interval_episodes: int = 500

    physics_dt: float = 0.02

    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_pole_length: float = 0.5
    max_force: float = 10.0

    cart_limit: float = 2.4
    angle_limit_degrees: float = 12.0

    survival_reward: float = 1.0
    angle_penalty: float = 4.0
    angular_velocity_penalty: float = 0.15
    cart_position_penalty: float = 0.03
    cart_velocity_penalty: float = 0.01
    force_penalty: float = 0.001
    failure_penalty: float = 10.0
    success_bonus: float = 10.0

    learning_rate: float = 3e-4
    buffer_size: int = 20_000
    learning_starts: int = 500
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 16
    gradient_steps: int = 1
    network_size: int = 32

    cpu_threads: int = 2
    seed: int = 2026

    model_path: str = "models/sac_cartpole"
    replay_buffer_path: str = "models/replay_buffer.pkl"
    metadata_path: str = "models/metadata.json"

    log_directory: str = "logs"
    csv_name: str = "result.csv"


class CSVLogger:
    fields = [
        "timestamp",
        "episode",
        "run_episode",
        "steps",
        "total_reward",
        "result",
        "initial_cart_position",
        "initial_cart_velocity",
        "initial_angle_degrees",
        "initial_angular_velocity",
        "final_cart_position",
        "final_cart_velocity",
        "final_angle_degrees",
        "final_angular_velocity",
        "mean_force",
        "mean_absolute_force",
        "maximum_absolute_force",
        "average_steps_100",
        "average_reward_100",
        "success_rate_100",
        "episodes_per_second",
        "environment_steps_per_second",
        "total_environment_steps",
    ]

    def __init__(self, config: Config) -> None:
        directory = Path(config.log_directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.path = directory / config.csv_name

        if not self.path.exists():
            with self.path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fields,
                )
                writer.writeheader()

    def write(self, values: dict[str, Any]) -> None:
        with self.path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )
            writer.writerow(values)


class ContinuousCartPoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        completed_episodes: int = 0,
        total_environment_steps: int = 0,
    ) -> None:
        super().__init__()

        self.config = config
        self.csv_logger = CSVLogger(config)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [-4.8, -20.0, -1.0, -1.0, -20.0],
                dtype=np.float32,
            ),
            high=np.array(
                [4.8, 20.0, 1.0, 1.0, 20.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.completed_episodes = completed_episodes
        self.run_completed_episodes = 0

        self.total_environment_steps = total_environment_steps
        self.starting_environment_steps = total_environment_steps

        self.current_episode = completed_episodes + 1
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

        self.initial_x = 0.0
        self.initial_x_dot = 0.0
        self.initial_theta = 0.0
        self.initial_theta_dot = 0.0

        self.force_sum = 0.0
        self.absolute_force_sum = 0.0
        self.maximum_absolute_force = 0.0

        self.recent_steps: deque[int] = deque(maxlen=100)
        self.recent_rewards: deque[float] = deque(maxlen=100)
        self.recent_successes: deque[int] = deque(maxlen=100)

        self.start_time = time.perf_counter()

    @property
    def angle_limit(self) -> float:
        return math.radians(
            self.config.angle_limit_degrees
        )

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.x, -4.8, 4.8),
                np.clip(self.x_dot, -20.0, 20.0),
                math.cos(self.theta),
                math.sin(self.theta),
                np.clip(self.theta_dot, -20.0, 20.0),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.current_episode = self.completed_episodes + 1
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.force_sum = 0.0
        self.absolute_force_sum = 0.0
        self.maximum_absolute_force = 0.0

        state = self.np_random.uniform(
            low=-0.05,
            high=0.05,
            size=4,
        )

        self.x = float(state[0])
        self.x_dot = float(state[1])
        self.theta = float(state[2])
        self.theta_dot = float(state[3])

        self.initial_x = self.x
        self.initial_x_dot = self.x_dot
        self.initial_theta = self.theta
        self.initial_theta_dot = self.theta_dot

        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        c = self.config

        normalized_action = float(
            np.clip(action[0], -1.0, 1.0)
        )

        force = normalized_action * c.max_force

        total_mass = c.mass_cart + c.mass_pole
        polemass_length = (
            c.mass_pole * c.half_pole_length
        )

        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        temporary = (
            force
            + polemass_length
            * self.theta_dot**2
            * sin_theta
        ) / total_mass

        theta_acceleration = (
            c.gravity * sin_theta
            - cos_theta * temporary
        ) / (
            c.half_pole_length
            * (
                4.0 / 3.0
                - c.mass_pole
                * cos_theta**2
                / total_mass
            )
        )

        cart_acceleration = (
            temporary
            - polemass_length
            * theta_acceleration
            * cos_theta
            / total_mass
        )

        self.x_dot += (
            c.physics_dt * cart_acceleration
        )

        self.theta_dot += (
            c.physics_dt * theta_acceleration
        )

        self.x += c.physics_dt * self.x_dot
        self.theta += c.physics_dt * self.theta_dot

        self.episode_steps += 1
        self.total_environment_steps += 1

        absolute_force = abs(force)

        self.force_sum += force
        self.absolute_force_sum += absolute_force

        self.maximum_absolute_force = max(
            self.maximum_absolute_force,
            absolute_force,
        )

        fallen = (
            abs(self.theta) > self.angle_limit
        )

        outside_track = (
            abs(self.x) > c.cart_limit
        )

        reached_time_limit = (
            self.episode_steps >= c.max_steps
        )

        terminated = (
            fallen or outside_track
        )

        truncated = (
            reached_time_limit
            and not terminated
        )

        if fallen:
            result = "fallen"
        elif outside_track:
            result = "outside_track"
        elif truncated:
            result = "success"
        else:
            result = "running"

        reward = (
            c.survival_reward
            - c.angle_penalty * self.theta**2
            - c.angular_velocity_penalty
            * self.theta_dot**2
            - c.cart_position_penalty * self.x**2
            - c.cart_velocity_penalty * self.x_dot**2
            - c.force_penalty * force**2
        )

        if terminated:
            reward -= c.failure_penalty
        elif truncated:
            reward += c.success_bonus

        self.episode_reward += reward

        info: dict[str, Any] = {
            "force": force,
            "normalized_action": normalized_action,
            "result": result,
            "episode_number": self.current_episode,
        }

        if terminated or truncated:
            self._finish_episode(result)

            info["episode_summary"] = {
                "reward": float(self.episode_reward),
                "length": int(self.episode_steps),
                "result": result,
            }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _finish_episode(
        self,
        result: str,
    ) -> None:
        episode = self.current_episode

        self.completed_episodes += 1
        self.run_completed_episodes += 1

        self.recent_steps.append(
            self.episode_steps
        )

        self.recent_rewards.append(
            self.episode_reward
        )

        self.recent_successes.append(
            1 if result == "success" else 0
        )

        average_steps = float(
            np.mean(self.recent_steps)
        )

        average_reward = float(
            np.mean(self.recent_rewards)
        )

        success_rate = float(
            np.mean(self.recent_successes)
        )

        elapsed = (
            time.perf_counter()
            - self.start_time
        )

        episodes_per_second = (
            self.run_completed_episodes
            / max(elapsed, 1e-9)
        )

        run_environment_steps = (
            self.total_environment_steps
            - self.starting_environment_steps
        )

        environment_steps_per_second = (
            run_environment_steps
            / max(elapsed, 1e-9)
        )

        denominator = max(
            self.episode_steps,
            1,
        )

        mean_force = (
            self.force_sum / denominator
        )

        mean_absolute_force = (
            self.absolute_force_sum / denominator
        )

        self.csv_logger.write(
            {
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "episode": episode,
                "run_episode": (
                    self.run_completed_episodes
                ),
                "steps": self.episode_steps,
                "total_reward": round(
                    self.episode_reward,
                    6,
                ),
                "result": result,
                "initial_cart_position": round(
                    self.initial_x,
                    6,
                ),
                "initial_cart_velocity": round(
                    self.initial_x_dot,
                    6,
                ),
                "initial_angle_degrees": round(
                    math.degrees(
                        self.initial_theta
                    ),
                    6,
                ),
                "initial_angular_velocity": round(
                    self.initial_theta_dot,
                    6,
                ),
                "final_cart_position": round(
                    self.x,
                    6,
                ),
                "final_cart_velocity": round(
                    self.x_dot,
                    6,
                ),
                "final_angle_degrees": round(
                    math.degrees(self.theta),
                    6,
                ),
                "final_angular_velocity": round(
                    self.theta_dot,
                    6,
                ),
                "mean_force": round(
                    mean_force,
                    6,
                ),
                "mean_absolute_force": round(
                    mean_absolute_force,
                    6,
                ),
                "maximum_absolute_force": round(
                    self.maximum_absolute_force,
                    6,
                ),
                "average_steps_100": round(
                    average_steps,
                    3,
                ),
                "average_reward_100": round(
                    average_reward,
                    6,
                ),
                "success_rate_100": round(
                    success_rate,
                    6,
                ),
                "episodes_per_second": round(
                    episodes_per_second,
                    3,
                ),
                "environment_steps_per_second": round(
                    environment_steps_per_second,
                    3,
                ),
                "total_environment_steps": (
                    self.total_environment_steps
                ),
            }
        )

        if (
            self.run_completed_episodes
            % self.config.console_log_interval
            == 0
            or result == "success"
        ):
            print(
                f"episode={episode} "
                f"steps={self.episode_steps} "
                f"reward={self.episode_reward:.3f} "
                f"result={result} "
                f"avg_steps_100={average_steps:.1f} "
                f"avg_reward_100={average_reward:.3f} "
                f"success_100={success_rate:.3f} "
                f"env_steps_per_second="
                f"{environment_steps_per_second:.1f}"
            )


class TrainingCallback(BaseCallback):
    def __init__(
        self,
        config: Config,
        target_episodes: int,
    ) -> None:
        super().__init__(verbose=0)

        self.config = config
        self.target_episodes = target_episodes
        self.last_saved_episode = -1

    def _completed_episodes(self) -> int:
        return int(
            self.training_env.get_attr(
                "completed_episodes"
            )[0]
        )

    def _environment_steps(self) -> int:
        return int(
            self.training_env.get_attr(
                "total_environment_steps"
            )[0]
        )

    def _save(self) -> None:
        completed = (
            self._completed_episodes()
        )

        Path(
            self.config.model_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save(
            self.config.model_path
        )

        self.model.save_replay_buffer(
            self.config.replay_buffer_path
        )

        metadata = {
            "completed_episodes": completed,
            "total_environment_steps": (
                self._environment_steps()
            ),
            "model_timesteps": int(
                self.model.num_timesteps
            ),
            "saved_at": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "config": asdict(self.config),
        }

        metadata_path = Path(
            self.config.metadata_path
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"saved episode={completed}")

    def _on_step(self) -> bool:
        completed = (
            self._completed_episodes()
        )

        if (
            completed > 0
            and completed
            % self.config.save_interval_episodes
            == 0
            and completed
            != self.last_saved_episode
        ):
            self._save()
            self.last_saved_episode = completed

        return completed < self.target_episodes

    def _on_training_end(self) -> None:
        self._save()


def load_metadata(
    config: Config,
) -> dict[str, Any]:
    path = Path(config.metadata_path)

    if not path.exists():
        return {
            "completed_episodes": 0,
            "total_environment_steps": 0,
        }

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        print(f"metadata_load_error={error}")

        return {
            "completed_episodes": 0,
            "total_environment_steps": 0,
        }


def save_state(
    model: SAC,
    environment: ContinuousCartPoleEnv,
    config: Config,
) -> None:
    Path(
        config.model_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        config.model_path
    )

    model.save_replay_buffer(
        config.replay_buffer_path
    )

    metadata = {
        "completed_episodes": (
            environment.completed_episodes
        ),
        "total_environment_steps": (
            environment.total_environment_steps
        ),
        "model_timesteps": int(
            model.num_timesteps
        ),
        "saved_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "config": asdict(config),
    }

    metadata_path = Path(
        config.metadata_path
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_int(
    prompt: str,
    default: int,
) -> int:
    value = input(
        f"{prompt} [{default}]: "
    ).strip()

    if value == "":
        return default

    return int(value)


def main() -> None:
    config = Config()

    config.additional_episodes = read_int(
        "additional episodes",
        config.additional_episodes,
    )

    config.max_steps = read_int(
        "maximum steps per episode",
        config.max_steps,
    )

    config.console_log_interval = read_int(
        "console log interval",
        config.console_log_interval,
    )

    config.save_interval_episodes = read_int(
        "save interval",
        config.save_interval_episodes,
    )

    torch.set_num_threads(
        config.cpu_threads
    )

    Path(
        config.log_directory
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        config.model_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata(config)

    completed_episodes = int(
        metadata.get(
            "completed_episodes",
            0,
        )
    )

    total_environment_steps = int(
        metadata.get(
            "total_environment_steps",
            0,
        )
    )

    target_episodes = (
        completed_episodes
        + config.additional_episodes
    )

    print(
        f"torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"threads={torch.get_num_threads()}"
    )

    print(
        f"completed={completed_episodes} "
        f"additional={config.additional_episodes} "
        f"target={target_episodes}"
    )

    print(
        f"max_steps={config.max_steps} "
        f"train_freq={config.train_freq} "
        f"gradient_steps={config.gradient_steps}"
    )

    print(
        f"csv={config.log_directory}/"
        f"{config.csv_name}"
    )

    environment = ContinuousCartPoleEnv(
        config=config,
        completed_episodes=completed_episodes,
        total_environment_steps=(
            total_environment_steps
        ),
    )

    model_file = Path(
        config.model_path + ".zip"
    )

    if model_file.exists():
        print("loading model")

        model = SAC.load(
            config.model_path,
            env=environment,
            device="cpu",
        )

        model.train_freq = TrainFreq(
            config.train_freq,
            TrainFrequencyUnit.STEP,
        )

        model.gradient_steps = (
            config.gradient_steps
        )

        model.batch_size = (
            config.batch_size
        )

        replay_buffer_path = Path(
            config.replay_buffer_path
        )

        if replay_buffer_path.exists():
            try:
                model.load_replay_buffer(
                    config.replay_buffer_path
                )
                print("replay buffer loaded")
            except Exception as error:
                print(
                    f"replay_buffer_load_error="
                    f"{error}"
                )

    else:
        print("creating model")

        model = SAC(
            policy="MlpPolicy",
            env=environment,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=config.tau,
            train_freq=(
                config.train_freq,
                "step",
            ),
            gradient_steps=(
                config.gradient_steps
            ),
            policy_kwargs={
                "net_arch": [
                    config.network_size,
                    config.network_size,
                ],
            },
            device="cpu",
            verbose=0,
            seed=config.seed,
        )

    callback = TrainingCallback(
        config=config,
        target_episodes=target_episodes,
    )

    maximum_timesteps = (
        config.additional_episodes
        * config.max_steps
    )

    try:
        model.learn(
            total_timesteps=maximum_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

    except KeyboardInterrupt:
        print("interrupted")

        save_state(
            model=model,
            environment=environment,
            config=config,
        )

        print(
            f"saved episode="
            f"{environment.completed_episodes}"
        )

    print(
        f"completed="
        f"{environment.completed_episodes}"
    )

    print(
        f"csv={config.log_directory}/"
        f"{config.csv_name}"
    )


if __name__ == "__main__":
    main()