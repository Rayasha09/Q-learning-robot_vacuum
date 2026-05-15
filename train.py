import random
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame


GRID_SIZE = 6
CELL = 95
PANEL = 320
WIDTH = GRID_SIZE * CELL + PANEL
HEIGHT = GRID_SIZE * CELL

EPISODES = 1000
MAX_STEPS = 80
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.992

TRAIN_SPEED = 8
DEMO_SPEED = 2

LEVEL = "easy"
Q_TABLE_FILE = Path("q_table.npy")
LEVELS = {
    "easy": {
        "obstacles": {(2, 2), (4, 1)},
        "dirt_spots": [(0, 5), (2, 0), (5, 1), (5, 5)],
        "start": (0, 0),
        "rewards": {"wall": -15, "obstacle": -20, "step": -1, "clean": 35, "finish": 140},
    },
    "medium": {
        "obstacles": {(1, 2), (3, 3), (4, 1)},
        "dirt_spots": [(0, 5), (2, 0), (2, 4), (5, 1), (5, 5)],
        "start": (0, 0),
        "rewards": {"wall": -30, "obstacle": -40, "step": -1, "clean": 35, "finish": 150},
    },
    "hard": {
        "obstacles": {(1, 1), (1, 4), (2, 2), (3, 3), (4, 1), (4, 4)},
        "dirt_spots": [(0, 5), (1, 3), (2, 0), (3, 5), (5, 1), (5, 5)],
        "start": (0, 0),
        "rewards": {"wall": -35, "obstacle": -45, "step": -2, "clean": 30, "finish": 160},
    },
}

CFG = LEVELS[LEVEL]
OBSTACLES = CFG["obstacles"]
DIRT_SPOTS = CFG["dirt_spots"]
START = CFG["start"]
REWARDS = CFG["rewards"]

q_table = np.zeros((GRID_SIZE * GRID_SIZE * (2 ** len(DIRT_SPOTS)), 4))
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG = (242, 245, 247)
GRID = (190, 198, 205)
WALL = (180, 65, 65)
TEXT = (35, 40, 45)
ROBOT = (55, 125, 255)
DIRTY = (234, 201, 112)
CLEAN = (204, 237, 210)
SIDE = (226, 232, 236)
DARK = (75, 80, 85)
GREEN = (55, 150, 90)


def state_id(pos, mask):
    return (pos[0] * GRID_SIZE + pos[1]) * (2 ** len(DIRT_SPOTS)) + mask


def update_mask(pos, mask):
    for i, dirt in enumerate(DIRT_SPOTS):
        if pos == dirt:
            mask |= 1 << i
    return mask


def step(pos, mask, action):
    dr, dc = MOVES[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return pos, mask, REWARDS["wall"], False, "wall"
    nxt = (nr, nc)
    if nxt in OBSTACLES:
        return pos, mask, REWARDS["obstacle"], False, "obstacle"
    reward = REWARDS["step"]
    new_mask = mask
    if nxt in DIRT_SPOTS:
        old = new_mask
        new_mask = update_mask(nxt, new_mask)
        if new_mask != old:
            reward = REWARDS["clean"]
    done = new_mask == (2 ** len(DIRT_SPOTS)) - 1
    if done:
        reward = REWARDS["finish"]
    return nxt, new_mask, reward, done, None


def draw(screen, font, big_font, pos, mask, ep, total, eps, mode, hit=None):
    screen.fill(BG)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            color = (250, 250, 250)
            if (r, c) in OBSTACLES:
                color = WALL
            elif (r, c) in DIRT_SPOTS:
                i = DIRT_SPOTS.index((r, c))
                color = CLEAN if mask & (1 << i) else DIRTY
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID, rect, 1)
            if (r, c) in DIRT_SPOTS and not mask & (1 << DIRT_SPOTS.index((r, c))):
                pygame.draw.circle(screen, DARK, (c * CELL + CELL // 2, r * CELL + CELL // 2), 10)

    x, y = pos[1] * CELL + CELL // 2, pos[0] * CELL + CELL // 2
    pygame.draw.circle(screen, ROBOT, (x, y), CELL // 3)
    pygame.draw.circle(screen, DARK, (x, y), CELL // 3, 4)
    pygame.draw.circle(screen, DARK, (x - 12, y - 8), 4)
    pygame.draw.circle(screen, DARK, (x + 12, y - 8), 4)
    pygame.draw.arc(screen, DARK, pygame.Rect(x - 16, y - 2, 32, 18), 0.2, 3.0, 3)

    pygame.draw.rect(screen, SIDE, (GRID_SIZE * CELL, 0, PANEL, HEIGHT))
    tx = GRID_SIZE * CELL + 20
    screen.blit(big_font.render("Robot Vacuum", True, TEXT), (tx, 25))
    screen.blit(font.render(mode, True, GREEN), (tx, 80))
    screen.blit(font.render(f"Level: {LEVEL}", True, TEXT), (tx, 140))
    screen.blit(font.render(f"Episode: {ep}", True, TEXT), (tx, 185))
    screen.blit(font.render(f"Reward: {total}", True, TEXT), (tx, 230))
    screen.blit(font.render(f"Epsilon: {eps:.3f}", True, TEXT), (tx, 275))
    screen.blit(font.render(f"Cleaned: {bin(mask).count('1')}/{len(DIRT_SPOTS)}", True, TEXT), (tx, 320))
    if hit == "wall":
        screen.blit(font.render("Penalty: wall", True, WALL), (tx, 380))
    if hit == "obstacle":
        screen.blit(font.render("Penalty: obstacle", True, WALL), (tx, 380))
    pygame.display.flip()


def frame(screen):
    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))


def run_demo(screen, clock, font, big_font):
    gif_frames = []
    pos, mask, total = START, update_mask(START, 0), 0
    for _ in range(MAX_STEPS):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        s = state_id(pos, mask)
        action = int(np.argmax(q_table[s]))
        pos, mask, reward, done, hit = step(pos, mask, action)
        total += reward
        draw(screen, font, big_font, pos, mask, EPISODES, total, 0.0, "Demo", hit)
        gif_frames.append(frame(screen))
        clock.tick(DEMO_SPEED)
        if done:
            for _ in range(8):
                gif_frames.append(frame(screen))
            break

    imageio.mimsave("robot_vacuum.gif", gif_frames, duration=0.3)


def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26)
    big_font = pygame.font.SysFont("arial", 34, bold=True)

    if Q_TABLE_FILE.exists():
        global q_table
        q_table = np.load(Q_TABLE_FILE)
        run_demo(screen, clock, font, big_font)
        pygame.quit()
        return

    epsilon = EPSILON
    rewards = []
    for ep in range(1, EPISODES + 1):
        pos, mask, total = START, update_mask(START, 0), 0
        for _ in range(MAX_STEPS):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
            s = state_id(pos, mask)
            action = random.randint(0, 3) if random.random() < epsilon else int(np.argmax(q_table[s]))
            nxt, nxt_mask, reward, done, hit = step(pos, mask, action)
            ns = state_id(nxt, nxt_mask)
            q_table[s, action] += ALPHA * (reward + GAMMA * np.max(q_table[ns]) - q_table[s, action])
            pos, mask, total = nxt, nxt_mask, total + reward
            if ep <= 15 or ep % 40 == 0 or ep > EPISODES - 80:
                draw(screen, font, big_font, pos, mask, ep, total, epsilon, "Training", hit)
                clock.tick(TRAIN_SPEED)
            if done:
                break
        rewards.append(total)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    np.save (Q_TABLE_FILE, q_table)
    run_demo(screen, clock, font, big_font)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color="steelblue", label="Reward")
    smooth = [np.mean(rewards[max(0, i - 29):i + 1]) for i in range(len(rewards))]
    plt.plot(smooth, color="darkgreen", linewidth=2.5, label="Moving average")
    plt.title("Learning Curve")
    plt.xlabel("es")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()
    pygame.quit()


if __name__ == "__main__":
    train()
