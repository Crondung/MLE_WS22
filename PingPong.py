import pygame
import random
import numpy as np

MAX_X_BALL = 11 + 1
MAX_Y_BALL = 12 + 1
MAX_V_X_BALL = 1 + 1
MAX_V_Y_BALL = 1 + 1
MAX_X_RACKET = 11 + 1
MAX_ARR = [MAX_X_RACKET, MAX_X_BALL, MAX_Y_BALL, MAX_V_X_BALL, MAX_V_Y_BALL]
EPSILON = 0.001
ALPHA = 0.05  # Higher alpha values make Q-values change faster
GAMMA = 0.9  # Higher gamma looks in broader future


# possible actions = -1: move left, 0: stay, 1: move right


def get_state(x: np.ndarray, max: np.ndarray = MAX_ARR):
    s = x[0]  # + max[0]
    for i in range(1, len(x)):
        s = s * max[i] + x[i]
    return s


def select_action(state: int, q_table: np.ndarray, epsilon: float = EPSILON):
    q_s = q_table[state]
    n = random.random()
    best_action = np.argmax(q_s)
    action = best_action
    if n <= epsilon:
        print('chose epsilon')
        actions = [0, 1, 2]
        actions.pop(best_action)
        action = random.choice(actions)
    """
    actions are as following in q:
    q_s[0] = move left, which is an x change of 0
    q_s[1] = move right, which is an x change of 1
    q_s[2] = move left, which is an x change of -1
    """
    return -1 if action == 2 else action


def update_q_table(q_table: np.ndarray, state: int, next_state: int, action: int, reward: int, gamma: float = GAMMA, alpha: float = ALPHA):
    q_table[state][action] = q_table[state][action] + alpha * (reward * gamma * max(q_table[next_state]) - q_table[state][action])


# initialize Q Table with all possible states
q_table = np.random.rand(get_state(MAX_ARR), 3)

pygame.init()
screen = pygame.display.set_mode((240, 260))
pygame.display.set_caption("Ping-Pong")

x_racket = 5
x_ball = 1
y_ball = 1
vx_ball = 1
vy_ball = 1
clock = pygame.time.Clock()
continueGame = True
score = 0
font = pygame.font.SysFont("arial", 20)

while continueGame:
    screen.fill((0, 0, 0))

    initial_state = get_state([x_racket, x_ball, y_ball, vx_ball, vy_ball])

    action = select_action(initial_state, q_table)

    if (x_racket + 4 + action <= 12) & (x_racket + action >= 0):
        x_racket += action
    text = font.render("score:" + str(score), True, (255, 255, 255))
    textrect = text.get_rect()
    textrect.centerx = screen.get_rect().centerx
    screen.blit(text, textrect)
    pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(x_racket * 20, 250, 80, 10))
    pygame.draw.rect(screen, (255, 100, 0), pygame.Rect(x_ball * 20, y_ball * 20, 20, 20))

    x_ball = x_ball + vx_ball
    y_ball = y_ball + vy_ball

    if x_ball > 10 or x_ball < 1:
        vx_ball *= -1
    if y_ball > 11 or y_ball < 1:
        vy_ball *= -1

    reward = 0
    if y_ball == 12:
        if (x_ball >= x_racket and x_ball <= x_racket + 4):
            score = score + 1
            reward = +1
        else:
            score = score - 1
            reward = -1

    next_state = get_state([x_racket, x_ball, y_ball, vx_ball, vy_ball])
    update_q_table(q_table, initial_state, next_state, action, reward)

    pygame.display.flip()
    clock.tick(30)  # Refresh-Zeiten festlegen 60 FPS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("ende")
            pygame.quit()
            continueGame = False
