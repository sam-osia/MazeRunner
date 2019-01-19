import matplotlib.pyplot as plt
import numpy as np
import math
from math import sin, cos, radians


# matplotlib handles points as (y, x), so the equations have y and x reversed
def get_pixels(lines):
    all_pixels = []
    for line in lines:
        pt1 = line[0]
        pt2 = line[1]
        m = (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
        b = pt1[0] - m * (pt1[1])
        # print(f'y={m}x+{b}')
        eq = lambda a: (m * a + b)
        x = np.linspace(pt1[1], pt2[1], abs(pt1[1] - pt2[1]) * 10)
        y = np.floor(eq(x))
        x = np.floor(x)
        pixels = np.unique(np.array(list(zip(x, y)), dtype='int_'), axis=0)
        all_pixels.append(pixels)
    all_pixels = [item for sublist in all_pixels for item in sublist]
    return all_pixels


def draw_point(game, point):
    for x in range(point[0] - 1, point[0] + 2):
        for y in range(point[1] - 1, point[1] + 2):
            game[x][y] = 255
    return game


# angle: 0° = E, 90° = N, 180° = W, 270° = S
def get_distance(position, obstacles, angle, heading):
    closest_point = None
    step = 0
    X = position[1]
    Y = position[0]

    while closest_point is None:
        y = math.floor(X + step * cos(radians(angle + heading)))
        x = math.floor(Y + step * sin(radians(angle + heading)))
        step += 0.1
        if obstacles[x][y] == 255:
            closest_point = [x, y]
    # print(f'my point: {position}')
    # print(f'closest point: {closest_point}')
    distance = math.sqrt(((Y - closest_point[0]) ** 2 + (X - closest_point[1]) ** 2))
    return distance


def laser_scan(game, player, angles, move, heading):
    distances = []
    for angle in angles:
        distances.append(get_distance(player, game, angle, heading))
    return distances


def record_data(distances, move):
    with open('data.txt', 'a') as f:
        for i in range(len(distances)):
            if i == len(distances) - 1:
                f.write(str(distances[i]) + '\t')
            else:
                f.write(str(distances[i]) + ',')
        f.write(move + '\n')


def draw_player(pos, game, color=100):
    for x in range(pos[0] - 1, pos[0] + 2):
        for y in range(pos[1] - 1, pos[1] + 2):
            game[x][y] = color
    return


def get_player_pos(pos, move, game):
    # black out current position
    draw_player(pos, game, color=0)
    if move == 'up':
        pos[0] -= 1
    elif move == 'down':
        pos[0] += 1
    elif move == 'right':
        pos[1] += 1
    elif move == 'left':
        pos[1] -= 1
    else:
        print('invalid move')
    return pos


def on_key(event):
    global last_key
    last_key = event.key


# create game map and put boundaries on the edges
game_map = np.zeros([100, 100])
game_map[0][:] = 255
game_map[-1][:] = 255
game_map.T[0][:] = 255
game_map.T[-1][:] = 255

# connect signal to figure for event handling
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)

player = [5, 5]
scan_angles = np.linspace(-180, 180, 37)
print(scan_angles)
draw_player(player, game_map)

key_list = ['up', 'down', 'left', 'right']

# tracking heading angle
heading_angle = {'up': 180, 'right': 90, 'down': 0, 'left': 270}

# a list of points pairs to connect => point pair format: [[x1, y1], [x2, y2]]
line_list = [[[20, 80], [15, 20]],
             [[15, 20], [75, 10]],
             [[75, 10], [60, 65]],
             [[60, 65], [25, 85]],
             [[98, 40], [45, 98]],
             [[5, 86], [10, 92]]]

pixels = np.array(get_pixels(line_list))
for pixel in pixels:
    game_map = draw_point(game_map, pixel)

# enable interactive plot and show before entering the game
plot = plt.imshow(game_map, cmap='gray')
plt.ion()
plt.show()

last_key = 'up'
distances = laser_scan(game_map, player, scan_angles, last_key, heading_angle[last_key])
record_data(distances, last_key)
while True:
    plt.pause(0.000001)
    if last_key != '':
        if last_key in key_list:
            player = get_player_pos(player, last_key, game_map)
            draw_player(player, game_map)
            plt.imshow(game_map, cmap='gray')
            plt.draw()
            record_data(distances, last_key)
            distances = laser_scan(game_map, player, scan_angles, last_key, heading_angle[last_key])
            print(list(zip(scan_angles, distances)))
        last_key = ''
