from serial.tools import list_ports
from pydobot import Dobot
from find_points import calcPoints

port = list_ports.comports()[3].device
device = Dobot(port=port)


def print_coords(p):
    print("x: ", p[0])
    print("y: ", p[1])
    print("z: ", p[2])
    print("r: ", p[3])


def get_x():
    return device.pose()[0]


def get_y():
    return device.pose()[1]


def get_z():
    return device.pose()[2]


def get_r():
    return device.pose()[3]


def draw_line(x1, y1, x2, y2):
    if 200 <= x1 <= 300 and 200 <= x2 <= 300 and -50 <= y1 <= 50 and -50 <= y2 <= 50:
        device.move_to(x1, y1, zUp, 0, wait=True)
        device.move_to(x1, y1, zDown, 0, wait=True)
        device.move_to(x2, y2, zDown, 0, wait=True)
        device.move_to(x2, y2, zUp, 0, wait=True)
    else:
        print("Coordinates not in range!")


def draw_by_nums(points):
    points = calcPoints(points)
    for x in range(0, len(points) - 1):
        print("drawing:", points[x][0], points[x][1], points[x + 1][0], points[x + 1][1])
        draw_line(points[x][0], points[x][1], points[x + 1][0], points[x + 1][1])


pose = device.pose()

x = get_x()
y = get_y()
z = get_z()
r = get_r()

zDown = -50
zUp = -35

device.speed(100, 100)

device.move_to(200, 0, zUp, 0, wait=True)  # Starting Position

input_points = [[89.72, 68.06], [55.0, 85.56], [19.54, 68.89]]

draw_by_nums(input_points)
