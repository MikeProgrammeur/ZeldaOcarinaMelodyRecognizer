
import os

# Allow controller events while this process is in the background.
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import time
import pygame
import pyautogui
import json

# ============================================================
# CONFIGURATION
# ============================================================

JSON_PATH = "keyConfiguration.json"

json_is_complete = True

n64_to_3ds_dict = {"A": "L", "Up": "A", "Right": "Y", "Down": "R", "Left": "X"}

try:
    with open(JSON_PATH, "r") as f:
        button_to_key_dic = json.load(f)
        for e in ['A', 'Up', 'Right', 'Down', 'Left']:
            if button_to_key_dic.get(e) is None:
                json_is_complete = False
        button_to_key_dic = {n64_to_3ds_dict[k]:button_to_key_dic[k] for k in ['A', 'Up', 'Right', 'Down', 'Left'] if k in button_to_key_dic}
except:   
    print(f"Opening the JSON file has been unsuccessull please verify that the following path is correct {JSON_PATH}")
    quit()

if not json_is_complete:
    print(f"Opening the JSON file has been successull but the content is not in the right format")
    quit()
    
POLL_INTERVAL = 0.005

# Xbox controller button -> keyboard key
# Only these five buttons are remapped.
BUTTON_MAPPING = {
    0: button_to_key_dic["A"],       # A(0) from the xbox controller mapped to A from 3DS
    2: button_to_key_dic["Y"],       # X(2) from the xbox controller mapped to Y from 3DS
    3: button_to_key_dic["X"],       # Y(3) from the xbox controller mapped to X from 3DS
} # !!WARNING!! here X controls Y and X controls Y, because their are inverted between Xbox and 3DS, if you don't like that, modify the two lines above

# Xbox triggers (analog axes) -> keyboard key
# Standard SDL Xbox mapping:
# Axis 4 = LT, Axis 5 = RT
TRIGGER_MAPPING = {
    4: button_to_key_dic["L"],       # LT(4) from the xbox controller mapped to L from 3DS
    5: button_to_key_dic["R"],       # RT(5) from the xbox controller mapped to R from 3DS
}

TRIGGER_THRESHOLD = 0.5

# ============================================================
# INITIALIZATION
# ============================================================

pygame.init()
pygame.joystick.init()

pyautogui.PAUSE = 0

joystick = None
pressed_keys = set()
trigger_states = {axis: False for axis in TRIGGER_MAPPING}


def press_key(key):
    if key and key not in pressed_keys:
        pyautogui.keyDown(key)
        pressed_keys.add(key)


def release_key(key):
    if key and key in pressed_keys:
        pyautogui.keyUp(key)
        pressed_keys.remove(key)


def release_all():
    for key in list(pressed_keys):
        release_key(key)

    for axis in trigger_states:
        trigger_states[axis] = False


def connect_controller():
    global joystick

    if pygame.joystick.get_count() == 0:
        return False

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Controller connected: {joystick.get_name()}")
    return True


def disconnect_controller():
    global joystick

    release_all()

    if joystick is not None:
        joystick.quit()
        joystick = None

    print("Controller disconnected")


# ============================================================
# MAIN LOOP
# ============================================================

print("Controller-to-keyboard bridge started.")
print("Keep Citra as the active window.")
print("Press Ctrl+C in this terminal to stop.")

try:
    while True:

        if joystick is None:
            if not connect_controller():
                pygame.time.wait(500)
                continue

        for event in pygame.event.get():

            if event.type == pygame.JOYDEVICEREMOVED:
                disconnect_controller()
                continue

            # Physical buttons: A, X, Y only
            elif event.type == pygame.JOYBUTTONDOWN:
                key = BUTTON_MAPPING.get(event.button)
                press_key(key)

            elif event.type == pygame.JOYBUTTONUP:
                key = BUTTON_MAPPING.get(event.button)
                release_key(key)

        # Analog triggers: LT and RT
        if joystick is not None:
            for axis, key in TRIGGER_MAPPING.items():

                value = joystick.get_axis(axis)
                is_pressed = value > TRIGGER_THRESHOLD

                if is_pressed and not trigger_states[axis]:
                    press_key(key)

                elif not is_pressed and trigger_states[axis]:
                    release_key(key)

                trigger_states[axis] = is_pressed

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    release_all()
    pygame.quit()
    print("Stopped.")
 