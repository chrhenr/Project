import pyautogui
import time
import keyboard

top_left_corner = (0, 89)
bottom_right_corner = (1918, 1016)

small_map = (1660, 841)
middle_map = (1287, 568)


def screenshot():
    keyboard.press_and_release('win + shift + s')
    pyautogui.moveTo(*top_left_corner, duration=0.5)
    pyautogui.mouseDown()
    pyautogui.moveTo(*bottom_right_corner, duration=0.5)
    pyautogui.mouseUp()

def move_to_map():
    pyautogui.moveTo(small_map, duration=0.5)

def expand_map():
    pyautogui.moveTo(middle_map, duration=0.5)
    pyautogui.scroll(-200)
    time.sleep(0.5)
    pyautogui.scroll(-200)

def close_screenshot_tool():
    pyautogui.moveTo(1878, 884, duration=0.5)
    pyautogui.click()


# Left click at (640, 231)
# Left click at (1886, 963)
# Left click at (1287, 568)

if __name__ == "__main__":
    screenshot()
    time.sleep(1)
    close_screenshot_tool()
    time.sleep(1)
    move_to_map()
    time.sleep(1)
    expand_map()