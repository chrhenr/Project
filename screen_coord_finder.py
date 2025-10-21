# screen_coord_finder.py
from pynput import mouse
import sys
import traceback

def on_click(x, y, button, pressed):
    try:
        # only react on press of left button
        if pressed and button == mouse.Button.left:
            print(f"Left click at ({x}, {y})", flush=True)
    except Exception:
        print("Exception in on_click:", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting mouse listener. Left-click anywhere to print coordinates.", flush=True)
    print("Press Ctrl+C in this terminal to stop.", flush=True)

    try:
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nUser requested stop. Exiting.")
    except Exception as e:
        print("Listener crashed with exception:", e, file=sys.stderr)
        traceback.print_exc()


