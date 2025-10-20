import threading
import time
import sys

def timer(stop_event):
    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        sys.stdout.write(f"\r⏱️   Elapsed time: {elapsed:.2f} seconds")
        sys.stdout.flush()
    print()

def long_running_task():
    # Simulate a long-running process
    for i in range(5):
        time.sleep(3)
    print("\nTask complete!")

def run_with_timer(func):

    stop_event = threading.Event()
    t = threading.Thread(target=timer, args=(stop_event,))
    t.start()

    try:
        return func()
    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nProcess interrupted by user.")
        else:
            print(f"\nAn error occurred: {e}")
    finally:
        stop_event.set()
        t.join()
        print("Timer stopped.")

if __name__ == "__main__":
    run_with_timer(long_running_task)