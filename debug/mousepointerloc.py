from pynput.mouse import Listener

def on_click(x, y, button, pressed):
    if button == button.left and pressed:
        print(f"Left Mouse Button Clicked - X: {x}, Y: {y}")

# Create a listener
with Listener(on_click=on_click) as listener:
    listener.join()
