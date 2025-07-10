import cv2
import imageio

gif_path = "mean.gif"
frames = imageio.mimread(gif_path)
idx = 0
n = len(frames)

# Labels from original script logic
def get_label(index):
    if index in [0, 6]:
        return "Original"
    elif 1 <= index <= 5:
        return "Interpolated"
    else:
        return "Forecast"

print("▶️ Use → (right arrow) and ← (left arrow) to step. ESC to exit.")

while True:
    frame = frames[idx]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    display = frame.copy()

    label = get_label(idx)
    frame_info = f"Frame {idx+1}/{n}"

    # Draw label and frame count (stacked)
    # cv2.putText(display, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(display, frame_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(f"Frame Viewer_{gif_path}", display)

    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == 81:  # Left arrow
        idx = max(0, idx - 1)
    elif key == 83:  # Right arrow
        idx = min(n - 1, idx + 1)

cv2.destroyAllWindows()