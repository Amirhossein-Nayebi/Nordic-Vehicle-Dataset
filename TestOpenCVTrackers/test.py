import cv2

# Load the video file
video = cv2.VideoCapture("./Videos/2022-12-02_Asjo_01.MP4")
# Get the frame rate of the video
frame_rate = video.get(cv2.CAP_PROP_FPS)
# Calculate the frame index from the frame time
startFrame = int(47 * frame_rate)

video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
# Select the object to track
success, frame = video.read()
# frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bbox = cv2.selectROI("Tracking", frame, False)


# Initialize the tracker
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

    
tracker.init(frame, bbox)

# Run the tracker on the video frames
while True:
    success, frame = video.read()
    if not success:
        break
    # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Update the tracker
    success, bbox = tracker.update(frame)
    
    # Draw the bounding box on the frame
    if success:
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(-1) & 0xFF == ord("q"):
        break

# Release the video file and close all windows
video.release()
cv2.destroyAllWindows()
