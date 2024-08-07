from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from torchvision.utils import draw_bounding_boxes
from motpy import Detection, MultiObjectTracker
import time
from QueueHolder import QueueHolder
import requests
import threading
import json
import os

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

settings_file = "settings.json"

xmin, xmax = 276, 513
ymin, ymax = 0, 480

distance_camera_center_robot_center = 400
picture_width_mm = 290
w_holder = 62

activatorOpenTime = 3

xcenter = int((xmax - xmin) / 2)
ycenter = int((ymax - ymin) / 2)

class_names = ['Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack', 'Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 'Food Can', 'Aerosol', 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton', 'Drink carton', 'Corrugated carton', 'Meal carton', 'Pizza box', 'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup', 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag', 'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 'Polypropylene bag', 'Crisp packet', 'Spread tub', 'Tupperware', 'Disposable food container', 'Foam food container', 'Other plastic container', 'Plastic gloves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw', 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette']

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

yolo_model = YOLO("best.pt")
yolo_model.to(device=DEVICE)

tracker = MultiObjectTracker(dt=1/30)

ROBOT_IP = "192.168.1.30"
ROBOT_STATUS = 42
ROBOT_LEFTARM = 43
ROBOT_RIGHTARM = 44

ROBOT_WAIT = 1
ROBOT_SORT = 2
ROBOT_WAIT_ACTIVATE = 3
ROBOT_ACTIVATE = 4
ROBOT_WAIT_RESET = 5
ROBOT_RESET = 6
ROBOT_END = 7

queueHolder = QueueHolder(xmin, xmax, ymin, ymax, w_holder, picture_width_mm, distance_camera_center_robot_center, activatorOpenTime, ROBOT_WAIT_ACTIVATE, ROBOT_ACTIVATE, ROBOT_WAIT_RESET, ROBOT_RESET)
queueHolder.setConveyorBeltSpeed(56)

CAP_PROP_BRIGHTNESS = 150
CAP_PROP_CONTRAST = 200

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_BRIGHTNESS, CAP_PROP_BRIGHTNESS)
vid.set(cv2.CAP_PROP_CONTRAST, CAP_PROP_CONTRAST)

stop_event = threading.Event()
final_image = None

frame = None
image_with_contours = None

conveyor_speed_thread = None
conveyor_speed_thread_stop_event = None
conveyor_speed = 56

def get_sector(bounding_box):
    areaA = (xcenter - bounding_box[0]) * (ycenter - bounding_box[1])
    areaB = (bounding_box[2] - xcenter) * (ycenter - bounding_box[1])
    areaC = (xcenter - bounding_box[0]) * (bounding_box[3] - ycenter)
    areaD = (bounding_box[2] - xcenter) * (bounding_box[3] - ycenter)

    areas = {'areaA': areaA, 'areaB': areaB, 'areaC': areaC, 'areaD': areaD}
    biggest_area = max(areas, key=areas.get)

    areaKeyToInt = {'areaA': 0, 'areaB': 1, 'areaC': 2, 'areaD': 3}
    return areaKeyToInt[biggest_area]

def generate_frames():
    global queueHolder
    global tracker
    global yolo_model
    global frame

    prev_time = time.time()

    while not stop_event.is_set():
        ret, frame = vid.read()

        if not ret:
            continue
        
        image_cv = frame[ymin:ymax, xmin:xmax]

        results = yolo_model.predict(source=image_cv, conf=0.6, device=0, verbose=False)

        image_cv = cv2.line(image_cv, (xcenter, 0), (xcenter, ymax - ymin), (0,0,255), 1)
        image_cv = cv2.line(image_cv, (0, ycenter), (xmax - xmin, ycenter), (0,0,255), 1)
        drawing_image = torch.from_numpy(image_cv).permute(2, 0, 1)

        detections = []
        for result in results:
            for data in result.boxes.data:
                detections.append(Detection(box=data[0:4].cpu().numpy(), score=data[4].cpu().numpy(), class_id=int(data[5].cpu().numpy())))
        
        tracker.step(detections=detections)
        tracks = tracker.active_tracks(min_steps_alive=3)

        dt = time.time() - prev_time
        prev_time = time.time()
        queueHolder.updateTrackingObjects(dt)
        drawing_image = torch.from_numpy(image_cv).permute(2, 0, 1)

        for track in tracks:
            box = track.box
            label = class_names[track.class_id]
            labels = [f"{label} {track.score:.2f} ID:{track.id}"]
            if label in ["Drink can"]:
                color = (0,255,255)
            elif label in ["Clear plastic bottle", "Other plastic bottle", "Plastic bottle cap"]:
                color = (0,0,255)
            else:
                color = (0,0,0)
                
            sector = get_sector(box)
            queueHolder.addObject(track.box, track.score, label, track.id, sector)
            drawing_image = draw_bounding_boxes(drawing_image, torch.from_numpy(np.array([track.box])), colors=[color] * len(torch.from_numpy(np.array([track.box]))), labels=labels, width=3, font="times new roman.ttf", font_size=15)
        
        final_image = drawing_image.numpy().transpose(1, 2, 0)
        _, buffer = cv2.imencode('.jpg', final_image)
        bufferBytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bufferBytes + b'\r\n')

def generate_camera_cut():
    global frame
    while not stop_event.is_set():
        if frame is None:
            continue
        
        if image_with_contours is None:
            image_cv = frame[ymin:ymax, xmin:xmax]
        else:
            image_cv = image_with_contours
            
        _, buffer = cv2.imencode('.jpg', image_cv)
        bufferBytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bufferBytes + b'\r\n')
        
def generate_camera_unedited():
    global frame
    while not stop_event.is_set():
        ret, frame = vid.read()

        if not ret:
            continue
        
        _, buffer = cv2.imencode('.jpg', frame)
        bufferBytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bufferBytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/nastavitve")
def settings():
    return render_template("settings.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera_cut")
def camera_cut():
    return Response(generate_camera_cut(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera_unedited")
def camera_unedited():
    return Response(generate_camera_unedited(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/robot_status')
def robot_status():
    r = requests.get(f'http://{ROBOT_IP}/KAREL/getreg?str_regnum={ROBOT_STATUS}')
    return jsonify({"status": r.text.strip()[0]})

@app.route("/remove_tracking_object", methods=['POST'])
def remove_tracking_object():
    global queueHolder
    data = request.get_json()
    id = data.get('id')
    queueHolder.removeTrackingObject(id)
    
    return "1"

@app.route('/get_detections')
def get_detections():
    try:
        detections = queueHolder.getTrackingObjects()
        return detections
    except Exception as e:
        return str(e)

@app.route("/get_data")
def getData():
    response = {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        "distance_camera_center_robot_center": distance_camera_center_robot_center,
        "picture_width_mm": picture_width_mm,
        "w_holder": w_holder,
        "ROBOT_IP": ROBOT_IP,
        "ROBOT_STATUS": ROBOT_STATUS,
        "ROBOT_LEFTARM": ROBOT_LEFTARM,
        "ROBOT_RIGHTARM": ROBOT_RIGHTARM,
        "CAP_PROP_BRIGHTNESS": CAP_PROP_BRIGHTNESS,
        "CAP_PROP_CONTRAST": CAP_PROP_CONTRAST,
        "conveyor_speed": conveyor_speed
    }
    return jsonify(response)

@app.route("/set_cut_coords", methods=['POST'])
def setCutCoords():
    global xmin
    global ymin
    global xmax
    global ymax

    global xcenter
    global ycenter

    data = request.get_json()
    xmin = data.get('xmin')
    ymin = data.get('ymin')
    xmax = data.get('xmax')
    ymax = data.get('ymax')

    xcenter = int((xmax - xmin) / 2)
    ycenter = int((ymax - ymin) / 2)

    return "1"

@app.route("/start_recording_conveyor_speed")
def start_recording_conveyor_speed():
    global conveyor_speed_thread
    global conveyor_speed_thread_stop_event

    conveyor_speed_thread_stop_event = threading.Event()
    conveyor_speed_thread = threading.Thread(target=calculate_conveyor_speed, args=(conveyor_speed_thread_stop_event,))
    conveyor_speed_thread.start()

    return "1"

@app.route("/stop_recording_conveyor_speed")
def stop_recording_conveyor_speed():
    global conveyor_speed_thread
    global conveyor_speed_thread_stop_event

    conveyor_speed_thread_stop_event.set()

    while conveyor_speed == None:
        pass

    conveyor_speed_thread.join()
    queueHolder.setConveyorBeltSpeed(conveyor_speed)

    return "1"

@app.route("/set_conveyor_speed", methods=['POST'])
def set_conveyor_speed():
    global conveyor_speed

    data = request.get_json()
    conveyor_speed = data.get('conveyor_speed')

    return "1"

@app.route("/set_distance_camera_center_robot_center", methods=['POST'])
def setDistanceCameraCenterRobotCenter():
    global distance_camera_center_robot_center

    data = request.get_json()
    distance_camera_center_robot_center = data.get('distance_camera_center_robot_center')

    return "1"

@app.route("/set_picture_width_mm", methods=['POST'])
def setPictureWidth():
    global picture_width_mm

    data = request.get_json()
    picture_width_mm = data.get('picture_width_mm')
    
    return "1"

@app.route("/set_w_holder", methods=['POST'])
def setWHolder():
    global w_holder

    data = request.get_json()
    w_holder = data.get('w_holder')
    
    return "1"

@app.route("/set_CAP_PROP_BRIGHTNESS", methods=['POST'])
def setCAP_PROP_BRIGHTNESS():
    global CAP_PROP_BRIGHTNESS

    data = request.get_json()
    CAP_PROP_BRIGHTNESS = data.get('CAP_PROP_BRIGHTNESS')

    if vid.isOpened():
        vid.set(cv2.CAP_PROP_BRIGHTNESS, CAP_PROP_BRIGHTNESS)
    
    return "1"

@app.route("/set_CAP_PROP_CONTRAST", methods=['POST'])
def setCAP_PROP_CONTRAST():
    global CAP_PROP_CONTRAST

    data = request.get_json()
    CAP_PROP_CONTRAST = data.get('CAP_PROP_CONTRAST')
    
    if vid.isOpened():
        vid.set(cv2.CAP_PROP_CONTRAST, CAP_PROP_CONTRAST)
    
    return "1"

@app.route("/set_ROBOT_IP", methods=['POST'])
def setROBOT_IP():
    global ROBOT_IP

    data = request.get_json()
    ROBOT_IP = data.get('ROBOT_IP')
    
    return "1"

@app.route("/set_ROBOT_STATUS", methods=['POST'])
def setROBOT_STATUS():
    global ROBOT_STATUS

    data = request.get_json()
    ROBOT_STATUS = data.get('ROBOT_STATUS')
    
    return "1"

@app.route("/set_ROBOT_LEFTARM", methods=['POST'])
def setROBOT_LEFTARM():
    global ROBOT_LEFTARM

    data = request.get_json()
    ROBOT_LEFTARM = data.get('ROBOT_LEFTARM')
    
    return "1"

@app.route("/set_ROBOT_RIGHTARM", methods=['POST'])
def setROBOT_RIGHTARM():
    global ROBOT_RIGHTARM

    data = request.get_json()
    ROBOT_RIGHTARM = data.get('ROBOT_RIGHTARM')
    
    return "1"

@app.route("/update_robot_status", methods=['POST'])
def set_robot_status():
    data = request.get_json()
    status = data.get('status')    
    requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_STATUS, status))

    return "1"

def robot_thread_func(stop_event):
    prev_loop_time = time.time()
    while not stop_event.is_set():
        if time.time() - prev_loop_time < 0.5:
            continue

        prev_loop_time = time.time()

        r = requests.get('http://' + ROBOT_IP + '/KAREL/getreg?str_regnum=' + str(ROBOT_STATUS))

        robot_status_value = r.text.strip()[0]
        #print("ROBOT_STATUS:", robot_status_value)
        if robot_status_value == str(ROBOT_WAIT):
            yoffset, leftarm, rightarm = queueHolder.getNextAction()

            if leftarm != -1 and rightarm != -1:
                requests.get('http://' + ROBOT_IP + '/KAREL/setprc?str_coord1=0.0&str_coord2={}&str_coord3=0.0&str_coord4=0.0&str_coord5=0.0&str_coord6=0.0'.format(yoffset))
                requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_LEFTARM, leftarm))
                requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_RIGHTARM, rightarm))
                requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_STATUS, ROBOT_SORT))
        elif robot_status_value == str(ROBOT_WAIT_ACTIVATE):
            robot_status_reg = queueHolder.getActivateAction()
            requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_STATUS, robot_status_reg))
        elif robot_status_value == str(ROBOT_WAIT_RESET):
            robot_status_reg = queueHolder.getRobotReset()
            requests.get('http://' + ROBOT_IP + '/KAREL/setreg?str_regnum={}&str_val={}'.format(ROBOT_STATUS, robot_status_reg))

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def calculate_conveyor_speed(stop_event):
    global frame
    global conveyor_speed
    global image_with_contours
    
    previous_center = None
    speeds = []
    prev_time = None

    while not stop_event.is_set():
        if frame is None:
            continue

        image_cv = frame[ymin:ymax,xmin:xmax]

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        
        high_thresh, thresholdedImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cannyImg = cv2.Canny(thresholdedImg, high_thresh * 0.5, high_thresh)

        contours, _ = cv2.findContours(cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []

        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue

            if len(approx) == 4 and area > 1500 and area < 2000:
                center = np.array(np.average(approx, axis=0)[0], dtype=int)
                shapes.append({"x": center[0], "y": center[1], "shape": "square"})
                image_with_contours = cv2.drawContours(image_cv, [contour], -1, (0,255,0), 3)
                
                if previous_center is not None:
                    distance = calculate_distance(previous_center, center)
                    if prev_time == None:
                        prev_time = time.time()
                        break
                    dt = time.time() - prev_time
                    prev_time = time.time()
                    speed = distance / dt
                    speeds.append(speed)

                previous_center = center
                break
    
    image_with_contours = None

    if speeds:
        conveyor_speed = sum(speeds) / len(speeds)
        print(f"Average Speed of the Conveyor Belt: {conveyor_speed:.2f} pixels per second")
    else:
        conveyor_speed = -1
        print("No movement detected to calculate speed.")


if __name__ == "__main__":
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as json_file:
            data = json.load(json_file)

        xmin = data['xmin']
        ymin = data['ymin']
        xmax = data['xmax']
        ymax = data['ymax']
        distance_camera_center_robot_center = data['distance_camera_center_robot_center']
        picture_width_mm = data['picture_width_mm']
        w_holder = data['w_holder']
        ROBOT_IP = data['ROBOT_IP']
        ROBOT_STATUS = data['ROBOT_STATUS']
        ROBOT_LEFTARM = data['ROBOT_LEFTARM']
        ROBOT_RIGHTARM = data['ROBOT_RIGHTARM']
        CAP_PROP_BRIGHTNESS = data['CAP_PROP_BRIGHTNESS']
        CAP_PROP_CONTRAST = data['CAP_PROP_CONTRAST']
        conveyor_speed = data['conveyor_speed']
        activatorOpenTime = data['activatorOpenTime']

        xcenter = int((xmax - xmin) / 2)
        ycenter = int((ymax - ymin) / 2)

        vid.set(cv2.CAP_PROP_BRIGHTNESS, CAP_PROP_BRIGHTNESS)
        vid.set(cv2.CAP_PROP_CONTRAST, CAP_PROP_CONTRAST)
        queueHolder = QueueHolder(xmin, xmax, ymin, ymax, w_holder, picture_width_mm, distance_camera_center_robot_center, activatorOpenTime, ROBOT_WAIT_ACTIVATE, ROBOT_ACTIVATE, ROBOT_WAIT_RESET, ROBOT_RESET)
        queueHolder.setConveyorBeltSpeed(conveyor_speed)

    robot_thread = threading.Thread(target=robot_thread_func, args=(stop_event,))
    robot_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        stop_event.set()
        robot_thread.join()
        vid.release()

        data = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            "distance_camera_center_robot_center": distance_camera_center_robot_center,
            "picture_width_mm": picture_width_mm,
            "w_holder": w_holder,
            "ROBOT_IP": ROBOT_IP,
            "ROBOT_STATUS": ROBOT_STATUS,
            "ROBOT_LEFTARM": ROBOT_LEFTARM,
            "ROBOT_RIGHTARM": ROBOT_RIGHTARM,
            "CAP_PROP_BRIGHTNESS": CAP_PROP_BRIGHTNESS,
            "CAP_PROP_CONTRAST": CAP_PROP_CONTRAST,
            "conveyor_speed": conveyor_speed,
            "activatorOpenTime": activatorOpenTime
        }

        with open(settings_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)