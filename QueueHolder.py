import time
import json

class QueueHolder:
    def __init__(self, xmin, xmax, ymin, ymax, w_holder, picture_width_mm, distance_between_camera_robot_mm, activatorOpenTime, ROBOT_WAIT_ACTIVATE, ROBOT_ACTIVATE, ROBOT_WAIT_RESET, ROBOT_RESET):
        self.conveyor_belt_speed = 0
        self.tracking_objects = []
        self.sectors = ["A", "B", "C", "D"]

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.camera_center = (xmax - xmin) / 2
        self.w_holder = w_holder
        self.px_to_mm_scale = picture_width_mm / (xmax - xmin)
        self.distance_between_camera_robot_mm = distance_between_camera_robot_mm

        self.activatorOpenTime = activatorOpenTime

        self.ROBOT_WAIT_ACTIVATE = ROBOT_WAIT_ACTIVATE
        self.ROBOT_ACTIVATE = ROBOT_ACTIVATE
        self.ROBOT_WAIT_RESET = ROBOT_WAIT_RESET
        self.ROBOT_RESET = ROBOT_RESET

        self.nearest_object_index = -1
        self.y2prev = 0
        self.path_length = 0
        self.traveled_path = 0
        self.yoffset = 0
        self.sendtime = 0
        
    def setConveyorBeltSpeed(self, conveyor_belt_speed):
        self.conveyor_belt_speed = conveyor_belt_speed
    
    def setActivatorOpenTime(self, activatorOpenTime):
        self.activatorOpenTime = activatorOpenTime

    def addObject(self, box, score, label, id, sector):
        for i, object in enumerate(self.tracking_objects):
            if object["id"] == id:
                self.tracking_objects[i]["box"] = box
                self.tracking_objects[i]["sector"] = sector
                return
        
        self.tracking_objects.append({"box": box, "score": score, "label": label, "id": id, "sector": sector})
    
    def updateTrackingObjects(self, delta_time):
        removeIndicis = []
        for i, object in enumerate(self.tracking_objects):
            object["box"][1] = object["box"][1] - self.conveyor_belt_speed * delta_time
            object["box"][3] = object["box"][3] - self.conveyor_belt_speed * delta_time

            if abs(object["box"][3] - self.camera_center) * self.px_to_mm_scale > self.distance_between_camera_robot_mm:
                removeIndicis.append(i)

        removeIndicis = sorted(removeIndicis, reverse=True)

        for index in removeIndicis:
            self.tracking_objects.pop(index)


    def getNextAction(self):
        yoffset = 0
        leftarm = 0
        rightarm = 0

        if len(self.tracking_objects) == 0 or self.nearest_object_index != -1:
            return (-1, -1, -1)

        nearest_object = None
        nearest_object_y1 = self.ymax
        nearest_object_index = -1

        for i, object in enumerate(self.tracking_objects):
            if object["box"][1] < nearest_object_y1:
                nearest_object = object
                nearest_object_y1 = object["box"][1]
                nearest_object_index = i
        
        if nearest_object["label"] in ["Drink can"]:
            rightarm = 1
            yoffset = (nearest_object["box"][0] - self.camera_center) * self.px_to_mm_scale - self.w_holder / 2
        elif nearest_object["label"] in ["Clear plastic bottle", "Other plastic bottle"]:
            leftarm = 1
            yoffset = (nearest_object["box"][2] - self.camera_center) * self.px_to_mm_scale + self.w_holder / 2

        self.yoffset = yoffset
        self.y2prev = nearest_object["box"][3]
        self.path_length = (self.y2prev - (self.ymax - self.ymin) / 2) * self.px_to_mm_scale + self.distance_between_camera_robot_mm
        self.traveled_path = 0
        self.sendtime = time.time()
        self.nearest_object_index = nearest_object_index

        return (yoffset, leftarm, rightarm)
    
    def getActivateAction(self):
        #lahko bi preverjal z drugo kamero poševno nad trakom
        #lahko bi preverjal z drugo in tretjo kamero nad območjem spusta za plastenke in pločevinke
        if self.nearest_object_index != -1 and len(self.tracking_objects) > self.nearest_object_index:
            self.traveled_path += abs(self.tracking_objects[self.nearest_object_index]["box"][3] - self.y2prev)
            self.y2prev = self.tracking_objects[self.nearest_object_index]["box"][3]

            print(f"Prepotovana pot: {self.traveled_path * self.px_to_mm_scale:.2f}\tCela pot: {self.distance_between_camera_robot_mm:.2f}\tDelež: {self.traveled_path * self.px_to_mm_scale / self.distance_between_camera_robot_mm * 100:.2f} %")
            if self.traveled_path * self.px_to_mm_scale >= self.distance_between_camera_robot_mm:
                self.tracking_objects.pop(self.nearest_object_index)
                self.nearest_object_index = -1
                self.sendtime = time.time()
                return self.ROBOT_ACTIVATE
        else:
            return self.ROBOT_RESET
            
        return self.ROBOT_WAIT_ACTIVATE
        
    def getRobotReset(self):
        if time.time() - self.sendtime >= self.activatorOpenTime:
            return self.ROBOT_RESET
        else:
            return self.ROBOT_WAIT_RESET
    
    def removeTrackingObject(self, id):
        index = -1
        for i, tracking_object in enumerate(self.tracking_objects):
            if tracking_object["id"] == id:
                index = i

        if index != -1:
            self.tracking_objects.pop(index)
        
    def print_tracking_objects(self):
        for object in self.tracking_objects:
            print("box:", object["box"], "\tscore:", object["score"], "\tlabel:", object["label"], "\tid:", object["id"], "\tsector:", object["sector"])
    
    def getTrackingObjects(self):
        retObjects = []

        for obj in self.tracking_objects:
            retObjects.append({"id": obj["id"], "label": obj["label"], "score": round(obj["score"], 2), "sector": obj["sector"], "box": str(round(obj["box"][0],2)) + "," + str(round(obj["box"][1],2)) + "    " + str(round(obj["box"][2],2)) + "," + str(round(obj["box"][3],2))})

        # Serialize to JSON
        return json.dumps(retObjects)