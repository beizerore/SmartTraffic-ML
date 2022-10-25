#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

video_path   = "./IMAGES/channel09-simpang_kodim.mp4"

def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    width = int(width)
    height = int(height)

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    cent_tracked_past = {}

    sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor
    ret, frame = vid.read()  # import image
    ratio = 1.0  # resize ratio
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    width2, height2, channels = image.shape

    count_cars = {}
    count_bike = {}

    while True:
        ret, frame = vid.read()

        con_cent = []

        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
            # cv2.imshow("image", image) #@
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            # cv2.imshow("gray", gray) #@
            fgmask = sub.apply(gray)  # uses the background subtraction
            # cv2.imshow("fgmask", fgmask) #@
            # applies different thresholds to fgmask to try and isolate cars
            # just have to keep playing around with settings until cars are easily identifiable
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow("closing", closing) #@
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            # cv2.imshow("opening", opening) #@
            dilation = cv2.dilate(opening, kernel)
            # cv2.imshow("dilation", dilation) #@
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            # cv2.imshow("retvalbin", retvalbin) #@
            # creates contours
            # cv2.imshow('bins',bins)
            contours, hierarchy = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # print(hierarchy)
            minarea = 1000
            # max area for contours, can be quite large for buses
            maxarea = 50000
            # vectors for the x and y locations of contour centroids in current frame
            cxx = np.zeros(len(contours))
            cyy = np.zeros(len(contours))
        
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        # con_cent.append((cx, cy))
                        # print(con_cent)

                        # for i, track in enumerate(tracker.tracks):
                        #     if not track.is_confirmed() or track.time_since_update > 5:
                        #         continue 
                        #     bbox = track.to_tlbr() # Get the corrected/predicted bounding box
                        #     class_name = track.get_class() #Get the class name of particular object
                        #     tracking_id = track.track_id # Get the ID for the particular track
                        #     # print(class_name)
                        #     index = key_list[val_list.index(class_name)] # Get predicted object index by object name
                        #     tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
                        #     if class_name == "car":
                        #         count_cars.append(tracking_id)
                        #     elif class_name == "motorbike":
                        #         count_bike.append(tracking_id)

                        #     list_bbox = bbox.tolist()
                        #     x, y, w, h = list_bbox
                        #     cx = int((x+w) / 2.0)
                        #     cy = int((y+h) / 2.0)
                        #     # print(f"bbox centroid: x:{x}, y:{y}, w:{w}, h:{h}")
                        #     # print(f"bbox centroid: {cx}, {cy}")

                        #     # for centroid in con_cent:
                        #     #     if centroid[0] 

                        #     cent_tracked.append((cx, cy))

                        # print(f"contour centroid: {cx}, {cy}")
                    
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # print(f"contour centroid: x:{x}, y:{y}, w:{w}, h:{h}")
                        
                        # hull = [cv2.convexHull(c) for c in contours]
                        # cv2.drawContours(image, hull, -1, (0, 255, 0), 2)
                        
                        # creates a rectangle around contour
                        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
        
        # cv2.imshow("countours", image)

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        cent_tracked = []
        # count_cars = []
        # count_bike = []
        count_mot = {}

        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            # print(class_name)
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            if class_name == "car":
                count_cars[tracking_id] = tracking_id
            elif class_name == "motorbike":
                count_bike[tracking_id] = tracking_id

            list_bbox = bbox.tolist()
            x, y, w, h = list_bbox
            cx = int((x+w) / 2.0)
            cy = int((y+h) / 2.0)
            # print(f"bbox centroid: x:{x}, y:{y}, w:{w}, h:{h}")
            # print(f"bbox centroid: {cx}, {cy}")

            # for centroid in con_cent:
            #     if centroid[0] 

            cent_tracked.append((cx, cy))
            try:
                c_past = cent_tracked_past[tracking_id]
                print(c_past, tracking_id)
                if cx > c_past[0] or cy > c_past[1]:
                    print(f"{tracking_id}, {class_name} is in motion")
                    cent_tracked_past[tracking_id] = (cx, cy)
                    if tracking_id not in count_mot:
                        count_mot[tracking_id] = (cx, cy)
            except:
                cent_tracked_past[tracking_id] = (cx, cy)

        # print(cent_tracked_past)
        # draw detection on frame

        image = draw_bbox(original_frame, tracked_bboxes, cent_tracked, CLASSES=CLASSES, tracking=True)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        firetruck = []
        ambulance = []

        image = cv2.rectangle(image, (10, 40), (210, 170), (255, 255, 255), -1)
        image = cv2.putText(image, "Cars: {}".format(len(count_cars)), (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)
        image = cv2.putText(image, "Motorbike: {}".format(len(count_bike)), (10, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)
        image = cv2.putText(image, "Ambulance: {}".format(len(ambulance)), (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)
        image = cv2.putText(image, "Fire Truck: {}".format(len(firetruck)), (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)
        image = cv2.putText(image, "Vehicle In Motion: {}".format(len(count_mot)), (10, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)
        image = cv2.putText(image, "Vehicle In Rest: {}".format(len(cent_tracked_past) - len(count_mot)), (10, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 0), 1)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            
    cv2.destroyAllWindows()


yolo = Load_Yolo_model()

Object_tracking(yolo, video_path, "detection1.mp4", 
    input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, 
    rectangle_colors=(255,0,0), Track_only = ["car", "motorbike"])