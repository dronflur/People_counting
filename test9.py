from darkflow.net.build import TFNet
import math
import cv2
import dlib 


# const
OPTIONS = {"model": "cfg/yolo-voc.cfg", "load": "bin/yolov2-voc.weights", "threshold": 0.1}
SAME_PERSON_THRESHOLD = 1
CHECK_POINT = 20
MAX_HUMAN_SIZE = 3000
MIN_HUMAN_SIZE = 2000

border_width = 0
border_height = 0

prev_locations = []
tracking = []
#path = 'sample_img/test_video_01f.mov'
path = 'sample_img/pinkloa.mp4'
trackers = []
tracking_locations = []

class Coordinate:
    P1 = None
    P2 = None

class Position:
    X = None
    Y = None

class TrackingLocation:
    current = None
    checkpoint = None

class Location:
    coor = None
    center = None

def createLine():
    line_in = Coordinate()
    line_in.P1 = Position()
    line_in.P2 = Position()
    line_in.P1.X = 0
    line_in.P1.Y = border_height / 2
    line_in.P2.X = border_width
    line_in.P2.Y = border_height / 2
    '''
    line_in.P1.X = 115
    line_in.P1.Y = 105
    line_in.P2.X = 557
    line_in.P2.Y = 230
    '''
    return line_in

def getLocation(x1, y1, x2, y2):
    coordinate = Coordinate()
    coordinate.P1 = Position()
    coordinate.P1.X = x1
    coordinate.P1.Y = y1
    coordinate.P2 = Position()
    coordinate.P2.X = x2
    coordinate.P2.Y = y2
    return coordinate

def getCenter(coordinate):
    result = Position()
    result.X = (coordinate.P1.X + coordinate.P2.X) / 2
    result.Y = (coordinate.P1.Y + coordinate.P2.Y) / 2
    return result

def getUserLocation(coordinate):
    result = Position()
    result.X = (coordinate.P1.X + coordinate.P2.X) / 2
    result.Y = (coordinate.P1.Y + coordinate.P2.Y) / 2
    location = Location()
    location.center = result
    location.coor = coordinate
    return location

def getDistance(p1, p2):
    return math.sqrt((p1.X - p2.X)**2 + (p1.Y - p2.Y)**2)

def isSamePerson(c1, c2):
    #if isHuman(c1) and isHuman(c2):
    '''
    if (c1.P1.X >= c2.P1.X and c1.P1.X < c2.P2.X) or (c1.P2.X >= c2.P1.X and c1.P2.X < c2.P2.X):
        return True
    if (c1.P1.Y >= c2.P1.Y and c1.P1.Y < c2.P2.Y) or (c1.P1.X < c2.P1.X and c1.P2.X > c2.P1.X):
        return True
    return False

    '''
    if isInside(c1, c2):
        return True
    if isInside(c2, c1):
        return True
    if isOverLap(c1, c2):
        return True
    if isOverLap(c2, c1):
        return True
    return False
    #return getDistance(p1, p2) <= SAME_PERSON_THRESHOLD

def isHuman(c1):
    return (c1.P2.X - c1.P1.X)*1.5 <= (c1.P2.Y - c1.P1.Y) #and getSize(c1) >= MIN_HUMAN_SIZE and getSize(c1) <= MAX_HUMAN_SIZE

def getSize(c1):
    return (c1.P2.X - c1.P1.X) * (c1.P2.Y - c1.P1.Y)

def isSameTracker(c1, c2):
    return hasSameLocation(c1, c2)

def isInside(c1, c2):
    return (c1.P1.X <= c2.P1.X and c1.P2.X >= c2.P2.X) or (c1.P1.Y >= c2.P1.Y and c1.P2.Y <= c2.P2.Y)

def isOverLap(c1, c2):
    return (c1.P1.X < c2.P1.X and c1.P2.X < c2.P2.X and c1.P2.X > c2.P1.X and c1.P2.X < c2.P2.X) or (c1.P1.Y > c2.P1.Y and c1.P2.Y > c2.P2.Y and c1.P2.Y < c2.P1.Y and c1.P2.Y > c2.P2.Y)

def getYfromParam(p1, p2, x):
    return ((p2.Y - p1.Y) / (p2.X - p1.X))*(x - p1.X) + p1.Y

def isAboveTheLine(line, p):
    print('expected Y: ', getYfromParam(line.P1, line.P2, p.X))
    return p.Y > getYfromParam(line.P1, line.P2, p.X)

def isPersonComeIn(line, current, previous):
    return isAboveTheLine(line_in, getCenter(current)) and not isAboveTheLine(line_in, getCenter(previous))

def isPersonComeOut(line, current, previous):
    return not isAboveTheLine(line_in, getCenter(current)) and isAboveTheLine(line_in, getCenter(previous))

def cleanUpTracking(trackers, tracking_locations, iteration_no):
    new_tracking = []
    new_trackers = []
    for i in range(len(tracking_locations)):
        if not (iteration_no >= CHECK_POINT and isExpire(tracking_locations[i])):
            new_tracking.append(tracking_locations[i])
            new_trackers.append(trackers[i])
        else:
            print("cleaned up tracking: ", i)
    tracking_locations = new_tracking
    trackers = new_trackers
    return trackers, tracking_locations

def cleanUpTracker(trackers, tracking_locations):
    new_tracking = []
    new_trackers = []
    for i in range(len(trackers)):
        rect_i = trackers[i].get_position()
        coor_i = getLocation(rect_i.left(), rect_i.top(), rect_i.right(), rect_i.bottom())
        shouldAddnewRow = True
        if coor_i.P2.Y >= height or coor_i.P1.Y <= 0 or coor_i.P1.X <= 0 or coor_i.P2.X >= width:
            shouldAddnewRow = False
        for j in range(i+1, len(trackers)):
            rect_j = trackers[j].get_position()
            coor_j = getLocation(rect_j.left(), rect_j.top(), rect_j.right(), rect_j.bottom())
            #center_i = getCenter(coor_i)
            #center_j = getCenter(coor_j)
            if isSameTracker(coor_i, coor_j):
                shouldAddnewRow = False

        if shouldAddnewRow:
            new_tracking.append(tracking_locations[i])
            new_trackers.append(trackers[i])
        else:
            print("cleaned up tracker: ", i)

    tracking_locations = new_tracking
    trackers = new_trackers
    return trackers, tracking_locations


def isExpire(tracking):
    return tracking.checkpoint and hasSameLocation(tracking.checkpoint, tracking.current)

def hasSameLocation(l1, l2):
    c1 = getCenter(l1)
    c2 = getCenter(l2)
    return getDistance(c1, c2) <= SAME_PERSON_THRESHOLD
    #return l1.P1.X == l2.P1.X and l1.P1.Y == l2.P1.Y and l1.P2.X == l2.P2.X and l1.P2.Y == l2.P2.Y

def counting(tracking_locations, count_in, count_out):
    for i in range(len(tracking_locations)):
        if tracking_locations[i].checkpoint and isPersonComeIn(line_in, tracking_locations[i].current, tracking_locations[i].checkpoint):
            count_in = count_in + 1
            tracking_locations[i].checkpoint = tracking_locations[i].current
            print('count in: '+str(count_in))
        if tracking_locations[i].checkpoint and isPersonComeOut(line_in, tracking_locations[i].current, tracking_locations[i].checkpoint):
            count_out = count_out + 1
            tracking_locations[i].checkpoint = tracking_locations[i].current
            print('count out: '+str(count_out))
    return count_in, count_out

def convertToTracking(tracker):
    rect = tracker.get_position()
    tracker_coor = getLocation(rect.left(), rect.top(), rect.right(), rect.bottom())
    print("added new tracker")
    #printCoor(tracker_coor)
    tracking_location = TrackingLocation()
    tracking_location.current = tracker_coor
    tracking_location.checkpoint = tracker_coor
    return tracking_location

def cleanUpCurrent(current_locations):
    new_current_locations = []
    for i in range(len(current_locations)):
        shouldAddnewRow = True
        if not isHuman(current_locations[i].coor):
            continue
        for j in range(i+1, len(current_locations)):
            if isSamePerson(current_locations[i].coor, current_locations[j].coor):
                shouldAddnewRow = False
        if shouldAddnewRow:
            new_current_locations.append(current_locations[i])
    return current_locations

def printCoor(c):
    if c:
        print('x1: %i y1: %i x2: %i y2: %i' % (c.P1.X, c.P1.Y, c.P2.X, c.P2.Y))
    else:
        print('None')


if __name__ == "__main__": 
    tfnet = TFNet(OPTIONS)

    cap = cv2.VideoCapture(path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    border_width= width
    border_height = height

    people = []
    count = 0
    count_out = 0

    line_in = createLine()

    iteration_no = 0

    while True:
        for i in range(0,2):
            ret,frame = cap.read()

        if not ret: 
            break

        temp_result = tfnet.return_predict(frame)
        results = list(filter(lambda x: x['label'] == 'person', temp_result))
        results = list(map(lambda x: getLocation(x['topleft']['x'], x['topleft']['y'], x['bottomright']['x'], x['bottomright']['y']), results))
        
        current_locations = list(map(lambda x: getUserLocation(x), results))
        people = []
        new_locations = []
        for c in current_locations:
            if isHuman(c.coor):
                new_locations.append(c)
        current_locations = new_locations
        for current_location in current_locations:
            cv2.rectangle(frame,(int(current_location.coor.P1.X), int(current_location.coor.P1.Y)),(int(current_location.coor.P2.X), int(current_location.coor.P2.Y)),(0,0,0),3)
        

        print("################")
        # Remove tracked people
        for i in range(len(trackers)):
            trackers[i].update(frame)
            rect = trackers[i].get_position()
            tracker_coor = getLocation(rect.left(), rect.top(), rect.right(), rect.bottom())
            cv2.rectangle(frame,(int(tracker_coor.P1.X), int(tracker_coor.P1.Y)),(int(tracker_coor.P2.X), int(tracker_coor.P2.Y)),(255,255,255),3)
            #tracking_locations[i].checkpoint = tracking_locations[i].current
            tracking_locations[i].current = tracker_coor
            if iteration_no > 0 and iteration_no % CHECK_POINT == 0:
                tracking_locations[i].checkpoint = tracker_coor
            print("current: ")
            printCoor(tracking_locations[i].current)
            print("checkpoint: ")
            printCoor(tracking_locations[i].checkpoint)
            print("************")
            del_indexs = []
            for j in range(len(current_locations)):
                #center_tracking_location = getCenter(tracker_coor)
                #printCoor(current_locations[j].coor)
                if isSamePerson(tracker_coor, current_locations[j].coor):
                    del_indexs.append(j)
                    #del current_locations[j]
                    print('removed duplicate person')
                    #break
            
            for k in sorted(del_indexs, key=int, reverse=True):
                del current_locations[k]

        
        print('total new people: ', len(current_locations))
        
        # Add a new tracker to a new people
        for current_location in current_locations:
            if isHuman(current_location.coor):
                tracker = dlib.correlation_tracker()
                cv2.rectangle(frame,(int(current_location.coor.P1.X), int(current_location.coor.P1.Y)),(int(current_location.coor.P2.X), int(current_location.coor.P2.Y)),(144,144,255),3)
                tracker.start_track(frame, dlib.rectangle(current_location.coor.P1.X,current_location.coor.P1.Y,current_location.coor.P2.X,current_location.coor.P2.Y))
                trackers.append(tracker)
                tracking_locations.append(convertToTracking(tracker))
        
        print('total trackers: ', len(trackers))
        print('total trackering: ', len(tracking_locations))

        # Clean up tracker
        trackers, tracking_locations = cleanUpTracker(trackers, tracking_locations)
        #Clean up tracking
        if iteration_no and iteration_no % CHECK_POINT == 0:
            trackers, tracking_locations = cleanUpTracking(trackers, tracking_locations, iteration_no)

        count, count_out = counting(tracking_locations, count, count_out)

        cv2.line(frame,(int(line_in.P1.X), int(line_in.P1.Y)),(int(line_in.P2.X), int(line_in.P2.Y)),(255,0,0),5)
        cv2.putText(frame,'countIn: '+str(count),(10,30),font, 1,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(frame,'countOut: '+str(count_out),(10,60),font, 1,(0,0,0),1,cv2.LINE_AA)
        
        cv2.imshow('Counting',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
         break 
        
        iteration_no = iteration_no + 1



    cap.release()