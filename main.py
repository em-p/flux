
import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
   
def calculateData():
    dataDir = "data"

    data = []
            #[filename ,[array of angles w/ combined confidence[angle,confidence]]]

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for filename in os.listdir(dataDir):
            f = os.path.join(dataDir, filename)
            if os.path.isfile(f):
                imgTest = cv2.imread(f, cv2.IMREAD_COLOR)
            else:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                LeftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LeftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                LeftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                LeftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                LeftKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LeftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                RightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                RightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                RightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                RightKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                

                anglesC = []
                leftElbowAng = calculate_angle(LeftShoulder, LeftElbow, LeftWrist)
                leftShoulderAng = calculate_angle(LeftHip, LeftShoulder, LeftElbow)
                leftHipAng = calculate_angle(LeftShoulder, LeftHip, LeftKnee)
                leftKneeAng = calculate_angle(LeftHip, LeftKnee, LeftAnkle)
                
                rightElbowAng = calculate_angle(RightShoulder, RightElbow, RightWrist)
                rightShoulderAng = calculate_angle(RightHip, RightShoulder, RightElbow)
                rightHipAng = calculate_angle(RightShoulder, RightHip, RightKnee)
                rightKneeAng = calculate_angle( RightHip, RightKnee, RightAnkle)
                
                anglesC.append(['left elbow', leftElbowAng])
                anglesC.append(['left shoulder',leftShoulderAng])
                anglesC.append(['left hip',leftHipAng])
                anglesC.append(['left knee',leftKneeAng])
                anglesC.append(['right elbow', rightElbowAng])
                anglesC.append(['right shoulder',rightShoulderAng])
                anglesC.append(['right hip',rightHipAng])
                anglesC.append(['right knee',rightKneeAng])
                data.append([f, anglesC])
            except:
                pass
            
            # Render detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
            #                         )               
            

    return data

def findBest(pose, data):
# pose = [leftElbowAng, leftShoulderAng,rightElbowAng, rightShoulderAng]
    best = 'data/photo1.jpeg'
    bestScore = 180 * 8
    score = bestScore
    leftElbowAng = pose[0]
    leftShoulderAng = pose[1]
    leftHipAng = pose[2] 
    leftKneeAng = pose[3]
    rightElbowAng = pose[4]
    rightShoulderAng = pose[5]
    rightHipAng = pose[6]
    rightKneeAng = pose[7]

    for p in data:
        score = abs(p[1][0][1] - leftElbowAng) + abs(p[1][1][1] - leftShoulderAng) + abs(p[1][2][1] - leftHipAng) + abs(p[1][3][1] - leftKneeAng)
        score += abs(p[1][4][1] - rightElbowAng) + abs(p[1][5][1] - rightShoulderAng) + abs(p[1][6][1] - rightHipAng) + abs(p[1][7][1] - rightKneeAng)
        print(score) #+ abs(p[1][1][1] - pose[1]) + abs(p[1][2][1] - pose[2]) + abs(p[1][3][1] - pose[3])
        if (score <= bestScore):
            best = p[0]
            bestScore = score
    print(best)
    return best


cap = cv2.VideoCapture(0)
data = calculateData()
print(data)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        best = 'data/photo1.jpeg'
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            LeftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            LeftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            LeftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            LeftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            LeftKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            LeftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            RightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            RightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            RightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            RightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            RightKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            RightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            leftElbowAng = calculate_angle(LeftShoulder, LeftElbow, LeftWrist)
            leftShoulderAng = calculate_angle(LeftHip, LeftShoulder, LeftElbow)
            leftHipAng = calculate_angle(LeftShoulder, LeftHip, LeftKnee)
            leftKneeAng = calculate_angle(LeftHip, LeftKnee, LeftAnkle)
            
            rightElbowAng = calculate_angle(RightShoulder, RightElbow, RightWrist)
            rightShoulderAng = calculate_angle(RightHip, RightShoulder, RightElbow)
            rightHipAng = calculate_angle(RightShoulder, RightHip, RightKnee)
            rightKneeAng = calculate_angle( RightHip, RightKnee, RightAnkle)

            best = findBest([leftElbowAng, leftShoulderAng, leftHipAng, leftKneeAng, rightElbowAng, rightShoulderAng, rightHipAng, rightKneeAng], data)
        except:
            pass
    
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        

        #resize and add border
        desired_size = 2048
        im = cv2.imread(best, cv2.IMREAD_COLOR)
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        cv2.imshow("image", new_im)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

