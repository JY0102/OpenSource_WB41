import cv2
import mediapipe as mp
import argparse
import json
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=True
)

# JSON 저장 함수
def save_coordinates_to_json(all_frames_data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_frames_data, f, ensure_ascii=False, indent=2)

def extract_pose_3d_coordinates(landmarks, image_width, image_height):
    coordinates_3d = []
    for i, landmark in enumerate(landmarks):
        coordinates_3d.append({
            'x': landmark.x * image_width,
            'y': landmark.y * image_height,
            'z': landmark.z * image_width            
        })
    return coordinates_3d

def main():
    parser = argparse.ArgumentParser(description='Holistic 3D 좌표 추출')
    parser.add_argument('--video', default=r'python/t.mp4', help='비디오 파일 유형 (0: 웹캠, path: 비디오 파일 경로)')
    args = parser.parse_args()
    
    video_source = args.video
    pose_frames_data = []
    l_hand_frames_data = []
    r_hand_frames_data = []
    frame_count = 0

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    if isinstance(video_source, str):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {video_source}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        if video_source == 0:
            frame = cv2.flip(frame, 1)
        
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image_rgb)
        pose_frame_coords = []
        l_hand_frame_coords = []
        r_hand_frame_coords = []
        
        # 포즈
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            pose_coords = extract_pose_3d_coordinates(results.pose_landmarks.landmark, width, height)
            pose_frame_coords.extend(pose_coords)
        
        # 왼손
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            left_hand_coords = extract_pose_3d_coordinates(results.left_hand_landmarks.landmark, width, height)
            l_hand_frame_coords.extend(left_hand_coords)

        # 오른손
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            right_hand_coords = extract_pose_3d_coordinates(results.right_hand_landmarks.landmark, width, height)
            r_hand_frame_coords.extend(right_hand_coords)
        

        pose_frames_data.append(pose_frame_coords)
        l_hand_frames_data.append(pose_frame_coords)
        r_hand_frames_data.append(r_hand_frame_coords)
        
        cv2.imshow('Holistic 3D Coordinate Extraction', frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 결과를 JSON으로 저장
    save_coordinates_to_json(pose_frames_data, "pose3d.json")
    save_coordinates_to_json(l_hand_frames_data, "hand_left3d.json")
    save_coordinates_to_json(r_hand_frames_data, "hand_right3d.json")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
