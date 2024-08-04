import os
import subprocess
import librosa
import cv2
import dlib

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

def detect_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_wav2lip(image_path, audio_path, checkpoint_path): 
    command = f"python inference.py --checkpoint_path {checkpoint_path} --face {image_path} --audio {audio_path} --pads 0 0 0 0 --nosmooth"
    subprocess.run(command, shell=True)

def main():
    # Paths
    audio_path = 'audio/medieval-gamer-voice-donx27t-forget-to-subscribe-226581.wav'
    image_path = 'img/00084-2009709178.png'
    checkpoint_path = 'path/wav2lip_gan.pth'

    # Preprocess audio
    preprocess_audio(audio_path)

    # Detect landmarks (for visualization, can be skipped in final run)
    detect_landmarks(image_path)

    # Run Wav2Lip model
    run_wav2lip(image_path, audio_path, checkpoint_path)

if __name__ == "__main__":
    main()
