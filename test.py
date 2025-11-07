import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

class HandGestureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=2)

        self.word_classifier = Classifier(
            "C:/Users/USER/Hand detector/converted_keras/keras_model.h5",
            "C:/Users/USER/Hand detector/converted_keras/labels.txt")
        self.letter_classifier = Classifier(
            "C:/Users/USER/Hand detector/allLetters/keras_model.h5",
            "C:/Users/USER/Hand detector/allLetters/labels.txt")
        self.number_classifier = Classifier(
            "C:/Users/USER/Hand detector/numbers_converted_keras/keras_model.h5",
            "C:/Users/USER/Hand detector/numbers_converted_keras/labels.txt")

        self.offset = 20
        self.img_size = 300
        self.mode = "word"
        self.last_warn = 0

        self.word_labels = [
            "Drink", "F*** you", "Food", "Hello", "I love you", "No", "Ok",
            "You're welcome", "Peace", "Please", "Sorry", "Thank you", "Yes",
            "Good bye", "Happy", "Sad", "Angry", "Hate", "Good luck", "Live long and prosper"
        ]
        self.letter_labels = [chr(i) for i in range(65, 91)]
        self.number_labels = ["1", "2", "3", "4", "5"]

        self.meanings = {
            "Drink": "To take in liquid refreshment.",
            "F*** you": "An offensive or rude gesture.",
            "Food": "Something people eat for nourishment.",
            "Hello": "A greeting used when meeting someone.",
            "I love you": "Words expressing affection or love.",
            "No": "Used to refuse or disagree politely.",
            "Ok": "Expressing agreement, acceptance, or acknowledgment.",
            "You're welcome": "A polite reply to thanks.",
            "Peace": "A state of calm and harmony.",
            "Please": "Used to make polite requests.",
            "Sorry": "Used to express regret or apology.",
            "Thank you": "Words expressing gratitude or appreciation.",
            "Yes": "Used to agree or give consent.",
            "Good bye": "Used when leaving or parting.",
            "Happy": "Feeling joy, pleasure, or satisfaction.",
            "Sad": "Feeling sorrow, unhappiness, or disappointment.",
            "Angry": "Feeling strong displeasure or hostility.",
            "Hate": "Intense dislike or strong aversion.",
            "Good luck": "Wishing success or positive outcome.",
            "Live long and prosper": "A common Vulcan greeting expressing good wishes.", # Corrected description
        }

    def preprocess_hand(self, img, bbox):
        x, y, w, h = bbox
        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        y1 = max(0, y - self.offset)
        y2 = min(img.shape[0], y + h + self.offset)
        x1 = max(0, x - self.offset)
        x2 = min(img.shape[1], x + w + self.offset)
        img_crop = img[y1:y2, x1:x2]
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = self.img_size / h
            w_cal = int(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
            w_gap = (self.img_size - w_cal) // 2
            img_white[:, w_gap:w_gap + w_cal] = img_resize
        else:
            k = self.img_size / w
            h_cal = int(k * h)
            img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
            h_gap = (self.img_size - h_cal) // 2
            img_white[h_gap:h_gap + h_cal, :] = img_resize
        return img_white

    def predict_gesture(self, img_white):
        if self.mode == "word":
            prediction, index = self.word_classifier.getPrediction(img_white, draw=False)
            label = self.word_labels[index]
            color = (0, 0, 255) if label == "F*** you" else (0, 255, 0)
            meaning = self.meanings.get(label, "No description available.")
            text = f"{label}: {meaning}"
        elif self.mode == "letter":
            prediction, index = self.letter_classifier.getPrediction(img_white, draw=False)
            label = self.letter_labels[index]
            color = (0, 255, 0)
            text = f"Letter: {label}"
        else:
            prediction, index = self.number_classifier.getPrediction(img_white, draw=False)
            label = self.number_labels[index]
            color = (0, 255, 0)
            text = f"Number: {label}"
        return text, color

    def draw_info(self, img_output, bbox, text, color):
        x, y, w, h = bbox
        font_scale = 0.6
        thickness = 1
        line_height = 20  

        if ": " in text:
            label, meaning = text.split(": ", 1)
            lines = [f"{label}:", meaning]
        else:
            lines = [text]

        rect_height = line_height * len(lines) + 10
        rect_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, thickness)[0][0] for line in lines) + 20

        top_left = (x - self.offset, y - self.offset - rect_height)
        bottom_right = (x - self.offset + rect_width, y - self.offset)

        cv2.rectangle(img_output, top_left, bottom_right, color, cv2.FILLED)

        for i, line in enumerate(lines):
            cv2.putText(img_output, line, (x - self.offset + 10, y - self.offset - rect_height + (i+1)*line_height),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0, 0, 0), thickness)

        cv2.rectangle(img_output, (x - self.offset, y - self.offset),
                      (x + w + self.offset, y + h + self.offset), color, 3)


    def draw_ui_labels(self, img_output):
        cv2.putText(img_output, "Press keys to switch mode:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_output, "W - Word | L - Letter | N - Number | Q - Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_output, f"Current mode: {self.mode.upper()}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Camera not detected.")
                break
            img = cv2.flip(img, 1)
            img_output = img.copy()
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                bbox = hand['bbox']
                img_white = self.preprocess_hand(img, bbox)
                text, color = self.predict_gesture(img_white)
                self.draw_info(img_output, bbox, text, color)
            self.draw_ui_labels(img_output)
            cv2.imshow("Hand Gesture Recognition", img_output)
            key = cv2.waitKey(1)
            if key == ord('w'):
                self.mode = "word"
            elif key == ord('l'):
                self.mode = "letter"
            elif key == ord('n'):
                self.mode = "number"
            elif key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandGestureApp()
    app.run()
