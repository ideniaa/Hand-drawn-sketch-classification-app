import sys
import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGraphicsScene, QGraphicsView, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from model import SketchCNN, class_labels  # Import your trained model and labels


class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the drawing variables
        self.drawing = False
        self.last_position = None
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.initUI()

        # Load trained model
        num_classes = len(class_labels)
        self.model = SketchCNN(num_classes)
        self.model.load_state_dict(torch.load("quickdraw_model.pth", map_location=torch.device('cpu')))
        self.model.eval()

        # Define image transformation for classification
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

        # Initialize MediaPipe Hand Tracker
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Start the webcam feed
        self.cap = cv2.VideoCapture(0)
        self.timer = self.startTimer(30)  # Update the screen every 30ms

    def initUI(self):
        """Set up the GUI."""
        self.setWindowTitle("Hand Drawing App")
        self.setGeometry(100, 100, 640, 520)

        # Create buttons
        self.clear_button = QPushButton("Clear Drawing", self)
        self.clear_button.clicked.connect(self.clear_canvas)

        self.save_button = QPushButton("Save Drawing", self)
        self.save_button.clicked.connect(self.save_and_classify)

        self.start_button = QPushButton("Start Drawing", self)
        self.start_button.clicked.connect(self.start_drawing)

        self.stop_button = QPushButton("Stop Drawing", self)
        self.stop_button.clicked.connect(self.stop_drawing)
        self.stop_button.setEnabled(False)  # Disable stop button initially

        # Create Layouts
        vbox = QVBoxLayout()
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.clear_button)
        vbox.addWidget(self.save_button)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setScene(self.scene)

        # Arrange layouts
        hbox = QHBoxLayout()
        hbox.addWidget(self.view)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

    def start_drawing(self):
        """Enable drawing."""
        self.drawing = True
        self.start_button.setEnabled(False)  # Disable start button while drawing
        self.stop_button.setEnabled(True)   # Enable stop button

    def stop_drawing(self):
        """Stop drawing."""
        self.drawing = False
        self.start_button.setEnabled(True)  # Enable start button to draw again
        self.stop_button.setEnabled(False)  # Disable stop button

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_canvas()

    def save_and_classify(self):
        """Save the drawing and classify it."""
        cv2.imwrite("drawing.png", self.canvas)
        print("Drawing saved. Classifying...")

        # Load and preprocess the saved drawing
        image = Image.open("drawing.png").convert("L")  # Convert to grayscale
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Predict class
        with torch.no_grad():
            output = self.model(image)
            predicted_class = output.argmax().item()

        print(f"Predicted drawing: {class_labels[predicted_class]}")

        # Show messagebox with prediction
        self.show_message(f"Predicted drawing: {class_labels[predicted_class]}")

    def show_message(self, message):
        """Show a message box."""
        msg = QMessageBox()
        msg.setText(message)
        msg.exec_()

    def timerEvent(self, event):
        """Update the drawing with the webcam feed."""
        if not self.drawing:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Flip for a mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip

                # Convert to pixel coordinates
                x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)

                # Draw smooth lines between consecutive points
                if self.last_position is not None:
                    cv2.line(self.canvas, self.last_position, (x, y), (255, 255, 255), 5)  # Draw thicker line
                cv2.circle(self.canvas, (x, y), 5, (255, 255, 255), -1)  # Draw on canvas

                # Update the last position
                self.last_position = (x, y)

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Update the canvas view
        self.update_canvas()

    def update_canvas(self):
        """Update the GUI with the drawing."""
        # Convert the canvas to QImage and display it
        height, width, channel = self.canvas.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.canvas.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DrawingApp()
    ex.show()
    sys.exit(app.exec_())
