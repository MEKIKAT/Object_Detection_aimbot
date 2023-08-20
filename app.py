import cv2
import numpy as np
import pyautogui
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer


class FaceDetectionWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.classifier = cv2.CascadeClassifier('datasets/fc.xml')

        self.screenshot_region = [(0, 0), (538, 0), (525, 763), (0, 767)]

   
        self.screen_size = (525, 767)  

     
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        
        self.auto_mouse_pointing_checkbox = QCheckBox("Automatic Mouse Pointing")
        self.auto_mouse_pointing_checkbox.setChecked(False)
        self.auto_mouse_pointing_checkbox.stateChanged.connect(self.toggle_auto_mouse_pointing)

        self.show_bounding_box_checkbox = QCheckBox("Show Bounding Box")
        self.show_bounding_box_checkbox.setChecked(True)
        self.show_bounding_box_checkbox.stateChanged.connect(self.update_frame)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.auto_mouse_pointing_checkbox)
        layout.addWidget(self.show_bounding_box_checkbox)
        self.setLayout(layout)

    
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

     
        self.update_frame()

    def update_frame(self):
        frame = np.array(pyautogui.screenshot(region=(
            self.screenshot_region[0][0],
            self.screenshot_region[0][1],
            self.screen_size[0],
            self.screen_size[1]
        )))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if self.show_bounding_box_checkbox.isChecked():
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)

   
        self.image_label.setPixmap(pixmap)

 
        if self.auto_mouse_pointing_checkbox.isChecked() and len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_x = x + (w // 2)
            center_y = y + (h // 2)
            pyautogui.moveTo(self.screenshot_region[0][0] + center_x, self.screenshot_region[0][1] + center_y)

    def toggle_auto_mouse_pointing(self, checked):
        if checked:
            self.point_mouse_to_first_face()

    def point_mouse_to_first_face(self):
  
        frame = np.array(pyautogui.screenshot(region=(
            self.screenshot_region[0][0],
            self.screenshot_region[0][1],
            self.screen_size[0],
            self.screen_size[1]
        )))

  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_x = x + (w // 2)
            center_y = y + (h // 2)
            pyautogui.moveTo(self.screenshot_region[0][0] + center_x, self.screenshot_region[0][1] + center_y)

    def closeEvent(self, event):
     
        self.timer.stop()


if __name__ == '__main__':

    app = QApplication([])

    window = QMainWindow()
    window.setWindowTitle("Face Detection")
    window.setFixedSize(800, 600)

    face_detection_widget = FaceDetectionWidget()

    window.setCentralWidget(face_detection_widget)

    window.show()

    app.exec_()
