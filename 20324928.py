import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QFileDialog, QAction, QApplication, QVBoxLayout, QSlider, \
    QHBoxLayout, QMessageBox, QDialog, QProgressBar, QProgressDialog
from PyQt5.QtGui import QPixmap, QImage


class PanoramaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Panorama Generator with PyQt5')
        self.setGeometry(100, 100, 1000, 600)
        self.statusBar()

        openAction = QAction('Open Video', self)
        openAction.triggered.connect(self.openVideo)

        generateAction = QAction('Generate Panorama', self)
        generateAction.triggered.connect(self.generatePanorama)

        cropAction = QAction('Crop Panorama', self)
        cropAction.triggered.connect(self.cropPanorama)

        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(generateAction)
        fileMenu.addAction(cropAction)
        fileMenu.addAction(exitAction)

        self.layout = QVBoxLayout()

        self.welcomeLabel = QLabel(self)
        self.welcomeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.welcomeLabel.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        self.welcomeLabel.setText("Welcome to Panorama Generator")
        self.layout.addWidget(self.welcomeLabel)

        self.label = QLabel(self)
        self.label.resize(1000, 540)
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # The picture is centered on the screen.
        self.layout.addWidget(self.label)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setDisabled(True)  # Disable the slider initially.
        self.slider.sliderMoved.connect(self.changeFrame)  # Connect to the sliderMoved signal
        self.layout.addWidget(self.slider)

        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.frameIndex = 0

        self.show()

    def openVideo(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov);;All Files (*)",
                                                  options=options)
        if fileName:
            self.videoPath = fileName
            self.capture = cv2.VideoCapture(self.videoPath)
            self.frames = []
            frame_count = 0
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = QProgressDialog("Loading frames...", "Cancel", 0, total_frames, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumWidth(300)
            progress.setMinimumHeight(100)
            progress.setFont(QtGui.QFont("Arial", 10))
            while True:
                ret, image = self.capture.read()
                if not ret:
                    break
                if frame_count % 15 == 0:  # Take one frame every 15 frames
                    self.frames.append(image)
                frame_count += 1
                progress.setValue(frame_count)
                if progress.wasCanceled():
                    break
            progress.close()
            self.hideWelcomeLabel()  # Hide the welcome text when displaying a frame image
            self.statusBar().showMessage(f'Loaded {len(self.frames)} frames from video.')
            self.frameIndex = 0
            self.slider.setMaximum(len(self.frames) - 1)
            self.slider.setEnabled(True)  # Enable the slider after loading the video

    def hideWelcomeLabel(self):
        self.welcomeLabel.hide()

    def changeFrame(self, value):
        self.frameIndex = value

        if self.frames:
            image = self.frames[self.frameIndex]
            self.displayImage(image)
            self.statusBar().showMessage(f'Showing frame {self.frameIndex + 1}/{len(self.frames)}')

        else:
            self.statusBar().showMessage('No video loaded.')

    def previewFrame(self):
        if self.frames:
            self.frameIndex = (self.frameIndex + 1) % len(self.frames)
            self.slider.setValue(self.frameIndex)

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))

    def generatePanorama(self):
        if not hasattr(self, 'frames') or not self.frames:
            self.statusBar().showMessage('No video loaded.')
            return

        try:
            # 初始化拼接器
            stitcher = cv2.Stitcher_create()
            src_imgs = [frame for frame in self.frames]  # Use the images directly from the frames list.

            start_time = time.time()  # Start timing here
            # 使用Stitcher进行拼接
            status, panorama = stitcher.stitch(src_imgs)

            if status == cv2.Stitcher_OK:
                self.panorama = panorama
                self.displayImage(panorama)
                # Save the panorama image to a file
                cv2.imwrite('generated_panorama.jpg', panorama)  # Save the panorama image
                end_time = time.time()  # Stop timing here
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                self.statusBar().showMessage(f'Panorama generated successfully in {elapsed_time:.2f} seconds.')
                self.askForCrop()
            else:
                self.statusBar().showMessage(f'Error during stitching. Code: {status}')
        except Exception as e:
            self.statusBar().showMessage(f'An error occurred: {str(e)}')
            print(f'An error occurred: {str(e)}')  # Output an error message to the console for debugging

    def askForCrop(self):
        reply = QMessageBox.question(self, 'Crop Panorama', 'Do you want to crop the panorama?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.cropPanorama()

    def detect_edge_bounds(self, channel, threshold1=10, threshold2=20, min_edge_length=3):
        edges = cv2.Canny(channel, threshold1, threshold2)
        edge_strength = np.mean(edges, axis=0)
        strong_edges = edge_strength > np.mean(edge_strength)  # Use the average intensity as the threshold

        edge_blocks = np.split(strong_edges, np.where(np.diff(strong_edges))[0] + 1)
        edge_lengths = [block.size for block in edge_blocks if np.all(block)]

        if edge_lengths and max(edge_lengths) >= min_edge_length:
            trim_amount = edge_lengths[0] if np.all(edge_blocks[0]) else 0
            return trim_amount, trim_amount + max(edge_lengths)
        return 0, 0

    def cropPanorama(self):
        if not hasattr(self, 'panorama'):
            self.statusBar().showMessage('No panorama to crop.')
            return

        try:
            start_time = time.time()  # Start timing the cropping process
            gray = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)

            _, right_trim = self.detect_edge_bounds(gray)
            rotated_180 = cv2.rotate(gray, cv2.ROTATE_180)
            _, left_trim = self.detect_edge_bounds(rotated_180)
            print(f"Left: {left_trim}, Right: {right_trim}")

            # Upper and lower edge detection can be duplicated with the same function by transposing the image
            _, bottom_trim = self.detect_edge_bounds(gray.T)
            rotated_90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            _, top_trim = self.detect_edge_bounds(rotated_90)
            print(f"Top: {top_trim}, Bottom: {bottom_trim}")

            # crop
            cropped_panorama = self.panorama[top_trim:self.panorama.shape[0] - bottom_trim, left_trim:self.panorama.shape[1] - right_trim]

            # Save the cropped image
            cv2.imwrite('cropped_panorama.jpg', cropped_panorama)

            # Display the cropped image on the interface
            self.displayImage(cropped_panorama)

            cropping_time = time.time() - start_time  # Calculate the time taken to crop the panorama
            self.statusBar().showMessage(f'Panorama cropped and saved successfully in {cropping_time:.2f} seconds.')
        except cv2.error as e:
            self.statusBar().showMessage(f'OpenCV error: {str(e)}')
            print(f'OpenCV error: {str(e)}')
        except Exception as e:
            self.statusBar().showMessage(f'An error occurred during cropping: {str(e)}')
            print(f'An error occurred during cropping: {str(e)}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PanoramaApp()
    sys.exit(app.exec_())
