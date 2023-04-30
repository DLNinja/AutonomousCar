# Autonomous Car

<img align="center" alt="RobotPhoto" width="500px" src="https://github.com/DLNinja/AutonomousCar/blob/main/Robot.jpg" />

This is an Autonomous robot that follows objects in a distinct color. </br> The project was done by my friend Dragos and me, he worked on the electronics side and I worked on the object-following part. </br>
The robot uses a Jetson Nano with a camera to identify the object and sends the information to an Arduino that controls the movements. </br>

<img align="center" alt="ComponentsPhoto" width="300px" src="https://github.com/DLNinja/AutonomousCar/blob/main/Robot2.jpg" />

---
## Object Following

For this part I used Python and OpenCV to capture images with a camera and detect a certain color (I also made a code to find the value for the color). If it detects a bigger area of that color it identifies it as the wanted object and the position of the robot is sent to the Arduino to decide if it needs to rotate or go forward/backwards. </br>
I also developed a code to detect if there's an obstacle in front of the robot and avoid it. For this I used the ResNet18 model and got pretty good results but it had some lag.

