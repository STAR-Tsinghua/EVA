
# Edge-assisted Video Analytics

### Setup environment

Python3.7 is proved to be ok, and install packages in requirements.txt

### Download the model

[model path](https://drive.google.com/drive/folders/1PpF4p9EHXWj3QScYRgeH3KcrCKIOfSsP?usp=sharing) (for yolo and fastrcnn)

### Run two terminals for ServerSystem and CameraSystem

### ServerSystem

To run backend, please cd to ServerSys/backend, paste downloaded fastrcnn model in current dir, and enter 'python -m flask run --port 5001'

### CameraSystem

Ensure that you have started ServerSys at localhost:5001(Changable in ./CameraSys/utils.py global variable 'HNAME')

cd to CameraSys, unzip downloaded yolo model into root of CameraSys

Run 'python main.py' to start camera pipeline
