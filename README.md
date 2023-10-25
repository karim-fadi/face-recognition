# Face Recognition

My Python face recognition project allows you to count and recognize faces in both photos and live video streams.

# Installation
* `pip install opencv-python torch torchvision facenet-pytorch`

# Usage
You can recognize faces using a webcam or in a photo. By default, if you run the file like this:

`python Face Recognition.py`

It will open your webcam and recognize faces in a live video stream.

If you want to recognize faces and count them in a photo, just set the `-m` or ``--mode`` flags to "photo", and the `-p` or `--photo` flags to the photo path.

`python Face Recognition.py --mode photo --photo ./Test 1.jpg`

You just need to put the face you want to recognize in the faces folder in a sub folder with the name of that person.

# Example

Test Image 1:


![Test 2](https://github.com/karim-fadi/face-recognition/assets/147660672/4cf16b41-0c64-4003-a8f3-753f55c25ee3)


Result:


![Result 2](https://github.com/karim-fadi/face-recognition/assets/147660672/e352931c-2df8-4406-bd68-2c4a66c73d5c)




Test Image 2:


![Test 3](https://github.com/karim-fadi/face-recognition/assets/147660672/3655f23d-92a2-475d-80b0-11ac4e25fa1f)


Result:


![Result 3](https://github.com/karim-fadi/face-recognition/assets/147660672/d992d086-4a44-4eca-bb4b-9c679729668d)


![Screenshot 2023-10-25 205818](https://github.com/karim-fadi/face-recognition/assets/147660672/2b926632-248b-4eeb-9f5f-623eb3e02025)
