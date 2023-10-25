# Importing Libraries
import os
import cv2
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse

# Handling Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, required=False)
parser.add_argument("-p", "--photo", type=str)
args = parser.parse_args()

mode = "webcam"
if args.mode == "webcam" or args.mode == "photo":
    mode = args.mode

photo_path = args.photo

num_faces = 0

# Getting All Image Files With Their Absolute Path
def get_all_files(directory_path):
  all_files = []
  for root, _, files in os.walk(directory_path):
    for file in files:
      all_files.append(os.path.join(root, file))
  return all_files

current_working_directory = os.path.abspath(__file__)
current_working_directory = os.path.dirname(current_working_directory)

faces_directory_path = os.path.join(current_working_directory, 'Faces')

all_files = get_all_files(faces_directory_path)

all_files = [file for file in all_files if os.path.isfile(file)]

# Initializing MTCNN and InceptionResnetV1 
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    keep_all=True
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Read Data From Folder
dataset = datasets.ImageFolder(faces_directory_path) 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = []
embedding_list = []

if not os.path.exists(os.path.join(current_working_directory, 'Data.pt')):
    print("Creating Data.pt")
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True) 
        if face is not None and prob>0.92:
            emb = resnet(face) 
            embedding_list.append(emb.detach()) 
            name_list.append(idx_to_class[idx])

    # Save Data
    data = [embedding_list, name_list] 
    torch.save(data, 'Data.pt') 

else:
    # Loading Data.pt File
    print("Data.pt Found")
    load_data = torch.load('Data.pt') 
    embedding_list = load_data[0] 
    name_list = load_data[1] 

source = cv2.VideoCapture(0)

counter = 0

if mode == "webcam":
    while True:
        ret, frame = source.read()

        frame = cv2.flip(frame, 1)

        if not ret:
            print("Failed To Access Webcam !")
            break
            
        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            num_faces = len(_)
                    
            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                    
                    dist_list = [] # List of Matched Distances
                    
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)
                    min_dist_idx = dist_list.index(min_dist)
                    name = name_list[min_dist_idx]
                    
                    box = boxes[i] 
                    
                    original_frame = frame.copy()
                    
                    if min_dist < 0.90:
                        frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                    else:
                        frame = cv2.putText(frame, "Unknown", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                    
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])) , (int(box[2]), int(box[3])), (255,0,0), 2)
                    frame = cv2.putText(frame, "Number of Faces Detected: " + str(num_faces), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("Face Recognition", frame)
            
        key = cv2.waitKey(1)

        if key == 27: # Escape Key
            break

elif mode == "photo":
    try:
        if os.path.exists(photo_path) == True:
            print("Reading Photo...")
            img = cv2.imread(photo_path)

            img_cropped_list, prob_list = mtcnn(img, return_prob=True) 

            if img_cropped_list is not None:
                print("Recognizing Faces...")
                boxes, _ = mtcnn.detect(img)
                        
                for i, prob in enumerate(list(prob_list)):
                    if prob > 0.9:
                        emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                    
                        dist_list = [] # List of Matched Distances
                        
                        for idx, emb_db in enumerate(embedding_list):
                            dist = torch.dist(emb, emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list)
                        min_dist_idx = dist_list.index(min_dist)
                        name = name_list[min_dist_idx]
                        
                        box = boxes[i] 
                        
                        if min_dist < 0.90:
                            cv2.putText(img, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(img, "Unknown", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1, cv2.LINE_AA)
                        
                        cv2.rectangle(img, (int(box[0]), int(box[1])) , (int(box[2]), int(box[3])), (255,0,0), 2)

                        cv2.imwrite("Result.jpg", img)
                        num_faces += 1

            print("Number of Faces Detected:", num_faces)
            print("Wrote Image at Result.jpg")

        else:
            print("Invalid Path to a Photo !")

    except:
        print("Error")


source.release()
cv2.destroyAllWindows()