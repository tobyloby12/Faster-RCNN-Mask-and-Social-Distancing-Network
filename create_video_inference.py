from network_again import *
import os

directory = os.getcwd()
print(directory)
exists = False
while exists == False:
    video = input('Enter video name: ')
    path = os.path.join(directory, f'videos/{video}')
    exists = os.path.exists(path)
    if exists == False:
        print('Entered invalid video name, remember to put filetype too.')
    
# # using video
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model_path = 'final_train\\mask_rcnn_mask_cfg_0025.h5'
model.load_weights(model_path, by_name=True)

capture = cv2.VideoCapture(f'videos/{video}')
number_of_frames = 0
frames = list()
filenames_mask = list()
filenames_bird = list()
filenames_combined = list()

def predict_and_save(image, path, frame, filenames_type):
    image_path = os.path.join(os.getcwd(), f'videos\\{path}')
    cv2.imwrite(os.path.join(image_path, f'image_{frame}.jpg'), image)
    filenames_type.append(os.path.join(image_path, f'image_{number_of_frames}.jpg'))
    height, width, depth = image.shape
    size = (width, height)
    return filenames_type, size

def video_write(size, filenames, name):
    img_array = []
    out = cv2.VideoWriter(f'videos\{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 29.97, size)
    for filename in filenames:
        img = cv2.imread(filename)
        img_array.append(img)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


while True:
    ret, frame = capture.read()
    if not ret:
        break
    frames.append(frame)

for frame in frames:
    predicted, bird, combined = detection_image_cv2(frame, model, cfg)
    filenames_mask, size_mask = predict_and_save(predicted, 'masks', number_of_frames, filenames_mask)
    filenames_bird, size_bird = predict_and_save(bird, 'bird', number_of_frames, filenames_bird)
    filenames_combined, size_combined = predict_and_save(combined, 'combined', number_of_frames, filenames_combined)
    number_of_frames += 1
    print(number_of_frames)

video_write(size_mask, filenames_mask, 'predicted')
video_write(size_bird, filenames_bird, 'birds_eye_view')
video_write(size_combined, filenames_combined, 'combined')
