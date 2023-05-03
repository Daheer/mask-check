import cv2
import numpy as np
import tempfile
import time
import streamlit as st
from PIL import Image 
from io import BytesIO
from ultralytics import YOLO
from torch.cuda import is_available

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'
MODEL_PATH = 'mask_check.pt'
device = '0' if is_available() else 'cpu'

pred_dict = {
    0: 'mask',
    1: 'no-mask'
}
color_dict = {
    'mask': (0, 255, 0),
    'no-mask': (0, 0, 255)
}

@st.cache_data
def load_model():
  model = YOLO(MODEL_PATH)
  return model

def draw_preds(model, image, confidence = 0.25, iou = 0.7):
  
  preds = model.predict(image, device = device, conf = confidence, iou = iou)[0]
  boxes = preds.boxes.data.cpu()

  for box in boxes:  

    x1, y1, x2, y2 = list(map(int, box[:4]))    
    
    pred_prob = int(box[4] * 100)
    pred_class = pred_dict[int(box[5])]
    pred_text = f"{pred_class}: {pred_prob}%"
    
    color = color_dict[pred_class]
    pred_text_size, _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
    pred_text_position = ((x2+x1)//2 - pred_text_size[0]//2, y1 - pred_text_size[1])
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, pred_text, (pred_text_position), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
  return image, (preds.boxes.cls == 0).sum(), (preds.boxes.cls == 1).sum()

st.set_page_config(page_title="Mask Check", page_icon="😷")
st.title('Mask Check 😷')
st.sidebar.title('Options')

@st.cache_data()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]

  if width is None and height is None:
    return image
  
  if width is None:
    r = width / float(w)
    dim = (int(width * r), height)
  else:
    r = width/float(w)
    dim = (width, int(h*r))
  
  resized = cv2.resize(image, dim, interpolation = inter)

  return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
                                   ['About the App', 'Check on Image', 'Check on Video'])

st.markdown(
      """
      <style>
          [data-testid = 'stSidebar'][aria-expanded = 'true'] > div:first-child{
            width: 350px
          }
          [data-testid = 'stSidebar'][aria-expanded = 'false'] > div:first-child{
            width: 350px
            margin-left: -350px
          }
      </style>
      """, unsafe_allow_html = True
)

if app_mode == 'About the App':
  st.success('This app is created and maintained by [Deedax Inc.](https://github.com/Daheer)')
  st.markdown(
    """
    The purpose of "Mask Check" is to provide a simple and easy-to-use tool for checking if people in images and videos are wearing face masks
    # Installation
    Simply run
    ` bash setup.sh `
    \n
    . This will download model weights, requirements and launch the app
    # Usage
    The application can be run in three distinct modes: Image mode, Video mode, and Webcam mode (which is a subset of Video mode). In Image mode, you upload an image, and the application will automatically detect and return the relevant results. 
    \n
    In Video mode, users are able to upload a video file and receive corresponding results. Webcam mode, which is nested within Video mode, enables users to run the model on input data captured directly from a webcam. Outputs from the Video mode and Webcam mode can be saved by checking the 'Record' checkbox.
    # Features
    - Image and Video upload
    - Face detection
    - Mask detection
    - FPS counter
    - Results display
    - Webcam inference
    # Built Using
    - [Python](https://python.org)
    - [YOLOv8](https://ultralytics.com/yolov8)
    - [Roboflow](https://roboflow.com/)
    - [Streamlit](https://streamlit.io/)
    # Details
    - Dataset: With regard to the [dataset](https://universe.roboflow.com/deedaxinc/face-mask-detection-uamjv/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true), approximately half of the 150+ images were self-collected by myself, using a webcam to capture a wide array of facial expressions and features, including frontal and side-facing poses, wearing sunglasses, hats, and headphones. The remaining images were sourced from the internet and were chosen to represent a diverse range of skin tones and races. To balance out the dataset, which was originally heavily skewed towards male subjects due to the local self-collected data being exclusively male, the additional images were chosen to include a slightly higher proportion of female subjects.
    - Data augmentation: The augmentation techniques employed included a range of carefully chosen image manipulations designed to enhance the dataset's diversity and improve model generalization. Each augmentation was meticulously considered for its potential impact on performance, ensuring that the resulting dataset was both comprehensive and representative of real-world scenarios.
        - Crop: 0% Minimum Zoom, 20% Maximum Zoom
        - Rotation: Between -10° and +10°
        - Blur: Up to 2px
        - Noise: Up to 5% of pixels
        - Cutout: 10 boxes with 8% size each
        - Bounding Box: Brightness: Between -25% and +25%
        - Bounding Box: Exposure: Between -25% and +25%
    - Model selection: The selection of YOLOv8s was based on its state-of-the-art design, combined with its compact size (~20MB) and high level of accuracy
    # Performance
    Results shown below are from training the model using YOLOv8s for 100 epochs. See [notebook](training_mask_check.ipynb)
    """)
  st.image('images/confusion_matrix.png')
  st.image('images/labels.jpg')
  st.image('images/labels_correlogram.jpg')
  st.image('images/PR_curve.png')
  st.image('images/results.png')
  st.markdown("""
    # Limitations
    - Relatively low frame rate of approximately 2 frames per second during video inference.
    # Contact
    Dahir Ibrahim (Deedax Inc) \n
    Email - dahiru.ibrahim@outlook.com \n
    Twitter - https://twitter.com/DeedaxInc \n
    YouTube - https://www.youtube.com/@deedaxinc \n
    Project Link - https://github.com/Daheer/mask-check
    """
  )
elif app_mode == 'Check on Image':
  
  model = load_model()

  st.sidebar.markdown('---')
  
  kpi1, kpi2, kpi3, kpi4 = st.columns(4)
  with kpi2:
    st.markdown('**Masks Detected**')
    kpi2_text = st.markdown('0')
  with kpi3:
    st.markdown('**No Masks Detected**')
    kpi3_text = st.markdown('0')

  detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value = 0.0, max_value = 1.0, value = 0.25)
  detection_iou = st.sidebar.slider('Min Detection IoU (Intecept Over Union)', min_value = 0.0, max_value = 1.0, value = 0.7)

  st.sidebar.markdown('---')

  img_file_buffer = st.sidebar.file_uploader('Upload an Image', type = ['jpg', 'png', 'jpeg'])
  if img_file_buffer:
    buffer = BytesIO(img_file_buffer.read())
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
  else:
    demo_image = DEMO_IMAGE
    image = cv2.imread(demo_image, cv2.IMREAD_COLOR)
  
  st.sidebar.text('Original Image')
  st.sidebar.image(image, channels = 'BGR')

  with st.spinner():
    results = draw_preds(model, image, confidence = detection_confidence, iou = detection_iou)
  out_image = results[0]
  masks = results[1]
  no_masks = results[2]

  kpi2_text.write(f'<h1 style = "text-align: center; color: green;"> {masks} </h1>', unsafe_allow_html = True)
  kpi3_text.write(f'<h1 style = "text-align: center; color: red;"> {no_masks} </h1>', unsafe_allow_html = True)

  st.subheader('Output Image')
  st.image(out_image, channels = 'BGR', use_column_width = True)

elif app_mode == 'Check on Video':

  st.set_option('deprecation.showfileUploaderEncoding', False)
  
  use_webcam = st.sidebar.button('Use Webcam')
  record = st.sidebar.checkbox('Record Video')

  if record:
    st.checkbox('Recording', True)

  st.sidebar.markdown('---')

  detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value = 0.0, max_value = 1.0, value = 0.25)
  detection_iou = st.sidebar.slider('Min Detection IoU (Intercept Over Union)', min_value = 0.0, max_value = 1.0, value = 0.7)
  
  kpi1, kpi2, kpi3 = st.columns(3)
  with kpi1:
    st.markdown('**Frame Rate**')
    kpi1_text = st.markdown('0')
  with kpi2:
    st.markdown('**Masks Detected**')
    kpi2_text = st.markdown('0')
  with kpi3:
    st.markdown('**No Masks Detected**')
    kpi3_text = st.markdown('0')
  
  st.sidebar.markdown('---')

  st.markdown('## Output Video')

  stframe = st.empty()
  video_file_buffer = st.sidebar.file_uploader('Upload a Video', type = ['mov', 'avi', 'mp4', 'mkv'])
  tffile = tempfile.NamedTemporaryFile(delete = False)

  if use_webcam:
    vid = cv2.VideoCapture(0)
  else:
    if video_file_buffer:
      tffile.write(video_file_buffer.read())
      vid = cv2.VideoCapture(tffile.name)
    else:   
      vid = cv2.VideoCapture(DEMO_VIDEO)
      tffile.name = DEMO_VIDEO

  width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_input = int(vid.get(cv2.CAP_PROP_FPS))

  codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
  out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))      

  st.sidebar.text('Input Video')
  st.sidebar.video(tffile.name)

  fps = 0
  i = 0
  model = load_model()
  
  masks = 0
  no_masks = 0
  prevTime = 0
  
  while vid.isOpened():
    i += 1
    ret, frame = vid.read()
    if not ret:
      continue
    
    results = model.predict(frame, device = device)
    out_image = results[0].plot()
    out_image.flags.writeable = True
    masks = (results[0].boxes.cls == 0).sum()
    no_masks = (results[0].boxes.cls == 1).sum()

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    kpi1_text.write(f'<h1 style = "text-align: center; color: blue;"> {int(fps)} </h1>', unsafe_allow_html = True)
    kpi2_text.write(f'<h1 style = "text-align: center; color: green;"> {masks} </h1>', unsafe_allow_html = True)
    kpi3_text.write(f'<h1 style = "text-align: center; color: red;"> {no_masks} </h1>', unsafe_allow_html = True)

    out_image = cv2.resize(out_image, (0, 0), fx = .8, fy = .8)
    out_image = image_resize(image = out_image, width = 640)

    if record:
      out.write(out_image)

    stframe.image(out_image, channels = 'BGR', use_column_width = True)

  vid.release()
