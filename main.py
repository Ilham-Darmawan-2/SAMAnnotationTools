#!/usr/bin/env python3
"""
Annotator + YOLO label saver + background quick-train (1 epoch)
Requirements:
  - python3, opencv-python, numpy
  - ultralytics installed in same env (pip install ultralytics)
"""
import cv2, os, xml.etree.ElementTree as ET, shutil, numpy as np
from xml.dom import minidom
import threading

# ======== TRAINING LIB ========
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work until installed.", e)

# ======== CONFIG ========
workspaceName = "my_project"
input_folder = "savedFrameNight4"
output_folder = f"output/{workspaceName}"                # Pascal VOC
inference_root = f"inference/{workspaceName}"            # YOLO data (images + labels)
inference_images = os.path.join(inference_root, "images")
inference_labels = os.path.join(inference_root, "labels")
model_folder = f"models/{workspaceName}"
model_path = os.path.join(model_folder, "modelAssistant.pt")

for d in [output_folder, inference_images, inference_labels, model_folder]:
    os.makedirs(d, exist_ok=True)

CLASSLIST = ["person", "chair"]
current_class = CLASSLIST[0]

# ======== GLOBALS ========
images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
images.sort()
if not images:
    raise SystemExit(f"No images found in {input_folder}")

current_index = 0
bboxes = []
selected_bbox = None
drawing = False
moving = False
resizing = False
ix, iy = -1, -1
display_scale = 1.0
frame = None
orig_shape = None
scroll_offset = 0
CLASS_HEIGHT = 35
CLASS_WINDOW_W = 220
CLASS_WINDOW_H = 360

training_running = False

# ======== UTIL ========
def prettify_xml(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="   ")

def save_pascal_voc(img_name, img_shape):
    xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0]+".xml")
    ann = ET.Element("annotation")
    ET.SubElement(ann,"folder").text = workspaceName
    ET.SubElement(ann,"filename").text = img_name
    size = ET.SubElement(ann,"size")
    ET.SubElement(size,"width").text = str(img_shape[1])
    ET.SubElement(size,"height").text = str(img_shape[0])
    ET.SubElement(size,"depth").text = str(img_shape[2] if len(img_shape)>2 else 3)
    for bbox in bboxes:
        x1 = int(bbox[0]/display_scale)
        y1 = int(bbox[1]/display_scale)
        x2 = int(bbox[2]/display_scale)
        y2 = int(bbox[3]/display_scale)
        cls = bbox[4]
        obj = ET.SubElement(ann,"object")
        ET.SubElement(obj,"name").text = cls
        bnd = ET.SubElement(obj,"bndbox")
        ET.SubElement(bnd,"xmin").text = str(max(0,x1))
        ET.SubElement(bnd,"ymin").text = str(max(0,y1))
        ET.SubElement(bnd,"xmax").text = str(max(0,x2))
        ET.SubElement(bnd,"ymax").text = str(max(0,y2))
    with open(xml_path,"w") as f: f.write(prettify_xml(ann))
    print(f"[INFO] Saved VOC: {xml_path}")

def save_yolo_label_and_image(img_name, orig_img):
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(inference_labels, base+".txt")
    dest_img = os.path.join(inference_images,img_name)
    h,w = orig_img.shape[:2]
    lines=[]
    for bbox in bboxes:
        x1=int(bbox[0]/display_scale); y1=int(bbox[1]/display_scale)
        x2=int(bbox[2]/display_scale); y2=int(bbox[3]/display_scale)
        cls = bbox[4]
        if cls not in CLASSLIST: continue
        idx=CLASSLIST.index(cls)
        bw=(x2-x1)/w; bh=(y2-y1)/h
        cx=(x1+x2)/2/w; cy=(y1+y2)/2/h
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path,"w") as f: f.write("\n".join(lines))
    shutil.copy2(os.path.join(input_folder,img_name),dest_img)
    print(f"[INFO] Saved YOLO label: {label_path}")

def draw_all(frame_draw):
    for i,(x1,y1,x2,y2,cls) in enumerate(bboxes):
        color=(0,255,0) if i!=selected_bbox else (0,0,255)
        cv2.rectangle(frame_draw,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame_draw,cls,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

def load_annotation_local(img_name_local):
    xml_path=os.path.join(output_folder,os.path.splitext(img_name_local)[0]+".xml")
    if not os.path.exists(xml_path): return []
    tree=ET.parse(xml_path)
    root=tree.getroot()
    boxes=[]
    for obj in root.findall("object"):
        cls=obj.find("name").text
        bb=obj.find("bndbox")
        x1=int(bb.find("xmin").text)
        y1=int(bb.find("ymin").text)
        x2=int(bb.find("xmax").text)
        y2=int(bb.find("ymax").text)
        boxes.append([int(x1*display_scale),int(y1*display_scale),int(x2*display_scale),int(y2*display_scale),cls])
    return boxes

# ======== MOUSE ========
def mouse_event(event,x,y,flags,param):
    global ix,iy,drawing,selected_bbox,moving,resizing,frame
    if event==cv2.EVENT_LBUTTONDOWN:
        ix,iy=x,y
        for i,(x1,y1,x2,y2,cls) in enumerate(bboxes):
            if x1<=x<=x2 and y1<=y<=y2:
                selected_bbox=i
                if abs(x-x2)<10 and abs(y-y2)<10: resizing=True
                else: moving=True
                return
        drawing=True; selected_bbox=None
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp=frame.copy()
            cv2.rectangle(temp,(ix,iy),(x,y),(255,0,0),2)
            draw_all(temp)
            cv2.imshow("Annotator",temp)
        elif moving and selected_bbox is not None:
            dx,dy=x-ix,y-iy
            bboxes[selected_bbox][0]+=dx;bboxes[selected_bbox][1]+=dy
            bboxes[selected_bbox][2]+=dx;bboxes[selected_bbox][3]+=dy
            ix,iy=x,y
        elif resizing and selected_bbox is not None:
            bboxes[selected_bbox][2]=max(bboxes[selected_bbox][0]+5,x)
            bboxes[selected_bbox][3]=max(bboxes[selected_bbox][1]+5,y)
    elif event==cv2.EVENT_LBUTTONUP:
        if drawing:
            x1,y1,x2,y2=sorted([ix,x])[0],sorted([iy,y])[0],sorted([ix,x])[1],sorted([iy,y])[1]
            w,h=abs(x2-x1),abs(y2-y1)
            if w>=5 and h>=5: bboxes.append([x1,y1,x2,y2,current_class])
            else: print("[INFO] Skipped tiny bbox (<5px).")
        drawing=moving=resizing=False

def class_mouse_event(event,x,y,flags,param):
    global current_class, scroll_offset
    if event==cv2.EVENT_LBUTTONDOWN:
        idx=(y+scroll_offset)//CLASS_HEIGHT
        if 0<=idx<len(CLASSLIST):
            current_class=CLASSLIST[idx]
            print(f"[INFO] Selected class: {current_class}")
    elif event==cv2.EVENT_MOUSEWHEEL:
        try: steps=int(flags/120)
        except: steps=0
        scroll_offset-=steps*CLASS_HEIGHT
        max_off=max(0,len(CLASSLIST)*CLASS_HEIGHT-CLASS_WINDOW_H)
        scroll_offset=max(0,min(scroll_offset,max_off))

def draw_class_window():
    canvas=np.zeros((CLASS_WINDOW_H,CLASS_WINDOW_W,3),dtype=np.uint8)
    start_idx=scroll_offset//CLASS_HEIGHT
    end_idx=min(start_idx+CLASS_WINDOW_H//CLASS_HEIGHT+1,len(CLASSLIST))
    y_pos=-(scroll_offset%CLASS_HEIGHT)
    for i in range(start_idx,end_idx):
        color=(0,0,255) if CLASSLIST[i]==current_class else (60,60,60)
        cv2.rectangle(canvas,(0,y_pos),(CLASS_WINDOW_W,y_pos+CLASS_HEIGHT-2),color,-1)
        cv2.putText(canvas,CLASSLIST[i],(10,y_pos+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        y_pos+=CLASS_HEIGHT
    cv2.imshow("ClassSelector",canvas)

# ======== TRAIN ========
def train_model():
    global training_running
    if training_running:
        print("[INFO] Training already running.")
        return
    training_running=True
    images_infer=[f for f in os.listdir(inference_images) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if len(images_infer)<10:
        print("[INFO] Not enough images to train (min 10 required).")
        training_running=False
        return
    np.random.shuffle(images_infer)
    split=len(images_infer)//2
    train_images=images_infer[:split]
    val_images=images_infer[split:]
    yaml_path = os.path.join(inference_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(inference_images)}\n")
        f.write(f"val: {os.path.abspath(inference_images)}\n")
        f.write(f"nc: {len(CLASSLIST)}\n")
        f.write("names: [" + ", ".join([f"'{n}'" for n in CLASSLIST]) + "]\n")
    init_model=model_path if os.path.exists(model_path) else "yolo11s.pt"
    if os.path.exists(model_path): print(f"[INFO] Found existing model {model_path}")
    else: print("[INFO] Using pretrained yolo11s.pt")
    try:
        model=YOLO(init_model)
        print("[INFO] Starting training (1 epoch)...")
        model.train(data=yaml_path,epochs=5,imgsz=640,batch=4,project=model_folder,name="train_run",exist_ok=True)
        model.save(model_path)
        print(f"[INFO] Training finished. Model saved to {model_path}")
    except Exception as e: print("[INFO] Training failed:",e)
    training_running=False

# ======== INFERENCE ========
def inference_current(conf=0.5):
    global bboxes, frame, display_scale, images, current_index, input_folder

    if not os.path.exists(model_path):
        print("[INFO] Model assistant does not exist.")
        return

    # baca gambar asli
    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)
    h, w = orig_img.shape[:2]

    model = YOLO(model_path)
    results = model.predict(orig_img, conf=conf)
    pred_boxes = []

    for r in results:
        if hasattr(r, 'boxes'):
            for box in r.boxes:
                x1 = int(box.xyxy[0,0].item())
                y1 = int(box.xyxy[0,1].item())
                x2 = int(box.xyxy[0,2].item())
                y2 = int(box.xyxy[0,3].item())
                cls_idx = int(box.cls[0].item())
                cls_name = CLASSLIST[cls_idx] if cls_idx < len(CLASSLIST) else str(cls_idx)
                # rescale ke ukuran frame display
                x1_disp = int(x1 * display_scale)
                y1_disp = int(y1 * display_scale)
                x2_disp = int(x2 * display_scale)
                y2_disp = int(y2 * display_scale)
                pred_boxes.append([x1_disp, y1_disp, x2_disp, y2_disp, cls_name])

    # update global bbox yang sudah di-rescale
    bboxes = pred_boxes

    # ===== simpan Pascal VOC + YOLO label + copy image =====
    # Pascal VOC (pakai koordinat asli)
    xml_path = os.path.join(output_folder, os.path.splitext(images[current_index])[0] + ".xml")
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = workspaceName
    ET.SubElement(annotation, "filename").text = images[current_index]
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(orig_img.shape[2] if len(orig_img.shape) > 2 else 3)
    for bbox in results[0].boxes:  # pakai koordinat asli untuk simpan
        x1 = int(bbox.xyxy[0,0].item())
        y1 = int(bbox.xyxy[0,1].item())
        x2 = int(bbox.xyxy[0,2].item())
        y2 = int(bbox.xyxy[0,3].item())
        cls_idx = int(bbox.cls[0].item())
        cls_name = CLASSLIST[cls_idx] if cls_idx < len(CLASSLIST) else str(cls_idx)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = cls_name
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)
    with open(xml_path, "w") as f:
        f.write(minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   "))

    # YOLO label
    label_path = os.path.join(inference_labels, os.path.splitext(images[current_index])[0] + ".txt")
    lines = []
    for bbox in results[0].boxes:
        x1 = int(bbox.xyxy[0,0].item())
        y1 = int(bbox.xyxy[0,1].item())
        x2 = int(bbox.xyxy[0,2].item())
        y2 = int(bbox.xyxy[0,3].item())
        cls_idx = int(bbox.cls[0].item())
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    # copy image
    dest_image_path = os.path.join(inference_images, images[current_index])
    shutil.copy2(img_path, dest_image_path)

    print("[INFO] Inference saved and bboxes updated for display.")

# ======== MAIN LOOP ========
cv2.namedWindow("Annotator",cv2.WINDOW_NORMAL)
cv2.namedWindow("ClassSelector",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator",mouse_event)
cv2.setMouseCallback("ClassSelector",class_mouse_event)

while True:
    img_name=images[current_index]
    orig=cv2.imread(os.path.join(input_folder,img_name))
    if orig is None:
        print(f"[INFO] Skip {img_name}")
        current_index=(current_index+1)%len(images)
        continue
    orig_shape=orig.shape
    h,w=orig_shape[:2]
    scale=720/h if h>720 else 1.0
    display_scale=scale
    frame=cv2.resize(orig,(int(w*scale),int(h*scale)))
    bboxes=load_annotation_local(img_name)

    while True:
        disp=frame.copy()
        print(bboxes)
        draw_all(disp)
        cv2.putText(disp,f"{img_name} [{current_index+1}/{len(images)}]",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(disp,f"Class: {current_class}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        draw_class_window()
        cv2.imshow("Annotator",disp)
        key=cv2.waitKey(30)&0xFF

        if key==ord('d'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            current_index=(current_index+1)%len(images)
            break
        elif key==ord('a'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            current_index=(current_index-1)%len(images)
            break
        elif key==ord('r') and selected_bbox is not None:
            del bboxes[selected_bbox]; selected_bbox=None
        elif key in [ord('s')] and selected_bbox is not None:
            idx=CLASSLIST.index(bboxes[selected_bbox][4])
            idx=(idx+1)%len(CLASSLIST)
            bboxes[selected_bbox][4]=CLASSLIST[idx]
        elif key==ord('t'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            train_model()
        elif key==ord('g'):
            inference_current(conf=0.5)
        elif key in [27, ord('q')]:
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            print("[INFO] Exiting and saved current annotations.")
            cv2.destroyAllWindows()
            exit(0)
        elif key in range(49,49+len(CLASSLIST)):
            idx=key-49
            current_class=CLASSLIST[idx]
            print(f"[INFO] Selected class via key: {current_class}")
