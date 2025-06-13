import cv2  
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from classifier import SignClassifier

class HybridSignSystem:
    def __init__(self, detection_weights='best_5.pt', classification_weights='my_classifier.pth'):
        
        self.detector = YOLO(detection_weights)
        self.detector.overrides = {
            'imgsz': 640,
            'conf': 0.5,  
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        self.classifier = SignClassifier()
        self.classifier.load_state_dict(torch.load(classification_weights, map_location='cpu'))
        self.classifier.eval()

        # Unified class mapping
        self.class_names = self._load_class_mapping()

        # try to matche training classifier dat a
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

   
    def _load_class_mapping(self):
        return {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
            4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
            8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing for vehicles over 3.5 tons', 11: 'Right-of-way at next intersection',
            12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 tons prohibited',
            17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right',
            21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
            29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
            32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
            35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
            39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
            42: 'End of no passing by vehicles over 3.5 tons'
        }


    def _ensemble_prediction(self, yolo_cls, yolo_conf, cls_id, cls_conf):
        """Combine model predictions using weighted confidence"""
        yolo_weight = 0.6  # Higher weight for detection model
        cls_weight = 0.4
        
        # If models agree, boost confidence
        if yolo_cls == cls_id:
            final_conf = (yolo_conf * yolo_weight + cls_conf * cls_weight) * 1.2
        else:
            final_conf = (yolo_conf * yolo_weight + cls_conf * cls_weight) * 0.8
            
        return final_conf

    def _classify_sign(self, image):
        try:
            # Convert to RGB and pad to square
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            size = max(h, w)
            image_padded = cv2.copyMakeBorder(image, 
                (size-h)//2, (size-h+1)//2,
                (size-w)//2, (size-w+1)//2,
                cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # use transformer 
            image_tensor = self.transform(image_padded).unsqueeze(0)
            
            
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            cls_conf, cls_id = torch.max(probs, 1)
            return cls_id.item(), cls_conf.item()
            
        except Exception as e:
            print(f"Classification error: {e}")
            return -1, 0.0

    def process_frame(self, frame):
        detections = self.detector(frame)[0]
        results = []

        for box in detections.boxes:
            yolo_cls = int(box.cls.item())
            yolo_conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            h, w = frame.shape[:2]
            pad_x = int(0.1 * (x2 - x1))
            pad_y = int(0.1 * (y2 - y1))
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)

            sign_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if sign_roi.size == 0:
                continue

            cls_id, cls_conf = self._classify_sign(sign_roi)
            final_conf = self._ensemble_prediction(yolo_cls, yolo_conf, cls_id, cls_conf)

            if final_conf < 0.75:
                continue

            if cls_conf > 0.85:
                final_class = cls_id
            elif yolo_conf > 0.9:
                final_class = yolo_cls
            else:
                final_class = yolo_cls if yolo_conf > cls_conf else cls_id

            class_name = self.class_names.get(final_class, f'Unknown ({final_class})')
            color = (0, 255, 0) if yolo_cls == cls_id else (0, 165, 255)

            cv2.rectangle(frame, (x1_pad, y1_pad), (x2_pad, y2_pad), color, 2)
            label = f"{class_name} Y:{yolo_conf:.2f} C:{cls_conf:.2f} F:{final_conf:.2f}"
            cv2.putText(frame, label, (x1_pad, y1_pad - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            
            print(f"\nDetected sign:")
            print(f"  YOLO        → {self.class_names.get(yolo_cls)} (conf: {yolo_conf:.2f})")
            print(f"  Classifier  → {self.class_names.get(cls_id, 'Unknown')} (conf: {cls_conf:.2f})")
            print(f"  Final Class → {class_name} (combined conf: {final_conf:.2f})")
            if yolo_cls == cls_id:
                print(" Both models agree — boosted confidence.")
            else:
                print(" Models disagree — using weighted decision.")

            results.append({
                'bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                'class': class_name,
                'yolo_class': self.class_names[yolo_cls],
                'classifier_class': self.class_names.get(cls_id, 'Unknown'),
                'final_confidence': final_conf
            })

        cv2.imshow("Ensemble Detection", frame)
        cv2.waitKey(1)
        return results
