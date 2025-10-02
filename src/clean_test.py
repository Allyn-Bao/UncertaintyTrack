import torch
import numpy as np
from PIL import Image
import cv2

from model.model_loader import ModelLoader

from model.tracker.uncertainty_tracker import UncertaintyTracker
from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

from datasets.mot17_dataset import MOT17CocoDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_model(model_checkpoint_path, type="yolox", class_names=('perdestrian',)):
    """
    Build MOT model from model type and checkpoint path.
    Args:
        model_checkpoint_path (str): Path to model checkpoint.
        type (str): Model type, e.g., 'yolox'.
        class_names (tuple): Class names for the model.
    Returns:
        model (torch.nn.Module): Loaded model.
    """
    if type == "yolox":
        from cjm_yolox_pytorch.model import build_model
        model_checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model_type = "yolox_x"
        model = build_model(model_type, len(class_names), pretrained=True)

        # --- fix for Missing key(s) in state_dict error ---
        # Extract state_dict
        state_dict = model_checkpoint["state_dict"] if "state_dict" in model_checkpoint else model_checkpoint

        # Remove "module." prefixes if they exist
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict, strict=False)
        # --- end fix ---

        return model
    

def build_mot17_dataloader(ann_file, img_prefix, batch_size=4, num_workers=4, input_size=(640, 640)):
    dataset = MOT17CocoDataset(ann_file, img_prefix, input_size=input_size)
    def detection_collate_fn(batch):
        images = torch.stack([b[0] for b in batch])      # stack image tensors
        targets = [b[1] for b in batch]                  # keep list of dicts/arrays
        return images, targets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)
    return dataset, dataloader


def preprocess_single_image(raw_image, input_size=(640, 640)):
    """
    Preprocess a single RGB numpy image for YOLOX inference.
    
    Steps:
    1. Resize while keeping aspect ratio
    2. Pad to (input_size, input_size) with 114 (YOLOX default gray)
    3. Convert to CHW torch tensor and normalize to [0, 1]

    Args:
        raw_image (np.ndarray): Input RGB image (H, W, 3).
        input_size (tuple): Desired size (h, w), must be multiple of 32.

    Returns:
        torch.Tensor: (1, 3, H, W) normalized image tensor.
    """
    h, w = raw_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    # Resize while keeping aspect ratio
    resized = cv2.resize(raw_image, (nw, nh))

    # Pad with 114 (standard YOLOX padding value)
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:nh, :nw, :] = resized

    # Convert to torch tensor (1, 3, H, W), normalize
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor / 255.0

    return tensor

def single_image_inference(model, image):
    with torch.no_grad():
        img = preprocess_single_image(image)
        outputs = model(img)
        print("Outputs:", outputs)


def multi_image_inference(model, images):
    with torch.no_grad():
        pass


def main():
    test_image = np.array(Image.open("/home/allynbao/project/UncertaintyTrack/src/data/MOT17/test/MOT17-03-FRCNN/img1/000001.jpg").convert("RGB"))
    model_checkpoint_path = "/home/allynbao/project/UncertaintyTrack/src/work_dirs/test_run/yolox_latest.pth"

    # --- Build MOT17 dataset + dataloader ---
    ann_file_path = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-train_cocoformat.json'
    image_prefix_path = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train'

    # --- Initialize modle for inference ---
    model = ModelLoader.build_model(model_checkpoint_path, type="yolox", class_names=('pedestrian',))
    model = model.to(device).eval()

    # --- Initialize Tracker
    tracker = UncertaintyTracker(
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30,
        with_covariance=True,
        det_score_mode='confidence',
        use_giou=False,
        expand_boxes=True,
        percent=0.3,
        ellipse_filter=True,
        filter_output=True,
        combine_mahalanobis=False
    )
    motion_model = KalmanFilterWithUncertainty(fps=30)
    tracker.motion = motion_model

    dataset, dataloader = build_mot17_dataloader(ann_file_path, image_prefix_path)

    all_results = []
    # --- Inference ---
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)     # (B=1, 3, H, W)
            raw_preds = model(imgs) 



            if i % 10 == 0:
                print(f"Processed {i} batches")

            
if __name__ == "__main__":
    main()
