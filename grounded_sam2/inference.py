from typing import Literal, Tuple, List, Dict, Optional
from pathlib import Path
from loguru import logger

import time
import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2_camera_predictor
import grounding_dino.groundingdino as gd


GSAM2_PATH = Path(__file__).parents[1]
GDINO_CONFIG_PATH = str(GSAM2_PATH / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT_PATH = str(GSAM2_PATH / "gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_CONFIG_BASE_PATH: Path = Path("configs/sam2.1")  # hydra already has the base path
SAM2_CKPT_BASE_PATH: Path = GSAM2_PATH / "checkpoints"


if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class GroundedSAM2Predictor:
    """Inference class for real-time optimized GroundedSAM2

    Args:
        sam2_ckpt: The sam2.1 checkpoint type to use.
        sam2_device: The device to run sam2.1 on.
        vos_optimized: Whether to run torch.compile on sam2.1. This will make
            inference faster but can take a while initially. Compilations should
            be cached after the first run.
        gdino_box_threshold: The box threshold for the gdino model.
        gdino_text_threshold: The text threshold for the gdino model.
        gdino_device: The device to run the gdino model on.
        verbosity: The verbosity level.
            0: No output.
            1: Print out gdino class names and input boxes.
            2: Print out time for one pass + 1's output
    """
    def __init__(self,
                 sam2_ckpt: Literal['tiny', 'small', 'base', 'large'] = 'tiny',
                 sam2_device: str = 'cuda:0',
                 vos_optimized: bool = True,
                 gdino_box_threshold: float = 0.30,
                 gdino_text_threshold: float = 0.25,
                 gdino_device: str = 'cuda:0',
                 verbosity: Literal[0, 1, 2] = 0):

        assert torch.cuda.is_available()

        self._gdino_model = gd.util.inference.load_model(model_config_path=GDINO_CONFIG_PATH,
                                                         model_checkpoint_path=GDINO_CKPT_PATH,
                                                         device=gdino_device)
        self._gdino_box_threshold = gdino_box_threshold
        self._gdino_text_threshold = gdino_text_threshold

        sam2_model_cfg = str(SAM2_CONFIG_BASE_PATH / f"sam2.1_hiera_{sam2_ckpt[0]}.yaml")
        sam2_ckpt = str(SAM2_CKPT_BASE_PATH / f"sam2.1_hiera_{sam2_ckpt}.pt")
        self._sam2_model = build_sam2_camera_predictor(sam2_model_cfg, sam2_ckpt,
                                                       device=sam2_device,
                                                       vos_optimized=vos_optimized)

        self._verbosity = verbosity
        self._class_names = None
        self._first_detect = True
        self._latest_language = ""

    def _query_gdino(self,
                     text_prompt: str,
                     frame: np.ndarray):
        image_source, image = gd.util.inference.load_image(frame)
        boxes, confidences, labels = gd.util.inference.predict(
            model=self._gdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=self._gdino_box_threshold,
            text_threshold=self._gdino_text_threshold,
        )
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            # confidences = confidences.numpy().tolist()
            class_names = labels

        if self._verbosity >= 1:
            logger.info(f"GDINO Class names: {class_names}")
            logger.info(f"GDINO Input boxes: {input_boxes}")
        return input_boxes, class_names

    def _load_first_frame_sam2(self, frame: np.ndarray, input_boxes: np.ndarray):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            self._sam2_model.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = 1

            for bbox_coord in input_boxes:
                bbox = np.array(bbox_coord, dtype=np.float32).reshape((2, 2))
                _, out_obj_ids, out_mask_logits = self._sam2_model.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )
                ann_obj_id += 1

    def _query_sam2(self, frame: np.ndarray) -> Tuple[List[int], torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = self._sam2_model.track(frame)
        return out_obj_ids, out_mask_logits

    def query(self, frame: np.ndarray, language: str, display: Optional[Tuple[int, int]] = None)  -> Dict[str, np.ndarray]:
        """Query the model with a frame and language prompt.

        Args:
            frame: The frame to query.
            language: The language prompt to use. Should be in the format of

                    "class1, class2, class3"

                If the language changes between queries, gdino inference call will be made.
                Note that this is quite expensive.
            display: If not None, display the visualization with the specified resolution.

        Returns:
            A dictionary mapping class names to numpy masks.

        """
        with torch.inference_mode():
            s = time.perf_counter()

            if language != self._latest_language:
                self._latest_language = language
                self._first_detect = True

            if self._first_detect:
                self._first_detect = False

                # We only query gdino when we need to initialize the segmentation.
                # This should be avoided as much as possible due to high inference cost.
                input_boxes, self._class_names = self._query_gdino(language, frame)

                num_instances = len(language.split(", "))

                assert num_instances == len(self._class_names), \
                    (f"Number of detected instances ({num_instances}) does not match the number "
                     f"of class names ({len(self._class_names)}). Generally, we assume exactly one"
                     f"instance per specified class.")

                self._load_first_frame_sam2(frame, input_boxes)

            out_obj_ids, out_mask_logits = self._query_sam2(frame)

        if self._verbosity >= 2:
            logger.info(f"Time for one pass: {time.perf_counter() - s}")

        if display is not None:
            width, height = frame.shape[:2][::-1]
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)
                # Convert the mask to greenish color
                mask = np.zeros((height, width, 3), dtype=np.uint8)
                mask[:, :, 1] = all_mask

                frame = cv2.addWeighted(frame, 1, mask, 0.75, 0)

            cv2.imshow("Frame", cv2.cvtColor(cv2.resize(frame, display), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        output = {}
        for name, mask in zip(self._class_names, out_mask_logits):
            output[name] = mask[0].cpu().numpy()

        return output


if __name__ == "__main__":
    gsam_predictor = GroundedSAM2Predictor(verbosity=2)

    cap = cv2.VideoCapture("frontal_video_small.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize down to hobot2 resolution
        frame = cv2.resize(frame, (234, 133))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Note that when we downsize the resolution significantly, gdino
        # is unable to detect objects well. Therefore, it's better to use
        # more primitive descriptors like yellow circle instead of yellow plate.
        # This is not an issue if using a higher resolution.
        language = "yellow circle, light blue square, red square"
        # language = "yellow plate, blue towel, red block"

        masks = gsam_predictor.query(frame, language, display=(938, 532))

        # key = "yellow circle"
        # cv2.imshow(key, masks[key])
        # cv2.waitKey(1)

    cap.release()
