import os
import requests
import tempfile
import logging
from typing import List, Dict, Any, Tuple
from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='korean',
                use_gpu=True,
                enable_mkldnn=True,
                det_db_box_thresh=0.4,
                rec_algorithm='SVTR_LCNet',
                rec_image_shape="3,32,320",
                ocr_server=True,
                use_space_char=True
            )
            logger.info("OCR 서비스가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"OCR 서비스 초기화 중 오류 발생: {str(e)}")
            raise

    def download_image_from_s3(self, s3_url: str) -> np.ndarray:
        try:
            response = requests.get(s3_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            height, width = gray.shape[:2]
            if max(height, width) > 1920:
                scale = 1920 / max(height, width)
                gray = cv2.resize(gray, (int(width * scale), int(height * scale)))
            if np.mean(gray) > 127:
                gray = cv2.bitwise_not(gray)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
            kernel = np.ones((1, 1), np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            dilated = cv2.dilate(opened, kernel, iterations=1)
            smoothed = cv2.GaussianBlur(dilated, (3, 3), 0)
            return smoothed
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            return image

    def try_multiple_preprocessings(self, image: np.ndarray) -> List[np.ndarray]:
        images = [image]
        images.append(self.preprocess_image(image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        images.append(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.append(binary)
        enhanced = cv2.equalizeHist(gray)
        images.append(enhanced)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        images.append(sharpened)
        return images

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            result = self.ocr.ocr(image, cls=True)
            extracted_texts = []
            if result and isinstance(result[0], list):
                for line in result[0]:
                    if len(line) >= 2:
                        box = line[0]
                        text, confidence = line[1][:2] if isinstance(line[1], (list, tuple)) else (str(line[1]), 0)
                        extracted_texts.append({"text": text, "confidence": float(confidence), "bounding_box": box})
            return extracted_texts
        except Exception as e:
            logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
            raise

    def extract_text_multi(self, image: np.ndarray) -> List[Dict[str, Any]]:
        all_results = []
        image_variants = self.try_multiple_preprocessings(image)
        for variant in image_variants:
            try:
                results = self.extract_text(variant)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"OCR 실패 - 변형 이미지에서 오류 발생: {e}")
        return self.merge_similar_texts(all_results)

    def merge_similar_texts(self, results: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
        def iou(box1, box2) -> float:
            x1_min, x1_max = min(p[0] for p in box1), max(p[0] for p in box1)
            y1_min, y1_max = min(p[1] for p in box1), max(p[1] for p in box1)
            x2_min, x2_max = min(p[0] for p in box2), max(p[0] for p in box2)
            y2_min, y2_max = min(p[1] for p in box2), max(p[1] for p in box2)
            xi1, yi1 = max(x1_min, x2_min), max(y1_min, y2_min)
            xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0

        filtered = []
        used = [False] * len(results)
        for i, base in enumerate(results):
            if used[i]:
                continue
            duplicates = [base]
            used[i] = True
            for j in range(i + 1, len(results)):
                if used[j]:
                    continue
                if iou(base["bounding_box"], results[j]["bounding_box"]) > iou_thresh:
                    ratio = SequenceMatcher(None, base["text"], results[j]["text"]).ratio()
                    if ratio > 0.6:
                        duplicates.append(results[j])
                        used[j] = True
            best = max(duplicates, key=lambda x: x["confidence"])
            filtered.append(best)
        return filtered

    def post_process_text(self, text: str) -> str:
        import re
        if not text:
            return text
        cleaned = ' '.join(text.split())
        cleaned = re.sub(r'[^\w\s가-힣.,:;!?()]', '', cleaned)
        cleaned = re.sub(r'([a-zA-Z])([가-힣])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([가-힣])([a-zA-Z])', r'\1 \2', cleaned)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        capitalized = [s[0].upper() + s[1:] if s and s[0].isalpha() else s for s in sentences]
        return ' '.join(capitalized)

    def process_image(self, s3_url: str) -> Dict[str, Any]:
        try:
            original_image = self.download_image_from_s3(s3_url)
            ocr_results = self.extract_text_multi(original_image)
            for item in ocr_results:
                box = item["bounding_box"]
                item["center_y"] = sum(coord[1] for coord in box) / len(box)
                item["center_x"] = sum(coord[0] for coord in box) / len(box)
            y_tolerance = 10
            lines, sorted_items = [], sorted(ocr_results, key=lambda x: x["center_y"])
            if sorted_items:
                current_line = [sorted_items[0]]
                current_y = sorted_items[0]["center_y"]
                for item in sorted_items[1:]:
                    if abs(item["center_y"] - current_y) <= y_tolerance:
                        current_line.append(item)
                    else:
                        lines.append(current_line)
                        current_line = [item]
                        current_y = item["center_y"]
                if current_line:
                    lines.append(current_line)
            for line in lines:
                line.sort(key=lambda x: x["center_x"])
            sorted_results = [item for line in lines for item in line]
            all_text = " ".join(item["text"] for item in sorted_results)
            processed_text = self.post_process_text(all_text)
            text_items = [item["text"] for item in sorted_results]
            return {
                "success": True,
                "message": "OCR 처리 완료",
                "full_text": processed_text,
                "original_text": all_text,
                "text_items": text_items,
                "detailed_items": sorted_results,
                "s3_url": s3_url
            }
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            return {
                "success": False,
                "message": f"OCR 처리 실패: {str(e)}",
                "full_text": "",
                "original_text": "",
                "text_items": [],
                "detailed_items": [],
                "s3_url": s3_url
            }

ocr_service = OCRService()