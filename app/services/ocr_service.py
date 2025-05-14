# ocr_service.py

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

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        """OCR 서비스 초기화 함수"""
        try:
            # PaddleOCR 초기화 (한글+영어 지원)
            self.ocr = PaddleOCR(
                use_angle_cls = True,
                lang = 'korean',
                use_gpu=True,       # GPU 사용
                enable_mkldnn=True, # Intel CPU 최적화 (GPU가 없는 경우)
                )
            logger.info("OCR 서비스가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"OCR 서비스 초기화 중 오류 발생: {str(e)}")
            raise

    def download_image_from_s3(self, s3_url: str) -> np.ndarray:
        """
        S3 URL에서 이미지를 다운로드하여 OpenCV 이미지로 변환
        
        Args:
            s3_url: S3 이미지 URL
            
        Returns:
            np.ndarray: OpenCV 이미지
        
        Raises:
            Exception: 이미지 다운로드 실패시
        """
        try:
            # S3 URL에서 이미지 다운로드
            response = requests.get(s3_url)
            response.raise_for_status()  # 오류 발생시 예외 발생
            
            # Bytes를 이미지로 변환
            image = Image.open(BytesIO(response.content))
            
            # PIL 이미지를 OpenCV 형식으로 변환
            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            
            return opencv_image
        
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {str(e)}")
            raise Exception(f"이미지 다운로드 실패: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        OCR 정확도 향상을 위한 이미지 전처리
        
        Args:
            image: 원본 OpenCV 이미지
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        try:
            # 원본 이미지 복사
            preprocessed = image.copy()
            
            # 컬러 이미지인 경우 그레이스케일로 변환
            if len(preprocessed.shape) == 3:
                gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
            else:
                gray = preprocessed
                
            # 이미지 크기 확인 및 리사이징 (너무 큰 이미지는 축소)
            height, width = gray.shape[:2]
            max_dimension = 1920  # 최대 차원
            
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # 적응형 히스토그램 평활화를 통한 대비 개선
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 적응형 이진화 (텍스트 강조)
            # - blockSize: 픽셀 주변 영역 크기
            # - C: 임계값 계산에 사용되는 상수
            binary = cv2.adaptiveThreshold(
                enhanced, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=11, 
                C=2
            )
            
            # 형태학적 연산을 통한 텍스트 영역 개선
            # 작은 노이즈 제거
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 텍스트 굵기 조정 (텍스트가 너무 가늘면 iterations 값을 늘림)
            kernel = np.ones((1, 1), np.uint8)
            dilation = cv2.dilate(opening, kernel, iterations=1)
            
            # 가우시안 블러를 사용한 가장자리 부드럽게
            smoothed = cv2.GaussianBlur(dilation, (3, 3), 0)
            
            # 결과 이미지 반환
            return smoothed
            
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            # 전처리 실패 시 원본 이미지 반환
            return image
    
    def try_multiple_preprocessings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        여러 전처리 방법을 적용하여 이미지 세트 생성
        
        Args:
            image: 원본 OpenCV 이미지
            
        Returns:
            List[np.ndarray]: 다양한 전처리를 적용한 이미지 리스트
        """
        images = []

        if image is None:
            logger.warning("원본 이미지가 None입니다.")
            return images
        
        # 원본 이미지 추가
        images.append(image)
        
        # 기본 전처리 이미지 추가
        images.append(self.preprocess_image(image))
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        images.append(gray)
        
        # 이진화 (단순 임계값)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.append(binary)
        
        # 대비 강화
        enhanced = cv2.equalizeHist(gray)
        images.append(enhanced)
        
        # 샤프닝 필터
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        images.append(sharpened)
        
        return images

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 추출
        
        Args:
            image: OpenCV 이미지
            
        Returns:
            List[Dict[str, Any]]: 추출된 텍스트 정보 리스트
        """
        try:
            # OCR 수행
            result = self.ocr.ocr(image, cls=True)
            
            # 결과 처리
            extracted_texts = []
            
            if result is None:
                logger.warning("OCR 결과가 None입니다.")
                return extracted_texts
            
            # 최신 PaddleOCR 결과 형식에 맞게 처리
            if isinstance(result, list) and len(result) > 0:
                page_results = result[0] if isinstance(result[0], list) else result
                
                for line in page_results:
                    if len(line) >= 2:
                        box = line[0]
                        
                        if isinstance(line[1], tuple) and len(line[1]) >= 2:
                            text, confidence = line[1]
                        elif isinstance(line[1], list) and len(line[1]) >= 2:
                            text, confidence = line[1][0], line[1][1]
                        else:
                            text, confidence = str(line[1]), 0
                        
                        extracted_texts.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bounding_box": box
                        })
            
            return extracted_texts
        
        except Exception as e:
            logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
            raise Exception(f"텍스트 추출 중 오류 발생: {str(e)}")

    def post_process_text(self, text: str) -> str:
        """
        OCR 결과 텍스트 후처리
        
        Args:
            text: OCR로 추출된 텍스트
            
        Returns:
            str: 후처리된 텍스트
        """
        # 텍스트가 비어있으면 그대로 반환
        if not text:
            return text
            
        # 불필요한 공백 제거
        cleaned = ' '.join(text.split())
        
        # 특수 문자 처리
        import re
        cleaned = re.sub(r'[^\w\s가-힣.,:;!?()]', '', cleaned)
        
        # 영어와 한글 사이에 공백 추가
        cleaned = re.sub(r'([a-zA-Z])([가-힣])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([가-힣])([a-zA-Z])', r'\1 \2', cleaned)
        
        # 문장 첫 글자 대문자화
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        capitalized = []
        
        for sentence in sentences:
            if sentence and sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]
            capitalized.append(sentence)
        
        result = ' '.join(capitalized)
        
        return result

    def process_image(self, s3_url: str) -> Dict[str, Any]:
        """
        S3 URL 이미지 처리 파이프라인 (동기 버전)
        
        Args:
            s3_url: S3 이미지 URL
            
        Returns:
            Dict[str, Any]: OCR 결과
        """
        try:
            # 이미지 다운로드
            original_image = self.download_image_from_s3(s3_url)
            
            # 이미지 전처리
            preprocessed_image = self.preprocess_image(original_image)
            
            # 원본본 이미지로 텍스트 추출
            ocr_results = self.extract_text(original_image)
            
            # # 결과가 없거나 불충분하면 원본 이미지로 시도
            # if len(ocr_results) < 2:
            #     logger.info("전처리된 이미지에서 충분한 텍스트를 추출하지 못했습니다. 원본 이미지로 시도합니다.")
            #     original_results = self.extract_text(original_image)
                
            #     # 원본 이미지 결과가 더 많으면 사용
            #     if len(original_results) > len(ocr_results):
            #         ocr_results = original_results
            
            # 여전히 결과가 불충분하면 여러 전처리 방법 시도
            # if len(ocr_results) < 2:
            #     logger.info("다양한 전처리 방법을 시도합니다.")
            #     best_results = ocr_results
                
            #     # 여러 전처리 이미지로 시도
            #     # images = self.try_multiple_preprocessings(original_image)
                
            #     for i, img in enumerate(images):
            #         try:
            #             current_results = self.extract_text(img)
                        
            #             # 더 많은 텍스트를 추출했거나 더 높은 신뢰도를 가진 경우 업데이트
            #             if len(current_results) > len(best_results) or (
            #                 len(current_results) == len(best_results) and 
            #                 sum(item["confidence"] for item in current_results) > 
            #                 sum(item["confidence"] for item in best_results)
            #             ):
            #                 best_results = current_results
            #                 logger.info(f"전처리 방법 {i}에서 더 좋은 결과 발견: {len(current_results)} 항목")
            #         except Exception as e:
            #             logger.warning(f"전처리 방법 {i}에서 오류 발생: {e}")
                
            #     ocr_results = best_results
                        
            # 텍스트 항목 추출 및 정렬 (위쪽 텍스트부터)
            if ocr_results:
                # y좌표(세로) 기준으로 정렬
                ocr_results.sort(key=lambda x: sum(coord[1] for coord in x["bounding_box"]) / len(x["bounding_box"]))
            
            # 전체 텍스트 추출 (모든 인식된 텍스트를 공백으로 연결)
            all_text = " ".join([item["text"] for item in ocr_results])
            
            # 텍스트 후처리
            processed_text = self.post_process_text(all_text)
            
            # 개별 텍스트 항목 추출
            text_items = [item["text"] for item in ocr_results]
            
            # 결과 반환
            return {
                "success": True,
                "message": "OCR 처리 완료",
                "full_text": processed_text,
                "original_text": all_text,
                "text_items": text_items,
                "detailed_items": ocr_results,
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

    # 비동기 메소드는 유지 (기존 코드와의 호환성을 위해)
    async def download_image_from_s3_async(self, s3_url: str) -> np.ndarray:
        return self.download_image_from_s3(s3_url)
    
    async def extract_text_async(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return self.extract_text(image)
    
    async def process_image_async(self, s3_url: str) -> Dict[str, Any]:
        return self.process_image(s3_url)

ocr_service = OCRService()