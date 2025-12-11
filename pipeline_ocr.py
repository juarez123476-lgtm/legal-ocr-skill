#!/usr/bin/env python3
"""
Legal Document OCR Pipeline
Optimized for Brazilian legal documents in Portuguese

Features:
- PDF to image conversion (PyMuPDF)
- Advanced image preprocessing (OpenCV)
- Multi-engine OCR (PaddleOCR + EasyOCR fallback)
- Legal dictionary post-processing
- Document structure identification
- Quality validation

Author: SIGEDEC Team
Version: 1.0.0
License: MIT
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time

import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ocr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LegalDocumentOCRPipeline:
    """
    Complete OCR pipeline for legal documents
    """

    def __init__(
        self,
        use_gpu: bool = True,
        quality: str = 'standard',
        confidence_threshold: float = 0.3,
        use_fallback: bool = True
    ):
        """
        Initialize OCR pipeline

        Args:
            use_gpu: Use GPU acceleration if available
            quality: 'standard' or 'high' (affects preprocessing intensity)
            confidence_threshold: Minimum confidence to accept OCR result
            use_fallback: Use EasyOCR as fallback when PaddleOCR fails
        """
        self.use_gpu = use_gpu
        self.quality = quality
        self.confidence_threshold = confidence_threshold
        self.use_fallback = use_fallback

        # Check GPU availability
        try:
            import torch
            self.gpu_available = torch.cuda.is_available() if use_gpu else False
            if self.gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            self.gpu_available = False
            logger.warning("PyTorch not installed, GPU acceleration disabled")

        # Initialize PaddleOCR
        logger.info("Initializing PaddleOCR (Portuguese)...")
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 3.x API - removed deprecated parameters
            self.paddleocr = PaddleOCR(
                use_textline_orientation=True,  # Renamed from use_angle_cls
                lang='pt'  # CRITICAL: Portuguese, not Chinese!
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

        # Initialize EasyOCR as fallback
        self.easyocr = None
        if use_fallback:
            try:
                import easyocr
                logger.info("Initializing EasyOCR (fallback)...")
                self.easyocr = easyocr.Reader(['pt'], gpu=self.gpu_available)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR fallback: {e}")

        # Load legal dictionary
        self.legal_dict = self._load_legal_dictionary()
        logger.info(f"Loaded {len(self.legal_dict)} legal terms")

    def _load_legal_dictionary(self) -> Dict[str, str]:
        """Load legal terminology dictionary"""
        dict_path = Path(__file__).parent / 'legal_dictionary.json'

        if dict_path.exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Fallback to basic dictionary
            logger.warning("legal_dictionary.json not found, using basic dictionary")
            return {
                'açao': 'ação', 'apelacao': 'apelação', 'decisao': 'decisão',
                'fundamentaçao': 'fundamentação', 'sentenca': 'sentença',
                'tribunal': 'tribunal', 'juiz': 'juiz', 'recurso': 'recurso'
            }

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to images using PyMuPDF

        Args:
            pdf_path: Path to PDF file
            dpi: Target DPI (default 300)

        Returns:
            List of images as numpy arrays
        """
        logger.info(f"Converting PDF to images (DPI: {dpi})...")
        start_time = time.time()

        pdf = fitz.open(pdf_path)
        images = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # Calculate zoom factor to achieve target DPI
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Render page as pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to numpy array
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)

            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            images.append(img_array)
            logger.debug(f"Converted page {page_num + 1}/{len(pdf)}")

        pdf.close()
        elapsed = time.time() - start_time
        logger.info(f"PDF conversion completed: {len(images)} pages in {elapsed:.2f}s")

        return images

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing pipeline

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Skew correction using Hough transform
        try:
            edges = cv2.Canny(gray, 100, 200, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is not None and len(lines) > 0:
                angles = []
                for rho, theta in lines[:20, 0]:
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)

                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:  # Only rotate if significant skew
                        h, w = gray.shape
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1)
                        gray = cv2.warpAffine(
                            gray, M, (w, h),
                            borderMode=cv2.BORDER_REPLICATE,
                            flags=cv2.INTER_CUBIC
                        )
                        logger.debug(f"Skew corrected by {median_angle:.2f}°")
        except Exception as e:
            logger.debug(f"Skew correction skipped: {e}")

        # 3. Noise removal - Median blur
        gray = cv2.medianBlur(gray, 5)

        # 4. Contrast enhancement using CLAHE
        clip_limit = 4.0 if self.quality == 'high' else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 5. Optional: Local brightness adjustment for uneven lighting
        if self.quality == 'high':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            gray = cv2.divide(gray, background, scale=255)
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # 6. Binarization - Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 7. Morphological operations
        # Dilation to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)

        return binary

    def ocr_image(
        self,
        image: np.ndarray,
        page_num: int
    ) -> Tuple[Optional[List], str, float]:
        """
        Apply OCR with fallback mechanism

        Args:
            image: Preprocessed image
            page_num: Page number (for logging)

        Returns:
            Tuple of (ocr_result, source_engine, confidence)
        """
        # Try PaddleOCR first
        try:
            result = self.paddleocr.ocr(image, cls=True)

            if result and result[0]:
                # Calculate average confidence
                confidences = []
                for line in result[0]:
                    if line and len(line) > 1:
                        text, conf = line[1]
                        confidences.append(conf)

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                if avg_confidence >= self.confidence_threshold:
                    logger.debug(f"Page {page_num}: PaddleOCR confidence {avg_confidence:.2f}")
                    return result, 'paddleocr', avg_confidence
                else:
                    logger.warning(
                        f"Page {page_num}: PaddleOCR low confidence {avg_confidence:.2f}"
                    )
            else:
                logger.warning(f"Page {page_num}: PaddleOCR returned empty result")

        except Exception as e:
            logger.error(f"Page {page_num}: PaddleOCR error: {e}")

        # Fallback to EasyOCR
        if self.use_fallback and self.easyocr:
            try:
                logger.info(f"Page {page_num}: Using EasyOCR fallback")
                result = self.easyocr.readtext(image)

                if result:
                    # Convert EasyOCR format to match PaddleOCR
                    confidences = [item[2] for item in result]
                    avg_confidence = sum(confidences) / len(confidences)

                    logger.debug(f"Page {page_num}: EasyOCR confidence {avg_confidence:.2f}")
                    return result, 'easyocr', avg_confidence

            except Exception as e:
                logger.error(f"Page {page_num}: EasyOCR error: {e}")

        return None, 'error', 0.0

    def extract_text_from_ocr(
        self,
        ocr_result: List,
        source: str
    ) -> str:
        """
        Extract text from OCR result (handles both PaddleOCR and EasyOCR formats)
        """
        if source == 'paddleocr':
            lines = []
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line and len(line) > 1:
                        text, conf = line[1]
                        lines.append(text)
            return '\n'.join(lines)

        elif source == 'easyocr':
            return '\n'.join([item[1] for item in ocr_result])

        return ""

    def post_process_text(self, text: str) -> str:
        """
        Post-process OCR text with legal dictionary
        """
        if not text:
            return text

        words = text.split()
        corrected = []

        for word in words:
            word_clean = word.lower().strip('.,;:!?()[]{}"""\'')

            # Direct dictionary match
            if word_clean in self.legal_dict:
                replacement = self.legal_dict[word_clean]
                # Preserve original punctuation
                suffix = word[len(word_clean):]
                corrected.append(replacement + suffix)
            else:
                # Fuzzy matching for close terms
                from difflib import SequenceMatcher

                best_match = None
                best_ratio = 0

                for legal_term in self.legal_dict.keys():
                    ratio = SequenceMatcher(None, word_clean, legal_term).ratio()
                    if ratio > best_ratio and ratio > 0.85:
                        best_ratio = ratio
                        best_match = legal_term

                if best_match:
                    corrected.append(self.legal_dict[best_match])
                    logger.debug(f"Fuzzy matched: {word_clean} → {best_match}")
                else:
                    corrected.append(word)

        return ' '.join(corrected)

    def identify_document_structure(self, text: str) -> Dict[str, str]:
        """
        Identify sections in legal document
        """
        structure = {
            'header': '',
            'preambulo': '',
            'relatorio': '',
            'fundamentacao': '',
            'dispositivo': '',
            'assinaturas': ''
        }

        # Section markers (case-insensitive)
        markers = {
            'relatorio': [
                'relatório', 'relatório:', 'i -', 'i.', 'vistos',
                'trata-se de', 'cuida-se de'
            ],
            'fundamentacao': [
                'fundamentação', 'fundamentação:', 'ii -', 'ii.',
                'é o relatório', 'decido', 'fundamento'
            ],
            'dispositivo': [
                'dispositivo', 'dispositivo:', 'iii -', 'iii.',
                'pelo exposto', 'ante o exposto', 'diante do exposto',
                'isto posto', 'resolve:', 'decide:',
                'homologo', 'julgo procedente', 'julgo improcedente'
            ],
            'assinaturas': [
                'assinado', 'juiz', 'juíz', 'desembargador',
                '__________', '___', 'assinatura'
            ]
        }

        lines = text.split('\n')
        current_section = 'header'
        section_content = {key: [] for key in structure.keys()}

        for line in lines:
            line_lower = line.lower().strip()

            # Check for section markers
            section_found = False
            for section, section_markers in markers.items():
                if any(marker in line_lower for marker in section_markers):
                    current_section = section
                    section_found = True
                    break

            section_content[current_section].append(line)

        # Join lines for each section
        for section in structure.keys():
            structure[section] = '\n'.join(section_content[section])

        return structure

    def validate_quality(self, text: str) -> Dict:
        """
        Validate OCR output quality
        """
        issues = []
        warnings = []
        confidence = 100

        if not text or len(text.strip()) == 0:
            return {
                'confidence': 0,
                'issues': ['No text extracted'],
                'warnings': [],
                'requires_review': True
            }

        # Check for common OCR confusion
        if text.count('O') > text.count('0') * 5 and '0' in text:
            issues.append("High O/0 confusion ratio")
            confidence -= 20

        if text.count('l') > text.count('1') * 5 and '1' in text:
            issues.append("High l/1 confusion ratio")
            confidence -= 20

        if text.count('S') > text.count('5') * 5 and '5' in text:
            issues.append("High S/5 confusion ratio")
            confidence -= 20

        # Check for expected legal elements
        legal_required = ['juiz', 'tribunal', 'processo']
        missing = []
        for required in legal_required:
            if required not in text.lower():
                missing.append(required)

        if missing:
            warnings.append(f"Missing expected elements: {', '.join(missing)}")
            confidence -= 10 * len(missing)

        # Check average word length
        words = text.split()
        if words:
            avg_length = sum(len(w) for w in words) / len(words)
            if avg_length < 3:
                issues.append("Average word length too short - possible poor OCR")
                confidence -= 15

        confidence = max(0, min(100, confidence))

        return {
            'confidence': confidence,
            'issues': issues,
            'warnings': warnings,
            'requires_review': confidence < 70
        }

    def process_legal_document(
        self,
        pdf_path: str,
        dpi: int = 300,
        output_json_path: Optional[str] = None
    ) -> Dict:
        """
        Complete OCR pipeline for a legal document

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion (default 300)
            output_json_path: Optional path to save JSON output

        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        logger.info(f"Processing: {pdf_path}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi)

        # Initialize results
        results = {
            'filename': Path(pdf_path).name,
            'timestamp': datetime.now().isoformat(),
            'pages': [],
            'metadata': {
                'total_pages': len(images),
                'gpu_used': self.gpu_available,
                'quality_mode': self.quality,
                'dpi': dpi
            }
        }

        fallback_pages = []

        # Process each page
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{len(images)}")

            try:
                # Preprocess
                processed = self.preprocess_image(image)

                # OCR
                ocr_result, source, confidence = self.ocr_image(processed, page_num)

                if source == 'easyocr':
                    fallback_pages.append(page_num)

                if ocr_result:
                    # Extract text
                    raw_text = self.extract_text_from_ocr(ocr_result, source)

                    # Post-process
                    corrected_text = self.post_process_text(raw_text)

                    # Validate
                    validation = self.validate_quality(corrected_text)

                    # Identify structure
                    structure = self.identify_document_structure(corrected_text)

                    results['pages'].append({
                        'page_num': page_num,
                        'text': corrected_text,
                        'confidence': float(confidence),
                        'source': source,
                        'validation': validation,
                        'structure_detected': any(structure.values())
                    })
                else:
                    results['pages'].append({
                        'page_num': page_num,
                        'error': 'OCR failed',
                        'source': 'error'
                    })

            except Exception as e:
                logger.error(f"Page {page_num} processing error: {e}")
                results['pages'].append({
                    'page_num': page_num,
                    'error': str(e),
                    'source': 'error'
                })

        # Compile full text
        full_text = '\n\n'.join([
            p['text'] for p in results['pages']
            if 'text' in p
        ])

        results['full_text'] = full_text

        # Document-level structure identification
        results['document_structure'] = self.identify_document_structure(full_text)

        # Quality summary
        confidences = [p['confidence'] for p in results['pages'] if 'confidence' in p]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        all_issues = []
        pages_requiring_review = 0
        for p in results['pages']:
            if 'validation' in p:
                all_issues.extend(p['validation']['issues'])
                if p['validation']['requires_review']:
                    pages_requiring_review += 1

        results['quality_summary'] = {
            'avg_confidence': round(avg_confidence, 2),
            'total_issues': len(all_issues),
            'pages_requiring_review': pages_requiring_review,
            'overall_quality': self._quality_rating(avg_confidence)
        }

        results['metadata']['fallback_used_pages'] = fallback_pages
        results['metadata']['processing_time_seconds'] = round(time.time() - start_time, 2)

        # Save to file if requested
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_json_path}")

        logger.info(
            f"Processing completed: {len(images)} pages, "
            f"avg confidence: {avg_confidence:.2f}, "
            f"quality: {results['quality_summary']['overall_quality']}"
        )

        return results

    def _quality_rating(self, avg_confidence: float) -> str:
        """Rate overall quality"""
        if avg_confidence >= 0.9:
            return 'excellent'
        elif avg_confidence >= 0.8:
            return 'good'
        elif avg_confidence >= 0.7:
            return 'acceptable'
        else:
            return 'poor'


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Legal Document OCR Pipeline - Extract text from scanned PDFs'
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input PDF file path'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path (default: <input>_ocr.json)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for image conversion (default: 300)'
    )

    parser.add_argument(
        '--quality',
        type=str,
        choices=['standard', 'high'],
        default='standard',
        help='Processing quality mode (default: standard)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )

    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help='Disable EasyOCR fallback'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold (default: 0.3)'
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Set output path
    output_path = args.output
    if not output_path:
        output_path = str(Path(args.input).with_suffix('')) + '_ocr.json'

    # Create pipeline
    pipeline = LegalDocumentOCRPipeline(
        use_gpu=not args.no_gpu,
        quality=args.quality,
        confidence_threshold=args.confidence_threshold,
        use_fallback=not args.no_fallback
    )

    # Process document
    try:
        results = pipeline.process_legal_document(
            args.input,
            dpi=args.dpi,
            output_json_path=output_path
        )

        # Print summary
        print("\n" + "="*80)
        print("OCR PROCESSING SUMMARY")
        print("="*80)
        print(f"Document: {results['filename']}")
        print(f"Pages: {results['metadata']['total_pages']}")
        print(f"Processing time: {results['metadata']['processing_time_seconds']}s")
        print(f"Average confidence: {results['quality_summary']['avg_confidence']}")
        print(f"Overall quality: {results['quality_summary']['overall_quality'].upper()}")
        print(f"Pages requiring review: {results['quality_summary']['pages_requiring_review']}")

        if results['metadata'].get('fallback_used_pages'):
            print(f"Fallback used on pages: {results['metadata']['fallback_used_pages']}")

        print(f"\nResults saved to: {output_path}")
        print("="*80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
