//! Full OCR pipeline orchestration.
//!
//! The pipeline mirrors PaddleOCR's flow:
//!
//! 1. (optional) document orientation classification
//! 2. (optional) document unwarping / rectification
//! 3. text detection
//! 4. per text-line crop: (optional) text-line orientation, then recognition
//!
//! Individual modules (detection, recognition, orientation, unwarping) are
//! implemented in later phases; this module will assemble them.
