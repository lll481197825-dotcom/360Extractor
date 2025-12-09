import cv2
import numpy as np
import pytest
from utils.image_utils import ImageUtils

def create_synthetic_image(blur=False):
    """
    Creates a synthetic image.
    If blur is True, applies Gaussian blur to make it blurry.
    """
    # Create a black image with white random noise/edges
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Draw some rectangles and circles to create edges
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)
    
    # Add random noise
    noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
    
    if blur:
        # Apply strong blur
        img = cv2.GaussianBlur(img, (21, 21), 0)
        
    return img

def test_blur_score_sharp():
    """Test that a sharp image has a relatively high blur score."""
    sharp_img = create_synthetic_image(blur=False)
    score = ImageUtils.calculate_blur_score(sharp_img)
    
    # Sharp image with edges and noise should have a high variance
    # The exact value depends on the image content, but should be significantly > 0
    assert score > 100.0, f"Expected sharp image to have score > 100, got {score}"

def test_blur_score_blurry():
    """Test that a blurry image has a lower blur score than a sharp one."""
    sharp_img = create_synthetic_image(blur=False)
    blurry_img = create_synthetic_image(blur=True)
    
    score_sharp = ImageUtils.calculate_blur_score(sharp_img)
    score_blurry = ImageUtils.calculate_blur_score(blurry_img)
    
    assert score_blurry < score_sharp, f"Expected blurry image score ({score_blurry}) to be lower than sharp image score ({score_sharp})"
    assert score_blurry < 100.0, f"Expected blurry image score to be low (< 100), got {score_blurry}"

def test_blur_score_none():
    """Test that None input returns 0.0."""
    score = ImageUtils.calculate_blur_score(None)
    assert score == 0.0

def test_blur_score_grayscale():
    """Test that it handles grayscale images correctly."""
    img = create_synthetic_image(blur=False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    score = ImageUtils.calculate_blur_score(gray)
    assert score > 100.0