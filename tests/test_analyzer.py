import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.analyzer import BlurAnalyzer

@pytest.fixture
def mock_video_capture():
    with patch('cv2.VideoCapture') as mock_cap:
        yield mock_cap

@pytest.fixture
def mock_geometry():
    # Patch where it is imported/used, not where it is defined
    with patch('src.core.analyzer.GeometryProcessor') as mock_geo:
        yield mock_geo

@pytest.fixture
def mock_image_utils():
    # Patch where it is imported/used, not where it is defined
    with patch('src.core.analyzer.ImageUtils') as mock_utils:
        yield mock_utils

def test_analyze_sample(mock_video_capture, mock_geometry, mock_image_utils):
    # Setup Mocks
    mock_cap_instance = mock_video_capture.return_value
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = lambda prop: 100 if prop == 7 else 0 # 7 is FRAME_COUNT
    
    # Create a dummy frame
    dummy_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mock_cap_instance.read.return_value = (True, dummy_frame)
    
    # Mock Geometry
    mock_geometry.generate_views.return_value = [("View_0", 0, 0, 0), ("View_1", 90, 0, 0)]
    mock_geometry.create_rectilinear_map.return_value = (np.zeros((10,10), dtype=np.float32), np.zeros((10,10), dtype=np.float32))
    
    # Mock ImageUtils
    mock_image_utils.calculate_blur_score.side_effect = [100.0, 50.0]
    
    # Test
    settings = {'resolution': 100, 'camera_count': 2}
    result = BlurAnalyzer.analyze_sample("dummy.mp4", settings)
    
    # Assertions
    assert result['average'] == 75.0
    assert result['min'] == 50.0
    assert result['max'] == 100.0
    assert len(result['details']) == 2
    assert result['details'][0] == ("View_0", 100.0)
    assert result['details'][1] == ("View_1", 50.0)

def test_analyze_sample_video_fail(mock_video_capture):
    mock_cap_instance = mock_video_capture.return_value
    mock_cap_instance.isOpened.return_value = False
    
    with pytest.raises(IOError):
        BlurAnalyzer.analyze_sample("bad.mp4", {})
