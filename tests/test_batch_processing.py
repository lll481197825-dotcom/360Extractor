import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.job import Job
from core.processor import ProcessingWorker

@pytest.fixture
def mock_cv2():
    with patch('core.processor.cv2') as mock_cv2:
        # Define constants
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_FRAME_COUNT = 2
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.INTER_LINEAR = 1
        mock_cv2.BORDER_WRAP = 1
        
        # Factory for video captures so each job gets a fresh iterator
        def create_mock_cap(*args, **kwargs):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 100.0,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 960.0
            }.get(prop, 0.0)
            
            # Mock reading frames: 5 frames then stop
            # returns (ret, frame)
            # Create a list of return values
            frames = [(True, MagicMock()) for _ in range(5)]
            frames.append((False, None))
            mock_cap.read.side_effect = frames
            
            mock_cap.release = MagicMock()
            return mock_cap
            
        mock_cv2.VideoCapture.side_effect = create_mock_cap
        
        # Mock remap to avoid errors with MagicMock inputs
        mock_cv2.remap.return_value = MagicMock()
        
        yield mock_cv2

@pytest.fixture
def mock_file_manager():
    with patch('core.processor.FileManager') as mock_fm:
        yield mock_fm

@pytest.fixture
def mock_ai_service():
    with patch('core.processor.AIService') as mock_ai:
        # Ensure the instance returned by the constructor is a mock that we can track
        mock_instance = MagicMock()
        mock_instance.process_image.return_value = (MagicMock(), False) # default return
        mock_ai.return_value = mock_instance
        yield mock_ai

def test_batch_processing_flow(mock_cv2, mock_file_manager, mock_ai_service):
    # Setup Jobs
    job1 = Job(file_path="/tmp/video1.mp4", settings={
        'interval_unit': 'Frames', 'interval_value': 1, 'camera_count': 2
    })
    job2 = Job(file_path="/tmp/video2.mp4", settings={
        'interval_unit': 'Frames', 'interval_value': 1, 'camera_count': 2
    })
    
    jobs = [job1, job2]
    
    # Initialize Worker
    worker = ProcessingWorker(jobs)
    
    # Connect signals to verifying mocks
    mock_job_started = MagicMock()
    mock_job_finished = MagicMock()
    mock_finished = MagicMock()
    
    worker.job_started.connect(mock_job_started)
    worker.job_finished.connect(mock_job_finished)
    worker.finished.connect(mock_finished)
    
    # Run synchronously
    worker.run()
    
    # Assertions
    assert mock_job_started.call_count == 2
    assert mock_job_finished.call_count == 2
    assert mock_finished.call_count == 1
    
    # Check call arguments
    mock_job_started.assert_any_call(0)
    mock_job_started.assert_any_call(1)
    mock_job_finished.assert_any_call(0)
    mock_job_finished.assert_any_call(1)

def test_job_settings_usage(mock_cv2, mock_file_manager, mock_ai_service):
    # Test if worker respects per-job settings (e.g. AI mode)
    # Ensure we process every frame to match mock length
    job_no_ai = Job(file_path="/tmp/video1.mp4", settings={
        'ai_mode': 'None', 'camera_count': 2,
        'interval_unit': 'Frames', 'interval_value': 1
    })
    job_with_ai = Job(file_path="/tmp/video2.mp4", settings={
        'ai_mode': 'Skip Frame', 'camera_count': 2,
        'interval_unit': 'Frames', 'interval_value': 1
    })
    
    jobs = [job_no_ai, job_with_ai]
    
    worker = ProcessingWorker(jobs)
    
    # The worker.ai_service attribute is set in __init__ using the class we mocked
    # Let's get that specific instance
    mock_ai_instance = worker.ai_service
    
    worker.run()
    
    # We expect process_image to be called only for the frames of job_with_ai
    # Each job processes 5 frames (from mock_cv2 fixture)
    # job_no_ai: 0 calls (AI mode None)
    # job_with_ai: 5 frames * 2 views = 10 calls
    
    assert mock_ai_instance.process_image.call_count == 10
