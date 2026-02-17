import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from predict import process_video


class TestPredict(unittest.TestCase):
    @patch("predict.YOLO")
    @patch("predict.sv")
    @patch("predict.reader")
    def test_process_video(self, mock_reader, mock_sv, mock_yolo):
        # Arrange
        source_weights_path = "dummy_weights.pt"
        source_video_path = "dummy_video.mp4"
        target_video_path = "output_video.mp4"

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_sv.get_video_frames_generator.return_value = [mock_frame]

        mock_video_info = MagicMock()
        mock_video_info.total_frames = 1
        mock_sv.VideoInfo.from_video_path.return_value = mock_video_info

        mock_sink = MagicMock()
        mock_sv.VideoSink.return_value.__enter__.return_value = mock_sink

        mock_reader.readtext.return_value = ["mock_text"]

        mock_results = MagicMock()
        mock_model.return_value = [mock_results]

        mock_detections = MagicMock()
        mock_detections.xyxy = np.empty((0, 4))
        mock_detections.confidence = np.empty((0,))
        mock_detections.tracker_id = np.empty((0,))
        mock_sv.Detections.from_ultralytics.return_value = mock_detections

        mock_tracker = MagicMock()
        mock_sv.ByteTrack.return_value = mock_tracker
        mock_tracker.update_with_detections.return_value = mock_detections
        mock_sv.pad_boxes.return_value = mock_detections.xyxy

        # Act
        process_video(source_weights_path, source_video_path, target_video_path)

        # Assert
        mock_yolo.assert_called_once_with(source_weights_path)
        mock_sv.get_video_frames_generator.assert_called_once_with(
            source_path=source_video_path
        )
        mock_sv.VideoInfo.from_video_path.assert_called_once_with(
            video_path=source_video_path
        )
        mock_sv.VideoSink.assert_called_once_with(
            target_path=target_video_path, video_info=mock_video_info
        )
        mock_sink.write_frame.assert_called_once()


if __name__ == "__main__":
    unittest.main()
