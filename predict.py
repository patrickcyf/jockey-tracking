import argparse
import easyocr

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

TIMER_TOPLEFT = [1650, 30]
TIMER_BOTTOMRIGHT = [1850, 135]

def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> None:
    reader = easyocr.Reader(['en'])
    model = YOLO(source_weights_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            ocr_results = reader.readtext(
                frame[TIMER_TOPLEFT[1]:TIMER_BOTTOMRIGHT[1], TIMER_TOPLEFT[0]:TIMER_BOTTOMRIGHT[0]], 
                allowlist ='0123456789.:', detail=0
            )
            if ocr_results:
                results = model(
                  frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
                )[0]
                detections = sv.Detections.from_ultralytics(results)
                detections.xyxy = sv.pad_boxes(xyxy=detections.xyxy, px=0, py=40)
                detections = tracker.update_with_detections(detections)

                labels = [
                  f"#{tracker_id} {confidence:0.2f}"
                  for tracker_id, confidence
                  in zip(detections.tracker_id, detections.confidence)
                ]

                annotated_frame = box_annotator.annotate(
                  scene=frame.copy(), detections=detections
                )
                frame = label_annotator.annotate(
                  scene=annotated_frame, detections=detections, labels=labels
                )

            sink.write_frame(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detector for jockey during horse racing"
    )
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.5,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.5, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
