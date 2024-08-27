import logging
import os

from dvgutils import setup_logger, load_config
from dvgutils.modules import ImageCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.pipeline import Pipeline, ShowImagePipe, MetricsPipe, ProgressPipe

from pipeline import (
    DetectFacePipe, TrackObjectPipe, HeadPosePipe, VisPipe,
    YawnDetectionPipe, LoggingPipe, SaveImagePipe, PoseMetricsPipe,
    CaptureImagePipe, AlarmPipe
)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/app_config.yml",
                        help="Path to the input configuration file (default: config/app_config.yml)")
    parser.add_argument("-cfo", "--conf-overwrites", nargs="+", type=str)
    parser.add_argument("-o", "--output", type=str,
                        help="path to output directory")
    parser.add_argument("--output-csv", type=str,
                        help="path to CSV output directory")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")

    parser.add_argument("--delay", default=0, type=int, help="Delay between visualization frames")

    return vars(parser.parse_args())


def run(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"], args["conf_overwrites"])
    
    # Setup pipeline steps
    capture_image_pipe = CaptureImagePipe(conf["imageCapture"])
    detect_face_pipe = DetectFacePipe(conf["faceDetector"])
    track_faces_pipe = TrackObjectPipe(conf["faceTracker"])
    head_pose_pipe = HeadPosePipe(conf["headPoseEstimator"])
    yawn_detection_pipe = YawnDetectionPipe(conf["yawnDetector"])
    vis_pipe = VisPipe('image', 'vis_image', conf['visPipe'], head_pose_pipe.pe) if "visPipe" in conf else None
    show_image_pipe = ShowImagePipe("vis_image", "Video", delay=args["delay"]) if args["display"] else None
    # save_image_pipe = SaveImagePipe("vis_image", output_path=args["output"],
    save_image_pipe = SaveImagePipe("vis_image", output_path=args["output"], overwrite=True) if args.get("output", None) else None
    pose_metrics_pipe = PoseMetricsPipe(conf['metrics'], 'tracked_faces')
    alarm_pipe = AlarmPipe(conf['alarms'], 'tracked_faces')
    save_csv = LoggingPipe('tracked_faces', output_path=args['output_csv']) if args.get('output_csv', None) else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_image_pipe)
    pipeline.map(detect_face_pipe)
    pipeline.map(track_faces_pipe)
    pipeline.map(head_pose_pipe)
    pipeline.map(yawn_detection_pipe)
    pipeline.map(pose_metrics_pipe)
    pipeline.map(alarm_pipe)
    pipeline.map(vis_pipe)
    pipeline.map(show_image_pipe)
    pipeline.map(save_image_pipe)
    pipeline.map(save_csv)
    pipeline.map(metrics_pipe)
    pipeline.map(progress_pipe)

    # Process pipeline
    try:
        logger.info("Capturing...")
        pipeline.run()

        logger.info(f"{len(metrics_pipe.metrics)} it, "
                    f"{metrics_pipe.metrics.elapsed():.3f} s, "
                    f"{metrics_pipe.metrics.sec_per_iter():.3f} s/it, "
                    f"{metrics_pipe.metrics.iter_per_sec():.2f} it/s")
    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup pipeline resources
        pipeline.close()


if __name__ == "__main__":
    setup_logger()

    args = parse_args()

    run(args)