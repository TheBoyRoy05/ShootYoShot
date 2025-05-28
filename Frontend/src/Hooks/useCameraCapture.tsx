import { useRef, useEffect } from "react";
import { Pose, POSE_CONNECTIONS, POSE_LANDMARKS } from "@mediapipe/pose";
import * as cam from "@mediapipe/camera_utils";
import type { PoseType } from "../Utils/types";
import { useStore } from "./useStore";

const KEYPOINTS = [
  "LEFT_SHOULDER",
  "RIGHT_SHOULDER",
  "LEFT_ELBOW",
  "RIGHT_ELBOW",
  "LEFT_WRIST",
  "RIGHT_WRIST",
  "LEFT_HIP",
  "RIGHT_HIP",
  "LEFT_KNEE",
  "RIGHT_KNEE",
  "LEFT_ANKLE",
  "RIGHT_ANKLE",
  "LEFT_THUMB",
  "RIGHT_THUMB",
  "LEFT_INDEX",
  "RIGHT_INDEX",
  "LEFT_PINKY",
  "RIGHT_PINKY",
];

export function useCameraCapture() {
  const { collect, setUserPose } = useStore();
  const collectRef = useRef(collect); // track the latest collect value
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<cam.Camera | null>(null);
  const frameCounter = useRef(0);
  const startTime = useRef<number | null>(null);

  useEffect(() => {
    collectRef.current = collect; // keep ref in sync with prop
  }, [collect]);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas?.getContext("2d");
      const landmarks = results.poseLandmarks;
      const worldLandmarks = results.poseWorldLandmarks;

      if (!canvas || !ctx || !video || !landmarks) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];
        ctx.beginPath();
        ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
        ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
        ctx.stroke();
      }

      for (const landmark of landmarks) {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      }

      if (collectRef.current && worldLandmarks) {
        const now = performance.now();
        if (startTime.current === null) startTime.current = now;

        const data: PoseType["landmarks"] = {};
        for (const key of KEYPOINTS) {
          const index = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
          const lm = worldLandmarks[index];
          data[key] = [lm.x, -lm.y, -lm.z];
        }

        const timestamp = ((now - startTime.current) / 1000).toFixed(3);
        setUserPose((prev) => [
          ...prev,
          {
            frame: frameCounter.current++,
            timestamp: parseFloat(timestamp),
            landmarks: data,
          },
        ]);
      }

      if (!collectRef.current) {
        frameCounter.current = 0;
        startTime.current = null;
      }
    });

    if (videoRef.current) {
      cameraRef.current = new cam.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            await pose.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });
      cameraRef.current.start();
    }

    return () => {
      cameraRef.current?.stop();
    };
  }, []);

  return { videoRef, canvasRef, cameraRef, frameCounter, startTime };
}
