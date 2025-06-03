import { useCameraCapture } from "../Hooks/useCameraCapture";

export default function PoseCamera() {
  const { videoRef, canvasRef } = useCameraCapture();

  return (
    <div style={{ position: "relative" }} className="w-[50%]">
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0 }}
        className="w-full"
      />
    </div>
  );
}
