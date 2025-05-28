import { useCameraCapture } from "../Hooks/useCameraCapture";

export default function PoseCamera() {
  const { videoRef, canvasRef } = useCameraCapture();

  return (
    <div style={{ position: "relative" }}>
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0 }}
        width={640}
        height={480}
        className="w-[50vw]"
      />
    </div>
  );
}
