import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import Model from "./Model";
import { useCameraCapture } from "../../Hooks/useCameraCapture";

const CV = ({ text }: { text: string }) => {
  const { videoRef, canvasRef } = useCameraCapture();

  return (
    <div className="flex gap-4 relative w-full">
      <div className="flex-1 min-w-0 aspect-[4/3] bg-base-300 shadow-lg rounded-lg p-4">
        <Canvas camera={{ position: [0, 0, 2], fov: 60 }}>
          <Model />
          <ambientLight intensity={1} />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>
      <div className="flex-1 min-w-0 aspect-[4/3] bg-base-300 shadow-lg rounded-lg p-4">
        <video ref={videoRef} style={{ display: "none" }} />
        <canvas
          ref={canvasRef}
          className="w-full h-full rounded-md"
        />
      </div>
      <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
        <h1 className="flex-1 text-[7vw] z-50 text-center sporting-outline">{text}</h1>
      </div>
    </div>
  );
};

export default CV;
