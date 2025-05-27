import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import Model from "./Model";

export default function Scene() {
  return (
    <Canvas camera={{ position: [0, 0, 2], fov: 60 }}>
      <ambientLight intensity={1} />
      <Model />
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
    </Canvas>
  );
}
