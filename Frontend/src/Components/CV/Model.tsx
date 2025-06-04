import { Connector } from "./Connector";
import { Vector3 } from "three";
import { useStore } from "../../Hooks/useStore";

const Model = () => {
  const { currentPose } = useStore();
  if (Object.keys(currentPose).length === 0) return null;
  
  type LandmarkKey = keyof typeof currentPose;
  const getPoint = (idx: LandmarkKey) => new Vector3(...currentPose[idx]);
  
  return (
    <>
      {Object.values(currentPose).map((point, i) => (
        <mesh key={i} position={new Vector3(...point)}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshStandardMaterial color="orange" transparent opacity={0.7} />
        </mesh>
      ))}
      <Connector start={getPoint("LEFT_SHOULDER")} end={getPoint("RIGHT_SHOULDER")} />
      <Connector start={getPoint("LEFT_SHOULDER")} end={getPoint("LEFT_ELBOW")} />
      <Connector start={getPoint("LEFT_ELBOW")} end={getPoint("LEFT_WRIST")} />
      <Connector start={getPoint("RIGHT_SHOULDER")} end={getPoint("RIGHT_ELBOW")} />
      <Connector start={getPoint("RIGHT_ELBOW")} end={getPoint("RIGHT_WRIST")} />
      <Connector start={getPoint("LEFT_HIP")} end={getPoint("RIGHT_HIP")} />
      <Connector start={getPoint("LEFT_HIP")} end={getPoint("LEFT_KNEE")} />
      <Connector start={getPoint("LEFT_KNEE")} end={getPoint("LEFT_ANKLE")} />
      <Connector start={getPoint("RIGHT_HIP")} end={getPoint("RIGHT_KNEE")} />
      <Connector start={getPoint("RIGHT_KNEE")} end={getPoint("RIGHT_ANKLE")} />
      {/* <Connector
        start={getPoint("NOSE")}
        end={getPoint("LEFT_SHOULDER").clone().add(getPoint("RIGHT_SHOULDER")).multiplyScalar(0.5)}
      /> */}
      {/* <Connector start={getPoint("LEFT_SHOULDER")} end={getPoint("LEFT_HIP")} />
      <Connector start={getPoint("RIGHT_SHOULDER")} end={getPoint("RIGHT_HIP")} /> */}
    </>
  );
};

export default Model;
