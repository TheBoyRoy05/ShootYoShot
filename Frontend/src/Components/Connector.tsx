import { useMemo } from 'react';
import { Quaternion, Vector3 } from 'three';

interface ConnectorProps {
  start: Vector3;
  end: Vector3;
  color?: string;
  radius?: number;
}

export const Connector: React.FC<ConnectorProps> = ({
  start,
  end,
  color = 'skyblue',
  radius = 0.08,
}) => {
  const position = useMemo(() => {
    return new Vector3().addVectors(start, end).multiplyScalar(0.5);
  }, [start, end]);

  const rotation = useMemo(() => {
    const dir = new Vector3().subVectors(end, start).normalize();
    const axis = new Vector3(0, 1, 0);
    const quaternion = new Quaternion().setFromUnitVectors(axis, dir);
    return quaternion;
  }, [start, end]);

  return (
    <mesh position={position} quaternion={rotation}>
      <cylinderGeometry args={[radius, radius, start.distanceTo(end)]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};
