export type PoseType = {
  frame: number;
  timestamp: number;
  landmarks: Record<string, number[]>;
};
