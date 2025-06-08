import { create } from "zustand";
import { createSetter } from "../Utils/functions";

const userPoseRef = { current: {} as Record<number, Record<string, number[]>> };
type Collect = "" | "Left" | "Right";

interface StoreType {
  collect: Collect;
  currentPose: Record<string, number[]>;
  userPoseRef: typeof userPoseRef;
  setCollect: (collect: Collect | ((prev: Collect) => Collect)) => void;
  setCurrentPose: (
    pose: Record<string, number[]> | ((prev: Record<string, number[]>) => Record<string, number[]>)
  ) => void;
}

export const useStore = create<StoreType>((set, get) => ({
  userPoseRef,
  collect: "" as Collect,
  currentPose: {} as Record<string, number[]>,
  setCollect: createSetter<StoreType>(set)("collect"),
  setCurrentPose: (update) => {
    const newValue = typeof update === "function" ? update(get().currentPose) : update;
    set({ currentPose: newValue });
  },
}));
