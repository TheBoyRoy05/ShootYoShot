import { create } from "zustand";
import { createSetter } from "../Utils/functions";
import type { PoseType } from "../Utils/types";

const userPoseRef = { current: [] as PoseType[] };

interface StoreType {
  collect: boolean;
  currentPose: Record<string, number[]>;
  userPoseRef: typeof userPoseRef;
  setCollect: (collect: boolean | ((prev: boolean) => boolean)) => void;
  setCurrentPose: (pose: Record<string, number[]> | ((prev: Record<string, number[]>) => Record<string, number[]>)) => void;
}

export const useStore = create<StoreType>((set, get) => ({
  userPoseRef,
  collect: false,
  currentPose: {} as Record<string, number[]>,
  setCollect: createSetter<StoreType>(set)("collect"),
  setCurrentPose: (update) => {
    const newValue = typeof update === "function" ? update(get().currentPose) : update;
    set({ currentPose: newValue });
  },
}));
