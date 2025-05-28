import { create } from "zustand";
import { createSetter } from "../Utils/functions";
import type { PoseType } from "../Utils/types";

const userPoseRef = { current: [] as PoseType[] };

interface StoreType {
  collect: boolean;
  userPose: PoseType[];
  userPoseRef: typeof userPoseRef;
  setCollect: (collect: boolean | ((prev: boolean) => boolean)) => void;
  setUserPose: (pose: PoseType[] | ((prev: PoseType[]) => PoseType[])) => void;
}

export const useStore = create<StoreType>((set, get) => ({
  userPoseRef,
  userPose: [],
  collect: false,
  setCollect: createSetter<StoreType>(set)("collect"),
  setUserPose: (update) => {
    const newValue = typeof update === "function" ? update(get().userPose) : update;
    userPoseRef.current = newValue;
    set({ userPose: newValue });
  },
}));
