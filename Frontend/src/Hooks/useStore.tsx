import { create } from "zustand";
import { createSetter } from "../Utils/functions";
import type { PoseType } from "../Utils/types";

interface StoreType {
  collect: boolean;
  userPose: PoseType;
  setCollect: (collect: boolean | ((prev: boolean) => boolean)) => void;
  setUserPose: (pose: PoseType | ((prev: PoseType) => PoseType)) => void;
}

export const useStore = create<StoreType>((set, get) => ({
  userPose: {} as PoseType,
  collect: true,
  setCollect: createSetter<StoreType>(set)("collect"),
  setUserPose: (update) => {
    const newValue = typeof update === "function" ? update(get().userPose) : update;
    set({ userPose: newValue });
  },
}));
