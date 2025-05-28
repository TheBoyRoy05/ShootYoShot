import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import PoseCamera from "./Components/PoseCamera";
import Scene from "./Components/Scene";
import { useState } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";

const App = () => {
  const { collect, setCollect, userPoseRef, setUserPose } = useStore();
  const { http } = useHTTP();
  const [text, setText] = useState("");

  const run = async () => {
    setText("Ready...");
    await sleep(1000);
    setText("Set...");
    await sleep(1000);
    setText("Go!");
    await sleep(1000);
    setText("");

    setCollect(true);
    await sleep(5000);
    await http({
      url: "/score",
      method: "POST",
      body: { pose: userPoseRef.current },
      retries: 0,
    });
    setCollect(false);
    setUserPose([]);
  };

  return (
    <div className="flex flex-col">
      <div>
        <h1>Shoot Yo' Shot</h1>
        <p>Want to learn how to shoot a basketball?</p>
        <p>Learn the history of good shooting form and how to shoot like the best in the game.</p>
        <History />
      </div>
      <div className="flex relative">
        <div className="w-[50vw] h-[calc(3/4*50vw)]">
          <Scene />
        </div>
        <PoseCamera />
        <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
          <h1 className="flex-1 text-[7vw] z-50 text-center mario text-red-500 drop-shadow-[3px_3px_0_#000] font-bold">{text}</h1>
        </div>
      </div>
      <div className="flex justify-center">
        <button className="btn btn-primary w-fit" disabled={collect} onClick={run}>
          Start
        </button>
      </div>
    </div>
  );
};

export default App;
