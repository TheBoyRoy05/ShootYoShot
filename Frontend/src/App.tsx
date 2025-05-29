import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import PoseCamera from "./Components/PoseCamera";
import Scene from "./Components/Scene";
import { useState } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";
import AlternatingVideoText from "./Components/AlternatingVideoText";
import { historySections } from "./Components/HistorySections";

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
    <div className="flex flex-col p-[5vw]">
      <div className="flex flex-col items-center justify-center gap-4">
        <h1 className="hero-text-shadow text-4xl font-bold">Shoot Yo' Shot</h1>
        <p className="text-xl max-w-xl text-center">Learn the history of good shooting form and how to shoot like the best in the game.</p>
        <History />
        <AlternatingVideoText sections={historySections} />
      </div>
      <div className="flex relative">
        <div className="w-[45vw] h-[calc(3/4*45vw)]">
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
