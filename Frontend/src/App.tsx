import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import PoseCamera from "./Components/PoseCamera";
import Scene from "./Components/Scene";
import { useRef, useState } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";
import playerData from "./Assets/tempPlayerData.json";
import PlayerCard from "./Components/PlayerCard";
import AlternatingVideoText from "./Components/AlternatingVideoText";
import { historySections } from "./Components/HistorySections";
import TableOfContents from "./Components/TableOfContents";
import Frame from "./Components/Frame";

type PlayerData = {
  score?: number;
  image?: string;
  video?: string;
  stats: Record<string, string>;
  free_throw: number;
};

const getClosestPlayers = (userRate: number): [string, PlayerData][] =>
  Object.entries(playerData as Record<string, PlayerData>)
    .filter(([, p]) => typeof p.free_throw === "number") // safety
    .sort(
      ([, a], [, b]) =>
        Math.abs(a.free_throw - userRate) - Math.abs(b.free_throw - userRate)
    )
    .slice(0, 3);

const App = () => {
  const { collect, setCollect, userPoseRef, setUserPose } = useStore();
  const [text, setText] = useState("");
  const { http } = useHTTP();
  const [userFT, setUserFT] = useState<number | null>(null);
  const [closestPlayers, setClosestPlayers] = useState<[string, PlayerData][]>(
    []
  );

  const choosePlayers = (rate: number) => {
    const randomFT = Number(rate.toFixed(3));
    setUserFT(randomFT);
    setClosestPlayers(getClosestPlayers(randomFT));
  };

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

    // pretending that there are some calculations for user shot similarity behind the scenes
    await sleep(1000);

    // ranges from 0.5 to 0.95
    const randomRate = 0.5 + Math.random() * 0.45;
    choosePlayers(randomRate);
  };

  const historyRef = useRef<HTMLDivElement>(null!);
  const visualRef = useRef<HTMLDivElement>(null!);

  const contents = {
    history: historyRef,
    visual: visualRef,
  };

  return (
    <div className="page-bg">
      <TableOfContents contents={contents} />
      <div className="flex flex-col items-center py-[10vh] w-[70vw] mx-auto">
        <div className="flex flex-col items-center gap-4">
          <h1 className="hero-text-shadow text-6xl sporting-outline">
            Shoot Yo' Shot
          </h1>
          <p className="text-lg max-w-lg text-center">
            Learn the history of good shooting form and how to shoot like the
            best in the game.
          </p>
        </div>
        <div
          className="fade-in-up pt-10 flex flex-col items-center justify-center gap-10"
          ref={historyRef}
        >
          <Frame midClass={"w-full min-w-[325px]"}>
            <div className="glare w-1/3" />
            <img
              src={"/ShootYoShot/Images/ShootYoShot.png"}
              className="border border-slate-500 rounded-xl"
            />
          </Frame>
          <History />
          <AlternatingVideoText sections={historySections} />
        </div>
        <div className="flex relative w-full" ref={visualRef}>
          <div className="w-[50%] aspect-[4/3]">
            <Scene />
          </div>
          <PoseCamera />
          <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
            <h1 className="flex-1 text-[7vw] z-50 text-center sporting-outline">
              {text}
            </h1>
          </div>
        </div>

        <div className="flex justify-center">
          <button
            className="btn btn-primary w-fit"
            disabled={collect}
            onClick={run}
          >
            Start
          </button>
        </div>

        {userFT !== null && (
          <p className="text-center mt-6 text-lg">
            Your free-throw rate: <b>{(userFT * 100).toFixed(1)}%</b>
          </p>
        )}

        <div className="flex flex-wrap justify-around gap-4 w-full mt-10">
          {closestPlayers.map(([name, data], index) => (
            <PlayerCard key={index} name={name} {...data} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default App;
