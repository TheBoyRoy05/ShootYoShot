import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import { useRef, useState } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";
import playerData from "./Assets/tempPlayerData.json";
import PlayerCard from "./Components/PlayerCard";
import TableOfContents from "./Components/TableOfContents";
import Frame from "./Components/Frame";
import CV from "./Components/CV/CV";
import ShotChart from "./Components/ShotChart";

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
    .sort(([, a], [, b]) => Math.abs(a.free_throw - userRate) - Math.abs(b.free_throw - userRate))
    .slice(0, 3);

const App = () => {
  const { collect, setCollect, userPoseRef } = useStore();
  const [text, setText] = useState("");
  const { http } = useHTTP();
  const [userFT, setUserFT] = useState<number | null>(null);
  const [closestPlayers, setClosestPlayers] = useState<[string, PlayerData][]>([]);

  const choosePlayers = (rate: number) => {
    const randomFT = Number(rate.toFixed(3));
    setUserFT(randomFT);
    const selectedPlayers = getClosestPlayers(randomFT);
    selectedPlayers.sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
    setClosestPlayers(selectedPlayers);
  };

  const run = async () => {
    setText("Ready...");
    await sleep(1000);
    setText("Set...");
    await sleep(1000);
    setText("Shoot!");
    await sleep(1000);
    setText("");

    setCollect(true);
    await sleep(1000);
    await http({
      url: "/score",
      method: "POST",
      body: { move: userPoseRef.current },
      retries: 0,
    });
    setCollect(false);
    console.log(userPoseRef.current);
    userPoseRef.current = [];

    // pretending that there are some calculations for user shot similarity behind the scenes
    await sleep(1000);

    // ranges from 0.5 to 0.95
    const randomRate = 0.5 + Math.random() * 0.45;
    choosePlayers(randomRate);
  };

  const titleRef = useRef<HTMLDivElement>(null!);
  const historyRef = useRef<HTMLDivElement>(null!);
  const instructionsRef = useRef<HTMLDivElement>(null!);
  const visualRef = useRef<HTMLDivElement>(null!);
  const shotChartRef = useRef<HTMLDivElement>(null!);

  const contents = {
    title: titleRef,
    history: historyRef,
    instructions: instructionsRef,
    visual: visualRef,
    "Shot Chart": shotChartRef,
  };

  return (
    <div className="page-bg">
      <TableOfContents contents={contents} />
      <div className="flex flex-col items-center w-[70vw] mx-auto">
        <div className="fade-in-up flex flex-col items-center gap-4 pt-[10vh]" ref={titleRef}>
          <h1 className="hero-text-shadow text-6xl sporting-outline">Shoot Yo' Shot</h1>
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
        </div>

        <div className="w-full pt-16" ref={instructionsRef}>
          <div className="h-[1px] w-full bg-gray-200/50" />
        </div>

        <div className="flex flex-col items-center gap-4 w-full py-16" >
          <h1 className="text-6xl sporting-outline">Try it out!</h1>
          <div className="flex justify-around w-full gap-4 font-semibold text-lg">
            <div className="flex flex-col gap-2">
              <p>1. Allow the website to access your camera</p>
              <p>2. Make sure you are fully in the frame</p>
            </div>
            <div className="flex flex-col gap-2">
              <p>3. Press the start button to Shoot Yo' Shot</p>
              <p>4. See how you compare to the NBA players</p>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center gap-4 w-full" ref={visualRef}>
          <CV text={text} />
          <button className="btn btn-success btn-lg font-semibold text-white w-fit" disabled={collect} onClick={run}>
            Start
          </button>
        </div>

        {userFT && (
          <p className="text-center mt-6 text-3xl sporting-outline">
            Your Shot Is Most Similar To:
          </p>
        )}

        <div className="flex flex-wrap justify-around gap-4 w-full mt-10">
          {closestPlayers.map(([name, data], index) => (
            <PlayerCard key={index} name={name} {...data} />
          ))}
        </div>

        <div className="h-[1px] w-full bg-gray-200/50 mt-16" />

        {/* Shot Chart Section */}
        <div className="w-full mt-16 mb-8" ref={shotChartRef}>
          <ShotChart defaultPlayer={closestPlayers.length > 0 ? closestPlayers[0][0] : undefined} />
        </div>
      </div>
    </div>
  );
};

export default App;
