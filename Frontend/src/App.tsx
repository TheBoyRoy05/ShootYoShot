import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import { useRef, useState } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";
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

const App = () => {
  const { collect, setCollect, userPoseRef } = useStore();
  const [text, setText] = useState("");
  const { http } = useHTTP();
  const [userFT, setUserFT] = useState<number | null>(null);
  const [closestPlayers, setClosestPlayers] = useState<Record<string, PlayerData>>({});

  const [gender, setGender] = useState<"Male" | "Female" | "">("");
  const [height, setHeight] = useState<number | null>(null);
  const [weight, setWeight] = useState<number | null>(null);

  const [processing, setProcessing] = useState(false);
  const paramsEntered =
    gender !== "" && height !== null && weight !== null && !collect;

  const run = async () => {
    setText("Ready...");
    await sleep(1000);
    setText("Set...");
    await sleep(1000);
    setText("Shoot!");
    await sleep(1000);
    setText("");

    setCollect(true);
    await sleep(3000);
    console.log(userPoseRef.current);
    await http({
      url: "/score",
      method: "POST",
      body: { move: userPoseRef.current },
      handleData: (data) => {
        setClosestPlayers(data.scores);
        console.log(data.scores);
      },
      retries: 0,
    });
    
    setCollect(false);
    userPoseRef.current = [];
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
            Learn the history of good shooting form and how to shoot like the best in the game.
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

        <div className="flex flex-col items-center gap-4 w-full py-16">
          <h1 className="text-6xl sporting-outline">Try it out!</h1>
          <div className="flex justify-around w-full gap-4 font-semibold text-lg">
            <div className="flex flex-col gap-2">
              <p>1. Allow the website to access your camera</p>
              <p>2. Enter your gender, height, and weight</p>
              <p>3. Make sure you are fully in the frame</p>
            </div>
            <div className="flex flex-col gap-2">
              <p>4. Press the start button to Shoot Yo' Shot</p>
              <p>5. Let the program predict your player archetype</p>
              <p>6. See how you compare to NBA players</p>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center gap-4 w-full" ref={visualRef}>
          {/* Height & Weight controls */}
          <div
            className="grid  gap-y-2 gap-x-8
                grid-cols-3
                justify-center"
          >
            {/* ─────────── Row 1 – labels ─────────── */}
            <label htmlFor="gender" className="font-semibold text-center">
              Gender
            </label>

            <label htmlFor="height" className="font-semibold text-center">
              Height&nbsp;(in)
            </label>

            <label htmlFor="weight" className="font-semibold text-center">
              Weight&nbsp;(lbs)
            </label>

            {/* ─────────── Row 2 – inputs ─────────── */}
            <select
              id="gender"
              value={gender}
              onChange={(e) => setGender(e.target.value as "Male" | "Female")}
              className="select select-bordered w-full text-center"
            >
              <option value="" disabled hidden>
                Select...
              </option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>

            <input
              id="height"
              type="number"
              min={48}
              max={96}
              step={1}
              value={height ?? ""}
              onChange={(e) => setHeight(+e.target.value)}
              className="input input-bordered w-full text-center"
            />
            <input
              id="weight"
              type="number"
              value={weight ?? ""}
              onChange={(e) => setWeight(+e.target.value)}
              className="input input-bordered w-full text-center"
            />
          </div>

          <CV text={text} />
          <button
            className="btn btn-success btn-lg font-semibold text-white w-fit disabled:opacity-40 disabled:cursor-not-allowed"
            disabled={!paramsEntered}
            onClick={run}
          >
            Start
          </button>
        </div>

        {processing && (
          <span className="loading loading-spinner loading-lg mt-4"></span>
        )}

        {userFT && (
          <p className="text-center mt-6 text-3xl sporting-outline">
            Your Archetype Is Most Similar To:
          </p>
        )}

        <div className="flex flex-wrap justify-around gap-4 w-full mt-10">
          {Object.entries(closestPlayers).map(([name, data], index) => (
            <PlayerCard key={index} name={name} {...data} />
          ))}
        </div>

        <div className="h-[1px] w-full bg-gray-200/50 mt-16" />

        {/* Shot Chart Section */}
        <div className="w-full mt-16 mb-8" ref={shotChartRef}>
          <ShotChart
            defaultPlayer={
              Object.keys(closestPlayers).length > 0 ? Object.keys(closestPlayers)[0] : undefined
            }
          />
        </div>
      </div>
    </div>
  );
};

export default App;
