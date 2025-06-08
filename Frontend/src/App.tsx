import { useStore } from "./Hooks/useStore";
import History from "./Components/History";
import { useRef, useState, useEffect } from "react";
import { sleep } from "./Utils/functions";
import useHTTP from "./Hooks/useHTTP";
import PlayerCard from "./Components/PlayerCard";
import TableOfContents from "./Components/TableOfContents";
import Frame from "./Components/Frame";
import CV from "./Components/CV/CV";
import ShotChart from "./Components/ShotChart";
import Inputs from "./Components/Inputs";
import BasketballWorldMap from "./Components/BasketballWorldMap";
import PlayerStats from "../../Backend/Data/stats.json";
import type { FormInputs } from "./Utils/types";

type PlayerData = {
  score?: number;
  image?: string;
  video?: string;
  stats: Record<string, string>;
  free_throw: number;
};

const App = () => {
  const { http } = useHTTP();
  const [text, setText] = useState("");
  const { collect, setCollect, userPoseRef } = useStore();
  const [closestPlayers, setClosestPlayers] = useState<Record<string, PlayerData>>({});
  const [predictedPosition, setPredictedPosition] = useState<string>("");

  const [inputs, setInputs] = useState<FormInputs>({
    gender: "",
    height: null,
    weight: null,
  });

  const paramsEntered =
    inputs.gender !== "" && inputs.height !== null && inputs.weight !== null && !collect;

  const run = async () => {
    // First get position prediction
    await http({
      url: "/predict_position",
      method: "POST",
      body: inputs,
      handleData: (data) => {
        if (data.position) {
          setPredictedPosition(data.position);
        }
      },
    });

    // Then start the shooting sequence
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
  const globalPopularityRef = useRef<HTMLDivElement>(null!);
  const instructionsRef = useRef<HTMLDivElement>(null!);
  const visualRef = useRef<HTMLDivElement>(null!);
  const shotChartRef = useRef<HTMLDivElement>(null!);

  const contents = {
    title: titleRef,
    history: historyRef,
    "Global Popularity": globalPopularityRef,
    instructions: instructionsRef,
    "Shot Visual": visualRef,
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
          <Frame midClass={"w-full min-w-[325px]"}>
            <div className="glare w-1/3" />
            <img
              src={"/ShootYoShot/Images/ShootYoShot.png"}
              className="border border-slate-500 rounded-xl"
            />
          </Frame>
        </div>

        <div className="w-full pt-16">
          <div className="h-[1px] w-full bg-gray-200/50" />
        </div>

        {/* Global Basketball Popularity Section */}
        <div className="fade-in-up pt-16 flex flex-col items-center gap-10 w-full" ref={globalPopularityRef}>
          <div className="text-center max-w-4xl">
            <h1 className="text-5xl sporting-outline mb-8">Basketball Worldwide</h1>
            <p className="text-2xl font-semibold mb-8">
              Across the world, over 610 million people play basketball.
              <span className="text-lg font-normal block mt-2 text-white">
                (Source: FIBA Basketball)
              </span>
            </p>
            <p className="text-lg mb-12">
              From the streets of Manila to the courts of Lithuania, basketball has become a global phenomenon. 
              Explore how the sport's popularity varies across different countries and regions.
            </p>
          </div>
          <BasketballWorldMap />
        </div>

        <div className="w-full pt-16">
          <div className="h-[1px] w-full bg-gray-200/50" />
        </div>

        {/* The Cost of Training Section */}
        <div className="fade-in-up pt-16 flex flex-col items-center gap-10 w-full">
          <div className="text-center max-w-4xl">
            <h1 className="text-5xl sporting-outline mb-8">The Cost of Training</h1>
            <p className="text-2xl font-semibold mb-4 text-white">
              BUT, basketball training can cost from $50 to $150 per hour. 
              And training sessions don't incorporate NBA player data to help users understand their player archetypes.
            </p>
            <p className="text-lg text-white mb-8">
              (Source: europrobasket.com)
            </p>
          </div>
          
          <div className="text-center max-w-4xl">
            <h2 className="text-5xl sporting-outline mb-6">What We Do</h2>
            <p className="text-2xl font-bold text-white">
              THEREFORE, we personalize basketball training by comparing your shooting form and body measurements to NBA players 
              in order to help you understand your player archetype and improve your game.
            </p>
          </div>
        </div>

        <div className="w-full pt-16">
          <div className="h-[1px] w-full bg-gray-200/50" />
        </div>

        {/* History Section */}
        <div
          className="fade-in-up pt-16 flex flex-col items-center justify-center gap-10"
          ref={historyRef}
        >
          <History />
        </div>

        <div className="w-full pt-16">
          <div className="h-[1px] w-full bg-gray-200/50" />
        </div>

        <div className="flex flex-col items-center gap-4 w-full py-16" ref={instructionsRef}>
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
          <Inputs inputs={inputs} setInputs={setInputs} />
          {predictedPosition && (
            <div className="text-xl font-semibold text-center">
              Predicted Position: {predictedPosition}
            </div>
          )}
          <CV text={text} />

          <button
            className="btn btn-success btn-lg font-semibold text-white w-fit disabled:opacity-40 disabled:cursor-not-allowed"
            disabled={!paramsEntered}
            onClick={run}
          >
            Start
          </button>
        </div>

        {Object.keys(closestPlayers).length > 0 && (
          <p className="text-center mt-6 text-3xl sporting-outline">
            Your Archetype Is Most Similar To:
          </p>
        )}

        <div className="flex flex-wrap justify-around gap-4 w-full mt-10">
          {Object.entries(closestPlayers)
            .slice(0, 3)
            .map(([name, data], index) => (
              <PlayerCard key={index} name={name} {...data} />
            ))}
        </div>

        <div className="h-[1px] w-full bg-gray-200/50 mt-16" />

        {/* Shot Chart Section */}
        <div className="w-full mt-16 mb-8" ref={shotChartRef}>
          <ShotChart />
        </div>
      </div>
    </div>
  );
};

export default App;
