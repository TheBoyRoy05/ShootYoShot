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
import Inputs from "./Components/Inputs";
import BasketballWorldMap from "./Components/BasketballWorldMap";
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
    hand: "",
    height: null,
    weight: null,
  });

  const paramsEntered =
    inputs.gender !== "" &&
    inputs.height !== null &&
    inputs.weight !== null &&
    inputs.hand !== "" &&
    !collect;

  const run = async () => {
    setText("Ready...");
    await sleep(1000);
    setText("Set...");
    await sleep(1000);
    setText("Shoot!");
    await sleep(1000);
    setText("");

    console.log("Starting pose collection for hand:", inputs.hand);
    setCollect(inputs.hand);
    await sleep(3000);
    
    console.log("Pose data collected:", userPoseRef.current);
    console.log("Number of pose frames:", Object.keys(userPoseRef.current).length);
    
    setText("Analyzing...");

    console.log("Sending score request with data:", { move: userPoseRef.current });
    
    await http({
      url: "/score",
      method: "POST",
      body: { move: userPoseRef.current },
      handleData: (data) => {
        console.log("Score response received:", data);
        setClosestPlayers(data.scores);
      },
      retries: 0,
    });

    console.log("Sending position prediction request with:", inputs);
    
    http({
      url: "/predict_position",
      method: "POST",
      body: inputs,
      handleData: (data: { position: string }) => {
        console.log("Position prediction received:", data);
        setPredictedPosition(data.position);
      },
    });

    setText("");
    setCollect("");
    userPoseRef.current = [];
  };

  const titleRef = useRef<HTMLDivElement>(null!);
  const historyRef = useRef<HTMLDivElement>(null!);
  const globalPopularityRef = useRef<HTMLDivElement>(null!);
  const whatWeDoRef = useRef<HTMLDivElement>(null!);
  const instructionsRef = useRef<HTMLDivElement>(null!);
  const visualRef = useRef<HTMLDivElement>(null!);
  const shotChartRef = useRef<HTMLDivElement>(null!);
  const takeawayRef = useRef<HTMLDivElement>(null!);

  const contents = {
    title: titleRef,
    "Global Popularity": globalPopularityRef,
    "What We Do": whatWeDoRef,
    history: historyRef,
    instructions: instructionsRef,
    "Shot Visual": visualRef,
    "Shot Chart": shotChartRef,
    takeaway: takeawayRef,
  };

  return (
    <div className="page-bg">
      <TableOfContents contents={contents} />
      <div className="flex flex-col items-center w-[70vw] mx-auto">
        <div className="fade-in-up flex flex-col items-center gap-4 pt-[10vh]" ref={titleRef}>
          <h1 className="hero-text-shadow text-6xl sporting-outline">Shoot Yo' Shot</h1>
          <p className="text-2xl max-w-lg text-center">
            Analyze Your Form, Match with NBA Players, and Level Up Your Court IQ
          </p>
          <div className="flex flex-wrap justify-center gap-6 mt-4 mb-6">
            <div className="text-lg font-semibold text-white bg-slate-700/80 px-4 py-2 rounded-lg">
              Issac Roy
            </div>
            <div className="text-lg font-semibold text-white bg-slate-700/80 px-4 py-2 rounded-lg">
              Noah Golder
            </div>
            <div className="text-lg font-semibold text-white bg-slate-700/80 px-4 py-2 rounded-lg">
              Ty Albao
            </div>
            <div className="text-lg font-semibold text-white bg-slate-700/80 px-4 py-2 rounded-lg">
              Rushyendra Katabathuni
            </div>
          </div>
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
            <p className="text-2xl mb-12">
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
        <div className="fade-in-up pt-16 flex flex-col items-center gap-10 w-full" ref={whatWeDoRef}>
          <div className="text-center max-w-4xl">
            <h1 className="text-5xl sporting-outline mb-8">The Cost of Training</h1>
            <p className="text-2xl font-medium mb-4 text-white">
              BUT, basketball training can cost from $50 to $150 per hour. 
              Furthermore, most players don't know how to optimize their own abilities — their form, their strengths, their role.
            </p>
            <p className="text-lg text-white mb-8">
              (Source: europrobasket.com)
            </p>
          </div>
          
          <div className="text-center max-w-4xl">
            <h2 className="text-5xl sporting-outline mb-6">What We Do</h2>
            <p className="text-2xl font-small text-white">
              THEREFORE, we personalize basketball training by comparing your shooting form and body measurements to NBA players 
              in order to help you understand your player archetype and improve your game, all on a free website.
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
          <div className="flex justify-around w-full gap-4 font-semibold text-xl">
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
          <CV text={text} />

          <button
            className="btn btn-success btn-lg font-semibold text-white w-fit disabled:opacity-40 disabled:cursor-not-allowed"
            disabled={!paramsEntered}
            onClick={run}
          >
            Start
          </button>
        </div>

        {predictedPosition && (
          <div className="mt-6 flex flex-col items-center gap-2">
            <p className="text-center text-3xl sporting-outline">
              Your Best Position: {predictedPosition}
            </p>
            <p className="text-center">
              (Relative to the average {inputs.gender.toLowerCase()} human)
            </p>
          </div>
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

        <div className="h-[1px] w-full bg-gray-200/50 mt-16" />

        <div
          className="fade-in-up w-full py-20 flex flex-col items-center"
          ref={takeawayRef}
        >
          <h2 className="text-5xl sporting-outline mb-4">Takeaways</h2>

          <div className="flex items-center gap-8">
            <p className="flex-1 font-light leading-relaxed">
              Great shooters share three things: balanced <strong>base</strong>,
              consistent <strong>release point</strong>, and a smooth
              <strong> follow-through</strong>. Our model compares your motion
              frame-by-frame against the best of the best, because who better is
              there to learn from? Use the model to tweak your stance, record
              another attempt, and watch your similarity score climb. By
              suggesting your best position and displaying the NBA players that
              you are most similar to, our project aims to help you learn how to
              optimize your basketball skills and be the best basketball player
              you can be.
            </p>
            <p className="flex-1 font-light leading-relaxed">
              Once you've found your most similar players and your recommended
              position, you can use the shot chart to identify what types of
              shots and what areas of the court those players work best in. This
              can help you understand what skills and shots to focus on in order
              to maximize your potential. Whether you are a hardcore basketball
              fan or you've never been interested before, with the tools present
              in this project, anyone can quickly learn how to improve their
              skills.
            </p>
          </div>
          <p className="flex-1 font-light leading-relaxed italic">
            "Don't practice until you get it right; practice until you can't get
            it wrong."
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;
