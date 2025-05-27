import History from "./Components/History";
import PoseCamera from "./Components/PoseCamera";
import Scene from "./Components/Scene";

const App = () => {
  return (
    <div className="flex flex-col">
      <div>
        <h1>Shoot Yo' Shot</h1>
        <p>Want to learn how to shoot a basketball?</p>
        <p>Learn the history of good shooting form and how to shoot like the best in the game.</p>
        <History />
      </div>
      <div className="flex">
        <div className="w-[50vw] h-[calc(3/4*50vw)]">
          <Scene />
        </div>
        <PoseCamera />
      </div>
    </div>
  );
};

export default App;
