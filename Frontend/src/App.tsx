import PoseCamera from "./Components/PoseCamera";
import Scene from "./Components/Scene";

const App = () => {
  return (
    <div className="flex">
      <div className="w-[50vw] h-[calc(3/4*50vw)]">
        <Scene />
      </div>
      <PoseCamera />
    </div>
  );
};

export default App;
