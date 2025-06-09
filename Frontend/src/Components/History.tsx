import historySections from "../Assets/historySections.json";

const History = () => {
  return (
    <div className="flex flex-col gap-4 mt-4">
      <h2 className="text-5xl sporting-outline text-center">A Quick Background on Shooting Form History</h2>
      <div className="flex gap-8 text-xl font-light">
        <p className="flex-1">
          From the early days of the sport, players have experimented with different techniques to
          improve their shooting accuracy. In the early 20th century, players like Naismith and
          Mikan laid the groundwork for modern shooting techniques.
        </p>
        <p className="flex-1">
          As the game evolved, players like Jerry West and Larry Bird refined their shooting forms,
          emphasizing balance and mechanics. Today, players like Steph Curry continue to perfect
          their craft through advanced training methods.
        </p>
      </div>

      <div className="container mx-auto flex flex-col space-y-24 mt-16">
        {historySections.map(({ videoSrc, title, description }, index) => (
          <div
            key={index}
            className={`flex flex-col gap-10 items-center ${
              index % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
            }`}
          >
            <div className="flex-1 w-full md:w-1/2">
              <video
                src={videoSrc}
                autoPlay
                muted
                loop
                className="w-full rounded-2xl shadow-lg"
                poster="/fallbacks/video_placeholder.png"
              />
            </div>

            <div className="flex-1 w-full md:w-1/2 flex flex-col gap-4">
              <h2 className="text-4xl font-semibold">{title}</h2>
              <div className="leading-relaxed text-xl font-light flex flex-col gap-4">
                {description.map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default History;
