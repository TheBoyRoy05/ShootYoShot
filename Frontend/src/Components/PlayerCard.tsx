import { useState } from "react";
import { twMerge } from "tailwind-merge";

interface PlayerCardProps {
  name: string;
  score: number;
  image?: string;
  video?: string;
  stats: Record<string, string>;
}

const PlayerCard = ({ name, score, stats, image, video }: PlayerCardProps) => {
  const [isLockedFlipped, setIsLockedFlipped] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const isFlipped = isLockedFlipped || isHovered;
  const cardClass =
    "absolute inset-0 flex flex-col items-center border gap-4 hover:border-(--color-primary) rounded-2xl overflow-hidden bg-base-200 [backface-visibility:hidden]";

  return (
    <div
      className="h-[450px] [perspective:1000px] cursor-pointer w-[300px]"
      onClick={() => setIsLockedFlipped((prev) => !prev)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`relative w-full h-full transition-transform duration-700 [transform-style:preserve-3d] ${
          isFlipped ? "[transform:rotateY(180deg)]" : ""
        }`}
      >
        {/* Front Side */}
        <div className={twMerge(cardClass, "[transform:rotateY(0deg)] p-4")}>
          <div className="skeleton relative group aspect-video w-full">
            <img
              src={image || "/"}
              className="size-full object-cover rounded-2xl"
              onError={(e) => (e.currentTarget.style.display = "none")}
              onLoad={(e) => (e.currentTarget.style.display = "block")}
            />
          </div>
          <div className="relative z-10 text-center px-8 h-[95%] flex flex-col gap-4 justify-start w-full">
            <h3 className="text-2xl font-bold">
              {name} - {score}% Similarity
            </h3>
            <ul className="flex flex-col gap-2">
              {Object.entries(stats).map(([key, value]) => (
                <li key={key}>
                  <span className="font-bold">{key}: </span>
                  {value}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Back Side */}
        <div className={twMerge(cardClass, "[transform:rotateY(180deg)]")}>
          <video 
            src={video} 
            className="w-full h-full object-cover" 
            autoPlay
            muted
            loop
          />
        </div>
      </div>
    </div>
  );
};

export default PlayerCard;
