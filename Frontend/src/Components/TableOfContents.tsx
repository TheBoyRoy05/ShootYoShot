import { useEffect, useState } from "react";
import { capitalize } from "../Utils/functions";

interface TableOfContentsProps {
  contents: { [key: string]: React.RefObject<HTMLDivElement> };
}

const TableOfContents = ({ contents }: TableOfContentsProps) => {
  const [activeSection, setActiveSection] = useState<string>("");

  useEffect(() => {
    const handleScroll = () => {
      let closestSection = "";
      let minDistance = Infinity;

      Object.entries(contents).forEach(([name, ref]) => {
        if (ref.current) {
          const rect = ref.current.getBoundingClientRect();
          const distance = Math.abs(rect.top);
          
          if (distance < minDistance) {
            minDistance = distance;
            closestSection = name;
          }
        }
      });

      setActiveSection(closestSection);
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll(); // Initial check

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [contents]);

  const handleClick = (ref: React.RefObject<HTMLDivElement>) => {
    ref.current!.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="hidden lg:flex lg:flex-col fixed left-[88vw] top-[150px] gap-4 z-[100] fade-in">
      <h6 className="text-gray-300 text-sm">CONTENTS</h6>
      {Object.entries(contents).map(([name, ref], index) => (
        <div
          key={index}
          onClick={() => handleClick(ref)}
          className={`hover:cursor-pointer hover:text-gray-200 font-light text-sm transition-all duration-200 ${
            activeSection === name ? "text-white font-medium" : "text-gray-500"
          }`}
        >
          {capitalize(name)}
        </div>
      ))}
    </div>
  );
};

export default TableOfContents;
