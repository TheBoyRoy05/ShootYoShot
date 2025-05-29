// src/Components/AlternatingVideoText.tsx
import React from "react";

export interface Section {
  videoSrc: string;
  title: string;
  description: string;
  isVideoLeft: boolean;
}

export interface AlternatingVideoTextProps {
  sections: Section[];
}

const AlternatingVideoText: React.FC<AlternatingVideoTextProps> = ({
  sections,
}) => (
  <div className="container mx-auto flex flex-col space-y-24 py-12">
    {sections.map(({ videoSrc, title, description, isVideoLeft }, idx) => (
      <div
        key={idx}
        className="grid gap-10 items-center md:grid-cols-2"
      >
        {/* Video */}
        <div className={isVideoLeft ? "" : "md:order-2"}>
          <video
            src={videoSrc}
            controls
            className="w-full h-auto rounded-2xl shadow-lg"
            // fallback poster for fast page load
            poster="/fallbacks/video_placeholder.png"
          />
        </div>

        {/* Text */}
        <div className={isVideoLeft ? "" : "md:order-1"}>
          <h2 className="text-2xl md:text-3xl font-bold mb-3">{title}</h2>
          <p className="text-base leading-relaxed whitespace-pre-line">
            {description}
          </p>
        </div>
      </div>
    ))}
  </div>
);

export default AlternatingVideoText;