// src/Content/historySections.ts
import type { Section } from "./AlternatingVideoText";
import video1 from "../Videos/60sShots.mp4";
import video2 from "../Videos/90sShots.mp4"
import video3 from "../Videos/2010sShots.mp4"

export const historySections: Section[] = [
  {
    videoSrc: video1,
    title: "1960s: The First Jump Shots",
    description: `
    In the 1960s, the league was still young, and players were just starting to experiment with the jump shot. This was a time of innovation, where athletes began to realize the potential of elevating their shots for better accuracy and style. The jump shot would soon become a staple in basketball, changing the game forever.
    The 1960s saw the rise of players like Wilt Chamberlain and Bill Russell, who were known for their dominant presence in the paint. However, it was also a period where the fundamentals of shooting were being redefined. Players started to incorporate the jump shot into their arsenal, leading to a more dynamic and exciting style of play.
    The jump shot allowed players to shoot over defenders, making it a crucial skill for scoring. This era laid the groundwork for future generations of shooters, as players began to understand the mechanics and benefits of this technique.
    `.trim(),
    isVideoLeft: true,
  },
  {
    videoSrc: video2,
    title: "1990s – Developing the Jump Shot",
    description: `
    The 1990s marked a turning point in basketball, as the jump shot became a fundamental skill for players. This decade saw the emergence of iconic shooters like Ray Allen and Reggie Miller, who mastered the art of the jump shot. Their ability to shoot from long range and create their own shots revolutionized the game.
    The jump shot was no longer just a tool for scoring; it became a symbol of skill and finesse. Players learned to fully use their bodies, creating a fluid motion that made their shots more accurate and difficult to defend. Although the 3-point line was added to the NBA in the 80s, it wasn't until the 90s that the three-point line became a cornerstone of the game and further emphasized the importance of shooting from distance.
    The evolution of the jump shot during this decade set the stage for the modern game, where shooting is a key component of success. Players began to focus on their shooting mechanics, leading to a new era of precision and style in basketball.
    `.trim(),
    isVideoLeft: false,
  },
  {
    videoSrc: video3,
    title: "2020s – The Modern Jump Shot",
    description: `
    The 2020s saw the jump shot evolve into a powerful weapon, with players like Stephen Curry and Klay Thompson redefining what it meant to be a shooter. This era was characterized by the rise of the 3-point shot, as teams began to prioritize spacing and shooting over traditional post play.
    The jump shot became a key part of offensive strategies, with players developing quick releases and deep shooting range. The ability to shoot from beyond the arc changed the dynamics of the game, forcing defenses to adapt and creating new opportunities for scoring.
    This decade also saw advancements in training and analytics, allowing players to refine their shooting techniques and improve their efficiency. The jump shot became a symbol of modern basketball, representing skill, precision, and the evolution of the game.
    `.trim(),
    isVideoLeft: true,
  },

  /* keep adding rows… */
];
