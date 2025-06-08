import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import {
  PLAYERS,
  SVG_DIMENSIONS,
  ZONE_ORDER,
  ZONE_LABELS,
  COURT_DIMENSIONS,
} from "../Utils/courtConstants";
import {
  transformCoordinates,
  getZoneColor,
  matchesPlayerName,
} from "../Utils/courtUtils";
import { createZonePath } from "../Utils/zoneUtils";
import { CourtRenderer } from "./CourtRenderer";

interface ShotData {
  player_name: string;
  loc_x: number;
  loc_y: number;
  shot_made_flag: number;
  game_date: string;
}

interface ZoneBreakdownData {
  PLAYER_NAME: string;
  SHOT_ZONE_BASIC: string;
  ATTEMPTS: number;
  MAKES: number;
  "SHOOTING%": number;
  "ATTEMPT%": number;
}

interface ShotChartProps {
  defaultPlayer?: string;
}

const ShotChart: React.FC<ShotChartProps> = ({ defaultPlayer }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedPlayer, setSelectedPlayer] = useState("Steph Curry");
  const [shotData, setShotData] = useState<ShotData[]>([]);
  const [loading, setLoading] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState<
    "points" | "zones"
  >("points");
  const [playerShotCache, setPlayerShotCache] = useState<
    Record<string, ShotData[]>
  >({});
  const [zoneBreakdownData, setZoneBreakdownData] = useState<
    ZoneBreakdownData[]
  >([]);
  const [selectedPlayerZones, setSelectedPlayerZones] = useState<
    Record<string, number>
  >({});

  const courtRenderer = CourtRenderer({ svgRef });

  // Update selected player when defaultPlayer prop changes
  useEffect(() => {
    if (defaultPlayer && PLAYERS.includes(defaultPlayer)) {
      setSelectedPlayer(defaultPlayer);
    }
  }, [defaultPlayer]);

  // Load zone breakdown data
  useEffect(() => {
    const loadZoneBreakdownData = async () => {
      try {
        const response = await fetch("./data/shots_breakdown.csv");
        const csvText = await response.text();
        const data = d3.csvParse(csvText, (d) => ({
          PLAYER_NAME: d.PLAYER_NAME?.trim(),
          SHOT_ZONE_BASIC: d.SHOT_ZONE_BASIC?.trim(),
          ATTEMPTS: +d.ATTEMPTS,
          MAKES: +d.MAKES,
          "SHOOTING%": +d["SHOOTING%"],
          "ATTEMPT%": +d["ATTEMPT%"],
        }));
        setZoneBreakdownData(data);
      } catch (error) {
        console.error("Error loading zone breakdown data:", error);
      }
    };
    loadZoneBreakdownData();
  }, []);

  // Update selected player zones
  useEffect(() => {
    if (zoneBreakdownData.length > 0) {
      const playerZones = zoneBreakdownData
        .filter((row) => matchesPlayerName(row.PLAYER_NAME, selectedPlayer))
        .reduce((acc, row) => {
          acc[row.SHOT_ZONE_BASIC] = row["ATTEMPT%"];
          return acc;
        }, {} as Record<string, number>);
      setSelectedPlayerZones(playerZones);
    }
  }, [selectedPlayer, zoneBreakdownData]);

  // Load shot data
  useEffect(() => {
    const loadShotData = async () => {
      if (playerShotCache[selectedPlayer]) {
        setShotData(playerShotCache[selectedPlayer]);
        return;
      }

      setLoading(true);
      try {
        const response = await fetch("./data/all_players_shots.csv");
        const csvText = await response.text();
        const data = d3.csvParse(csvText, (d) => ({
          player_name: d.PLAYER_NAME?.trim(),
          loc_x: +d.LOC_X,
          loc_y: +d.LOC_Y,
          shot_made_flag: +d.SHOT_MADE_FLAG,
          game_date: d.GAME_DATE,
        }));

        const playerShots = data.filter((shot) =>
          matchesPlayerName(shot.player_name, selectedPlayer)
        );

        const frontcourtShots = playerShots.filter((shot) => shot.loc_y >= 0);
        const madeShots = frontcourtShots.filter(
          (shot) => shot.shot_made_flag === 1
        );
        const sampledShots =
          madeShots.length > 500
            ? d3.shuffle(madeShots.slice()).slice(0, 500)
            : madeShots;

        setPlayerShotCache((prev) => ({
          ...prev,
          [selectedPlayer]: sampledShots,
        }));
        setShotData(sampledShots);
      } catch (error) {
        console.error("Error loading shot data:", error);
      } finally {
        setLoading(false);
      }
    };
    loadShotData();
  }, [selectedPlayer, playerShotCache]);

  // Render shot points
  const renderShotPoints = (
    court: d3.Selection<SVGGElement, unknown, null, undefined>
  ) => {
    shotData.forEach((shot) => {
      const coords = transformCoordinates(shot.loc_x, shot.loc_y);
      const { width, height } = COURT_DIMENSIONS;

      if (
        coords.x >= -50 &&
        coords.x <= width + 50 &&
        coords.y >= -50 &&
        coords.y <= height + 50
      ) {
        court
          .append("circle")
          .attr("cx", coords.x)
          .attr("cy", coords.y)
          .attr("r", 3)
          .attr("fill", "#00aa00")
          .attr("opacity", 0.7)
          .attr("stroke", "#000")
          .attr("stroke-width", 0.5)
          .append("title")
          .text(
            `Made shot from ${
              Math.round(
                (Math.sqrt(shot.loc_x * shot.loc_x + shot.loc_y * shot.loc_y) /
                  14.4) *
                  150
              ) / 100
            } feet from the basket`
          );
      }
    });
  };

  // Render zone overlays
  const renderZoneOverlays = (
    court: d3.Selection<SVGGElement, unknown, null, undefined>
  ) => {
    ZONE_ORDER.forEach((zoneName) => {
      const attemptPercentage = selectedPlayerZones[zoneName] || 0;
      const zonePath = createZonePath(zoneName);

      if (zonePath) {
        const zoneColor = getZoneColor(attemptPercentage);
        court
          .append("path")
          .attr("d", zonePath)
          .attr("fill", zoneColor)
          .attr("stroke", "rgba(0, 100, 0, 0.3)")
          .attr("stroke-width", 1)
          .attr("fill-rule", "nonzero")
          .append("title")
          .text(
            `${zoneName}: ${(attemptPercentage * 100).toFixed(
              1
            )}% of ${selectedPlayer}'s shots`
          );
      }
    });
  };

  // Render zone labels
  const renderZoneLabels = (
    court: d3.Selection<SVGGElement, unknown, null, undefined>
  ) => {
    ZONE_LABELS.forEach(({ zone, displayName, x, y, anchor }) => {
      const attemptPercentage = selectedPlayerZones[zone] || 0;

      // Zone name label
      court
        .append("text")
        .attr("x", x)
        .attr("y", y - 8)
        .attr("text-anchor", anchor)
        .attr("font-size", "12px")
        .attr("font-weight", "bold")
        .attr("fill", "#000")
        .attr("stroke", "#fff")
        .attr("stroke-width", 2)
        .attr("paint-order", "stroke")
        .text(displayName);

      // Percentage label
      court
        .append("text")
        .attr("x", x)
        .attr("y", y + 8)
        .attr("text-anchor", anchor)
        .attr("font-size", "12px")
        .attr("font-weight", "bold")
        .attr("fill", "#000")
        .attr("stroke", "#fff")
        .attr("stroke-width", 3)
        .attr("paint-order", "stroke")
        .text(`${(attemptPercentage * 100).toFixed(1)}%`);
    });
  };

  // Main drawing function
  const drawCourt = () => {
    const court = courtRenderer.drawCourtLines();
    if (!court || shotData.length === 0) return;

    if (visualizationMode === "points") {
      renderShotPoints(court);
    } else {
      renderZoneOverlays(court);
      renderZoneLabels(court);
    }
  };

  useEffect(() => {
    drawCourt();
  }, [shotData, visualizationMode, selectedPlayerZones]);

  const playerAnnotations: Record<string, string> = {
    "Steph Curry":
      "Observe how Steph Curry's largest share of shots comes from beyond the arc—especially above the break.",
    "Anthony Edwards":
      "Notice how Anthony Edwards either attacks the rim or launches from three, largely skipping the mid-range.",
    "Shaquille O'Neal":
      "Observe how dominant Shaquille O'Neal was in the paint, with a majority of his points coming from within the paint and restricted area.",
    "Kobe Bryant":
      "Notice how Kobe Bryant takes shots from all over the court but loved the mid-range, where he took the largest percentage of his shots.",
    "LeBron James":
      "See how well-rounded Lebron James' game is, as he takes shots from all over the court.",
    "Shai Gilgeous-Alexander":
      "Shai Gilgeous-Alexander loves to attack the basket and take shots from every pocket of the midrange, but he's not afraid to launch from 3 either.",
    "Jeremy Lin":
      "A dynamic scorer, Jeremy Lin was able to get a bucket from all over the court whenever he wanted.",
    "DeAndre Jordan":
      "An elite rebounder, DeAndre Jordan mostly gets his buckets from within the paint",
    "Giannis Antetokounmpo":
      "An extremely strong and aggresive attacker, Giannis Antetokounmpo focuses his energy on driving hard to the paint, taking a majority of his shots from this area.",
    "Rudy Gobert":
      "The big man from France is an elite defender, but offensively his game is limited to scoring within the paint."
  };

  const annotation = playerAnnotations[selectedPlayer] ?? "";

  return (
    <div className="flex flex-col items-center gap-4">
      <h3 className="text-5xl sporting-outline">Basketball Shot Chart</h3>
      <div className="text-lg font-semibold">
        Learn more about your similar players!
      </div>

      {/* Controls */}
      <div className="flex gap-4 mb-4">
        <select
          className="select select-bordered"
          value={selectedPlayer}
          onChange={(e) => setSelectedPlayer(e.target.value)}
        >
          {PLAYERS.map((player) => (
            <option key={player} value={player}>
              {player}
            </option>
          ))}
        </select>

        <select
          className="select select-bordered"
          value={visualizationMode}
          onChange={(e) =>
            setVisualizationMode(e.target.value as "points" | "zones")
          }
        >
          <option value="points">Shot Sample</option>
          <option value="zones">Shot Zones</option>
        </select>
      </div>

      {shotData.length > 0 && (
        <div className="text-lg font-semibold">
          {loading
            ? "Loading shot data..."
            : visualizationMode === "zones"
            ? `Percentage of Shots Attempted from Court Zones Across Career for ${selectedPlayer}`
            : `Sample of 500 Made Shots Across Career for ${selectedPlayer}`}
        </div>
      )}

      {shotData.length === 0 && !loading && (
        <div className="text-sm text-gray-600">
          No shot data available for {selectedPlayer}
        </div>
      )}
      <div className="relative">
        {/* Court SVG */}
        <svg
          ref={svgRef}
          width={SVG_DIMENSIONS.width}
          height={SVG_DIMENSIONS.height}
          className="border border-gray-300 rounded-lg bg-base-300 shadow-lg"
        />
        {annotation && (
          <div
            className="
              absolute top-4 right-4                /* anchor in corner  */
              max-w-xs w-60                         /* fixed-ish width  */
              bg-yellow-200/90                      /* post-it color    */
              rounded-lg p-3 shadow-xl ring-1 ring-black/10
              text-gray-900 text-sm italic leading-snug
            "
          >
            {annotation}
          </div>
        )}
      </div>
    </div>
  );
};

export default ShotChart;
