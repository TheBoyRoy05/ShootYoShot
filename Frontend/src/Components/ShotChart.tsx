import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

// Store court dimensions for zone calculations (NBA dimensions scaled to 14.4px per foot)
export const COURT_DIMENSIONS = {
  width: 720, // 50 feet -> 720px (was 600px)
  height: 600, // 47 feet -> 677px (was 564px)
  paintWidth: 230, // 16 feet -> 230px (was 192px)
  paintHeight: 274, // 19 feet -> 274px (was 228px)
  freeThrowLineDistance: 216, // 15 feet -> 216px (was 180px)
  threePointRadius: 342, // 23'9" -> 342px (was 285px)
  threePointStraightDistance: 43, // 3 feet -> 43px (was 36px)
  restrictedAreaRadius: 58, // 4 feet -> 58px (was 48px)
  basketRadius: 11, // 0.75 feet -> 11px (was 9px)
  backboardWidth: 86, // 6 feet -> 86px (was 72px)
  basketX: 360, // Center of court width (was 300)
  basketY: 601, // At baseline (was 564)
};

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

const PLAYERS = [
  'Steph Curry',
  'Shaquille O\'Neal',
  'Shai Gilgeous-Alexander',
  'Jeremy Lin',
  'DeAndre Jordan',
  'Giannis Antetokounmpo',
  'LeBron James',
  'Rudy Gobert',
  'Kobe Bryant'
];

const ShotChart: React.FC<ShotChartProps> = ({ defaultPlayer }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedPlayer, setSelectedPlayer] = useState('Steph Curry');
  const [shotData, setShotData] = useState<ShotData[]>([]);
  const [loading, setLoading] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState<"points" | "zones">("points");
  const [playerShotCache, setPlayerShotCache] = useState<Record<string, ShotData[]>>({});
  const [zoneBreakdownData, setZoneBreakdownData] = useState<ZoneBreakdownData[]>([]);
  const [selectedPlayerZones, setSelectedPlayerZones] = useState<Record<string, number>>({});

  // Update selected player when defaultPlayer prop changes
  useEffect(() => {
    if (defaultPlayer && PLAYERS.includes(defaultPlayer)) {
      setSelectedPlayer(defaultPlayer);
    }
  }, [defaultPlayer]);

  // Transform NBA coordinates to our court visualization using known data ranges
  const transformCoordinates = (x: number, y: number) => {
    // Known coordinate ranges from the dataset:
    // loc_x: -250 to 250 (500 units wide)
    // loc_y: -50 to 490 (540 units tall)
    const { width, height } = COURT_DIMENSIONS;

    // Map loc_x (-250 to 250) to screen x (0 to width)
    const normalizedX = (x - -250) / (250 - -250);

    // Map loc_y (-50 to 490) to screen y (height to 0, flipped)
    const normalizedY = (y - -50) / (490 - -50);

    return {
      x: normalizedX * width,
      y: height - normalizedY * height * 677/600, // Flip Y axis so y=490 is at top
    };
  };

  // Load zone breakdown data once on component mount
  useEffect(() => {
    const loadZoneBreakdownData = async () => {
      try {
        console.log("Loading zone breakdown data...");
        const response = await fetch("./data/shots_breakdown.csv");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const csvText = await response.text();
        const data = d3.csvParse(csvText, (d) => ({
          PLAYER_NAME: d.PLAYER_NAME?.trim(),
          SHOT_ZONE_BASIC: d.SHOT_ZONE_BASIC?.trim(),
          ATTEMPTS: +d.ATTEMPTS,
          MAKES: +d.MAKES,
          "SHOOTING%": +d["SHOOTING%"],
          "ATTEMPT%": +d["ATTEMPT%"],
        }));

        console.log("Zone breakdown data loaded:", data.length, "rows");
        console.log("Sample data:", data[0]);
        setZoneBreakdownData(data);
      } catch (error) {
        console.error("Error loading zone breakdown data:", error);
      }
    };

    loadZoneBreakdownData();
  }, []);

  // Update selected player zones when player or breakdown data changes
  useEffect(() => {
    if (zoneBreakdownData.length > 0) {
      const playerZones = zoneBreakdownData
        .filter((row) => {
          const name = row.PLAYER_NAME?.toLowerCase() || "";
          const selectedLower = selectedPlayer.toLowerCase();

          // Use same name matching logic as shot data
          if (selectedPlayer === 'Steph Curry') {
            return name.includes('curry') || name.includes('stephen');
          } else if (selectedPlayer === 'Shaquille O\'Neal') {
            return name.includes('shaquille') || name.includes('o\'neal');
          } else if (selectedPlayer === 'Shai Gilgeous-Alexander') {
            return name.includes('shai') || name.includes('gilgeous');
          } else if (selectedPlayer === 'Jeremy Lin') {
            return name.includes('jeremy') && name.includes('lin');
          } else if (selectedPlayer === 'DeAndre Jordan') {
            return name.includes('deandre') || name.includes('jordan');
          } else if (selectedPlayer === 'Giannis Antetokounmpo') {
            return name.includes('giannis') || name.includes('antetokounmpo');
          } else if (selectedPlayer === 'LeBron James') {
            return name.includes('lebron') || name.includes('james');
          } else if (selectedPlayer === 'Rudy Gobert') {
            return name.includes('rudy') && name.includes('gobert');
          } else if (selectedPlayer === 'Kobe Bryant') {
            return name.includes('kobe') || name.includes('bryant');
          }

          const selectedParts = selectedLower.split(" ");
          return selectedParts.every((part) => name.includes(part));
        })
        .reduce((acc, row) => {
          acc[row.SHOT_ZONE_BASIC] = row["ATTEMPT%"];
          return acc;
        }, {} as Record<string, number>);

      console.log("Player zones for", selectedPlayer, ":", playerZones);
      setSelectedPlayerZones(playerZones);
    }
  }, [selectedPlayer, zoneBreakdownData]);

  // Load shot data
  useEffect(() => {
    let firstRowLogged = false;

    const loadShotData = async () => {
      setLoading(true);
      try {
        // Check if we already have cached shots for this player
        if (playerShotCache[selectedPlayer]) {
          console.log("Using cached shots for:", selectedPlayer);
          setShotData(playerShotCache[selectedPlayer]);
          setLoading(false);
          return;
        }

        console.log("Loading shot data for:", selectedPlayer);

        const response = await fetch("./data/all_players_shots.csv");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const csvText = await response.text();
        console.log("CSV loaded, length:", csvText.length);
        console.log("CSV headers (first line):", csvText.split("\n")[0]);

        const data = d3.csvParse(csvText, (d) => {
          // Log first row to see column names
          if (!firstRowLogged) {
            console.log("First CSV row keys:", Object.keys(d));
            console.log("First CSV row values:", d);
            firstRowLogged = true;
          }

          return {
            player_name: d.PLAYER_NAME?.trim(),
            loc_x: +d.LOC_X,
            loc_y: +d.LOC_Y,
            shot_made_flag: +d.SHOT_MADE_FLAG,
            game_date: d.GAME_DATE,
          };
        });

        console.log("Total parsed rows:", data.length);
        console.log(
          "Sample of all player names:",
          [...new Set(data.map((d) => d.player_name))].slice(0, 10)
        );

        // Filter for selected player
        const playerShots = data.filter((shot) => {
          const name = shot.player_name?.toLowerCase() || "";
          const selectedLower = selectedPlayer.toLowerCase();

          // Handle different name matching strategies
          if (selectedPlayer === 'Steph Curry') {
            return name.includes('curry') || name.includes('stephen') || name === 'stephen curry';
          } else if (selectedPlayer === 'Shaquille O\'Neal') {
            return name.includes('shaquille') || name.includes('o\'neal') || name.includes('oneal');
          } else if (selectedPlayer === 'Shai Gilgeous-Alexander') {
            return name.includes('shai') || name.includes('gilgeous') || name.includes('alexander');
          } else if (selectedPlayer === 'Jeremy Lin') {
            return name.includes('jeremy') && name.includes('lin');
          } else if (selectedPlayer === 'DeAndre Jordan') {
            return name.includes('deandre') || (name.includes('de') && name.includes('andre')) || name.includes('jordan');
          } else if (selectedPlayer === 'Giannis Antetokounmpo') {
            return name.includes('giannis') || name.includes('antetokounmpo');
          } else if (selectedPlayer === 'LeBron James') {
            return name.includes('lebron') || name.includes('james');
          } else if (selectedPlayer === 'Rudy Gobert') {
            return name.includes('rudy') && name.includes('gobert');
          } else if (selectedPlayer === 'Kobe Bryant') {
            return name.includes('kobe') || name.includes('bryant');
          }

          // Fallback: check if selected player name parts are in the shot player name
          const selectedParts = selectedLower.split(" ");
          return selectedParts.every((part) => name.includes(part));
        });

        // Filter for frontcourt shots only
        const frontcourtShots = playerShots.filter((shot) => shot.loc_y >= 0);

        console.log("Found shots for", selectedPlayer, "(frontcourt only):", {
          totalShots: frontcourtShots.length,
          sampleShot: frontcourtShots[0],
          uniqueNames: [...new Set(playerShots.map((s) => s.player_name))],
          backcourtShotsFiltered: playerShots.filter((shot) => shot.loc_y < 0).length,
        });

        // Filter for only made shots
        const madeShots = frontcourtShots.filter((shot) => shot.shot_made_flag === 1);

        console.log("Found made shots for", selectedPlayer, ":", {
          totalMadeShots: madeShots.length,
          sampleMadeShot: madeShots[0],
          missedShots: frontcourtShots.filter((shot) => shot.shot_made_flag === 0).length,
        });

        // Sample 500 random made shots, or all if less than 500
        const sampledShots =
          madeShots.length > 500 ? d3.shuffle(madeShots.slice()).slice(0, 500) : madeShots;

        console.log("Final sampled shots:", {
          count: sampledShots.length,
          requestedCount: Math.min(500, madeShots.length),
          firstShot: sampledShots[0],
          coordinateRange:
            sampledShots.length > 0
              ? {
                  xMin: Math.min(...sampledShots.map((s) => s.loc_x)),
                  xMax: Math.max(...sampledShots.map((s) => s.loc_x)),
                  yMin: Math.min(...sampledShots.map((s) => s.loc_y)),
                  yMax: Math.max(...sampledShots.map((s) => s.loc_y)),
                }
              : "No shots found",
        });

        // Cache the sampled shots for this player
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
  }, [selectedPlayer]);

  // Helper function to get zone color based on attempt percentage
  const getZoneColor = (attemptPercentage: number): string => {
    if (attemptPercentage === 0) return "rgba(0, 150, 0, 0.1)"; // Very light green for 0%

    // Use a more sensitive opacity calculation for better granularity
    // Square the normalized percentage to make differences more pronounced
    const maxAttemptPercentage = 0.4; // 40% is very high for any single zone
    const normalizedPercentage =
      Math.min(attemptPercentage, maxAttemptPercentage) / maxAttemptPercentage;
    const squaredPercentage = normalizedPercentage * normalizedPercentage; // Square for more dramatic differences
    const opacity = 0.15 + squaredPercentage * 0.65; // Range from 0.15 to 0.8

    return `rgba(0, 150, 0, ${opacity})`;
  };

  // Helper functions to create zone paths
  const createZonePath = (zoneName: string): string => {
    const { width, height } = COURT_DIMENSIONS;
    const wingLineY = height - (2 / 5) * height;

    // Calculate 3-point line intersection points
    const leftWingEndX =
      width / 2 -
      Math.sqrt(
        COURT_DIMENSIONS.threePointRadius * COURT_DIMENSIONS.threePointRadius -
          (height - 24 - wingLineY) * (height - 24 - wingLineY)
      );
    const rightWingEndX =
      width / 2 +
      Math.sqrt(
        COURT_DIMENSIONS.threePointRadius * COURT_DIMENSIONS.threePointRadius -
          (height - 24 - wingLineY) * (height - 24 - wingLineY)
      );

    switch (zoneName) {
      case "Restricted Area":
        // Semicircle only - arc plus diameter line at bottom
        return `M ${width / 2 - COURT_DIMENSIONS.restrictedAreaRadius} ${height - 4 * 14.4} 
                A ${COURT_DIMENSIONS.restrictedAreaRadius} ${
          COURT_DIMENSIONS.restrictedAreaRadius
        } 0 0 1 ${width / 2 + COURT_DIMENSIONS.restrictedAreaRadius} ${height - 4 * 14.4} 
                Z`;

      case "In The Paint (Non-RA)": {
        // Paint area minus restricted area - single continuous path
        const paintLeft = (width - COURT_DIMENSIONS.paintWidth) / 2;
        const paintRight = (width + COURT_DIMENSIONS.paintWidth) / 2;
        const paintTop = height - 24 - COURT_DIMENSIONS.freeThrowLineDistance;
        const paintBottom = height - 24;
        const restrictedY = height - 4 * 14.4;
        const restrictedRadius = COURT_DIMENSIONS.restrictedAreaRadius;
        const centerX = width / 2;

        // Create path that goes around paint area and around restricted area
        return `M ${paintLeft} ${paintTop} 
                L ${paintRight} ${paintTop} 
                L ${paintRight} ${paintBottom} 
                L ${centerX + restrictedRadius} ${paintBottom}
                L ${centerX + restrictedRadius} ${restrictedY}
                A ${restrictedRadius} ${restrictedRadius} 0 0 0 ${
          centerX - restrictedRadius
        } ${restrictedY}
                L ${centerX - restrictedRadius} ${paintBottom}
                L ${paintLeft} ${paintBottom}
                L ${paintLeft} ${paintTop} Z`;
      }

      case "Mid-Range": {
        // Mid-Range: Inside 3-point line, outside paint
        const arcStartAngle = Math.asin(
          (width / 2 - COURT_DIMENSIONS.threePointStraightDistance) /
            COURT_DIMENSIONS.threePointRadius
        );
        const arcStartX = width / 2 - Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcStartY = height - 24 - Math.cos(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcEndX = width / 2 + Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;

        const paintLeft = (width - COURT_DIMENSIONS.paintWidth) / 2;
        const paintRight = (width + COURT_DIMENSIONS.paintWidth) / 2;
        const paintTop = height - 24 - COURT_DIMENSIONS.freeThrowLineDistance;
        const paintBottom = height - 24;

        // Create area inside 3-point line, outside paint
        return `M ${COURT_DIMENSIONS.threePointStraightDistance} ${paintBottom}
                L ${COURT_DIMENSIONS.threePointStraightDistance} ${arcStartY}
                L ${arcStartX} ${arcStartY}
                A ${COURT_DIMENSIONS.threePointRadius} ${
          COURT_DIMENSIONS.threePointRadius
        } 0 0 1 ${arcEndX} ${arcStartY}
                L ${width - COURT_DIMENSIONS.threePointStraightDistance} ${arcStartY}
                L ${width - COURT_DIMENSIONS.threePointStraightDistance} ${paintBottom}
                L ${paintRight} ${paintBottom}
                L ${paintRight} ${paintTop}
                L ${paintLeft} ${paintTop}
                L ${paintLeft} ${paintBottom}
                L ${COURT_DIMENSIONS.threePointStraightDistance} ${paintBottom} Z`;
      }

      case "Above the Break 3": {
        // Above the Break 3: Single continuous path tracing actual boundary
        const arcStartAngle = Math.asin(
          (width / 2 - COURT_DIMENSIONS.threePointStraightDistance) /
            COURT_DIMENSIONS.threePointRadius
        );
        const arcStartX = width / 2 - Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcStartY = height - 24 - Math.cos(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcEndX = width / 2 + Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;

        // Trace the actual perimeter: outside court edges, then along 3-point arc
        return `M ${leftWingEndX} ${wingLineY}
                L 0 ${wingLineY}
                L 0 0
                L ${width} 0
                L ${width} ${wingLineY}
                L ${rightWingEndX} ${wingLineY}
                L ${arcEndX} ${arcStartY}
                A ${COURT_DIMENSIONS.threePointRadius} ${COURT_DIMENSIONS.threePointRadius} 0 0 0 ${arcStartX} ${arcStartY}
                L ${leftWingEndX} ${wingLineY} Z`;
      }
      
      case 'Left Corner 3': {
        // Left corner 3: continuous path tracing actual boundaries
        const arcStartAngle = Math.asin((width/2 - COURT_DIMENSIONS.threePointStraightDistance) / COURT_DIMENSIONS.threePointRadius);
        const arcStartX = width/2 - Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcStartY = height - 24 - Math.cos(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        
        // Calculate wing intersection angle and point on arc
        const wingIntersectionAngle = Math.acos((height - 24 - wingLineY) / COURT_DIMENSIONS.threePointRadius);
        const wingArcX = width/2 - Math.sin(wingIntersectionAngle) * COURT_DIMENSIONS.threePointRadius;
        
        return `M 0 ${wingLineY}
                L ${leftWingEndX} ${wingLineY}
                L ${wingArcX} ${height - 24 - Math.cos(wingIntersectionAngle) * COURT_DIMENSIONS.threePointRadius}
                A ${COURT_DIMENSIONS.threePointRadius} ${COURT_DIMENSIONS.threePointRadius} 0 0 0 ${arcStartX} ${arcStartY}
                L ${COURT_DIMENSIONS.threePointStraightDistance} ${arcStartY}
                L ${COURT_DIMENSIONS.threePointStraightDistance} ${height - 24}
                L 0 ${height - 24}
                L 0 ${wingLineY} Z`;
      }
      
      case 'Right Corner 3': {
        // Right corner 3: continuous path tracing actual boundaries
        const arcStartAngle = Math.asin((width/2 - COURT_DIMENSIONS.threePointStraightDistance) / COURT_DIMENSIONS.threePointRadius);
        const arcEndX = width/2 + Math.sin(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        const arcEndY = height - 24 - Math.cos(arcStartAngle) * COURT_DIMENSIONS.threePointRadius;
        
        // Calculate wing intersection angle and point on arc
        const wingIntersectionAngle = Math.acos((height - 24 - wingLineY) / COURT_DIMENSIONS.threePointRadius);
        const wingArcX = width/2 + Math.sin(wingIntersectionAngle) * COURT_DIMENSIONS.threePointRadius;
        
        return `M ${rightWingEndX} ${wingLineY}
                L ${width} ${wingLineY}
                L ${width} ${height - 24}
                L ${width - COURT_DIMENSIONS.threePointStraightDistance} ${height - 24}
                L ${width - COURT_DIMENSIONS.threePointStraightDistance} ${arcEndY}
                L ${arcEndX} ${arcEndY}
                A ${COURT_DIMENSIONS.threePointRadius} ${COURT_DIMENSIONS.threePointRadius} 0 0 0 ${wingArcX} ${height - 24 - Math.cos(wingIntersectionAngle) * COURT_DIMENSIONS.threePointRadius}
                L ${rightWingEndX} ${wingLineY} Z`;
      }

      default:
        return "";
    }
  };

  const drawCourt = () => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const {
      width,
      height,
      paintWidth,
      freeThrowLineDistance,
      threePointRadius,
      threePointStraightDistance,
      restrictedAreaRadius,
      basketRadius,
      backboardWidth,
    } = COURT_DIMENSIONS;

    const margin = { top: 40, right: 40, bottom: 40, left: 40 };

    // Create main group
    const court = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);

    // Court background - restored to full height
    court
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", height - 20)
      .attr("fill", "#f4e4bc") // Basketball court color
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Half-court line (top of the court)
    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", width)
      .attr("y2", 0)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Three-point line - arc with shortened straight sections
    // Calculate where straight sections should end (where they meet the arc)
    const arcIntersectionY =
      height -
      20 -
      Math.sqrt(
        threePointRadius * threePointRadius -
          (width / 2 - threePointStraightDistance) * (width / 2 - threePointStraightDistance)
      );

    // Left straight section (3' from sideline, shortened)
    court
      .append("line")
      .attr("x1", threePointStraightDistance)
      .attr("y1", height - 20)
      .attr("x2", threePointStraightDistance)
      .attr("y2", arcIntersectionY)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Right straight section (3' from sideline, shortened)
    court
      .append("line")
      .attr("x1", width - threePointStraightDistance)
      .attr("y1", height - 20)
      .attr("x2", width - threePointStraightDistance)
      .attr("y2", arcIntersectionY)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Three-point arc
    const arcStartAngle = Math.asin((width / 2 - threePointStraightDistance) / threePointRadius);

    court
      .append("path")
      .attr(
        "d",
        `M ${width / 2 - Math.sin(arcStartAngle) * threePointRadius} ${
          height - 20 - Math.cos(arcStartAngle) * threePointRadius
        } 
                  A ${threePointRadius} ${threePointRadius} 0 0 1 ${
          width / 2 + Math.sin(arcStartAngle) * threePointRadius
        } ${height - 20 - Math.cos(arcStartAngle) * threePointRadius}`
      )
      .attr("fill", "none")
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Free throw lane (paint) - only bottom and sides from free throw line down
    // Left side (only from free throw line to baseline)
    court
      .append("line")
      .attr("x1", (width - paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width - paintWidth) / 2)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Right side (only from free throw line to baseline)
    court
      .append("line")
      .attr("x1", (width + paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width + paintWidth) / 2)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Bottom edge only
    court
      .append("line")
      .attr("x1", (width - paintWidth) / 2)
      .attr("y1", height - 20)
      .attr("x2", (width + paintWidth) / 2)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Free throw line - 15' from baseline, 2" wide
    court
      .append("line")
      .attr("x1", (width - paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width + paintWidth) / 2)
      .attr("y2", height - 20 - freeThrowLineDistance)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Restricted area - 4' radius semicircle from center of basket (closed at bottom)
    court
      .append("path")
      .attr(
        "d",
        `M ${width / 2 - restrictedAreaRadius} ${height - 4 * 14.4} 
                  A ${restrictedAreaRadius} ${restrictedAreaRadius} 0 0 1 ${
          width / 2 + restrictedAreaRadius
        } ${height - 4 * 14.4}
                  L ${width / 2 - restrictedAreaRadius} ${height - 4 * 14.4} Z`
      )
      .attr("fill", "none")
      .attr("stroke", "#000")
      .attr("stroke-width", 2);

    // Basket/Rim - moved up slightly and aligned with backboard
    court
      .append("circle")
      .attr("cx", width / 2)
      .attr("cy", height - 5.25 * 14.4)
      .attr("r", basketRadius)
      .attr("fill", "none")
      .attr("stroke", "#d2691e") // Orange color for rim
      .attr("stroke-width", 3);

    // Backboard - 6' wide, aligned with basket
    court
      .append("line")
      .attr("x1", width / 2 - backboardWidth / 2)
      .attr("y1", height - 4 * 14.4)
      .attr("x2", width / 2 + backboardWidth / 2)
      .attr("y2", height - 4 * 14.4)
      .attr("stroke", "#000")
      .attr("stroke-width", 6);

    // Court boundary lines
    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    court
      .append("line")
      .attr("x1", width)
      .attr("y1", 0)
      .attr("x2", width)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Baseline - shifted up
    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", height - 24)
      .attr("x2", width)
      .attr("y2", height - 24)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Wing lines (2/5 up from bottom, with gaps for 3-point line)
    const wingLineY = height - (2 / 5) * height;
    const leftWingEndX =
      width / 2 -
      Math.sqrt(
        threePointRadius * threePointRadius - (height - 24 - wingLineY) * (height - 24 - wingLineY)
      );
    const rightWingEndX =
      width / 2 +
      Math.sqrt(
        threePointRadius * threePointRadius - (height - 24 - wingLineY) * (height - 24 - wingLineY)
      );

    // Left wing line
    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", wingLineY)
      .attr("x2", leftWingEndX)
      .attr("y2", wingLineY)
      .attr("stroke", "#666")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    // Right wing line
    court
      .append("line")
      .attr("x1", rightWingEndX)
      .attr("y1", wingLineY)
      .attr("x2", width)
      .attr("y2", wingLineY)
      .attr("stroke", "#666")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    // Plot shots based on visualization mode
    if (shotData.length > 0) {
      console.log("Plotting made shots:", shotData.length);

      if (visualizationMode === "points") {
        // Original point plotting
        let plotted = 0;
        let outOfBounds = 0;
        shotData.forEach((shot, i) => {
          const coords = transformCoordinates(shot.loc_x, shot.loc_y);

          if (i < 5) {
            // Log first 5 shots for debugging
            console.log(`Shot ${i}: (${shot.loc_x}, ${shot.loc_y}) -> (${coords.x}, ${coords.y})`);
          }

          // With proper normalization, most shots should be within bounds
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
              .attr("fill", "#00aa00") // Green for made shots only
              .attr("opacity", 0.7)
              .attr("stroke", "#000")
              .attr("stroke-width", 0.5)
              .append("title")
              .text(
                `Made shot from ${
                  Math.round(Math.sqrt(shot.loc_x * shot.loc_x + shot.loc_y * shot.loc_y) / 14.4 * 140) / 100
                } feet from the basket`
              );
            plotted++;
          } else {
            outOfBounds++;
          }
        });

        console.log(
          `Plotted ${plotted} out of ${shotData.length} shots (${outOfBounds} out of bounds)`
        );
      } else {
        // Zone visualization using breakdown data
        console.log("Drawing zone overlays for", selectedPlayer);
        console.log("Available zones:", selectedPlayerZones);

        // Define the 6 zones in order of drawing (back to front for proper layering)
        const zoneOrder = [
          "Above the Break 3",
          "Mid-Range",
          "In The Paint (Non-RA)",
          "Left Corner 3",
          "Right Corner 3",
          "Restricted Area",
        ];

        // Draw zone overlays
        zoneOrder.forEach((zoneName) => {
          const attemptPercentage = selectedPlayerZones[zoneName] || 0;
          const zonePath = createZonePath(zoneName);
          
          if (zonePath) {
            const zoneColor = getZoneColor(attemptPercentage);

            console.log(
              `Drawing ${zoneName}: ${(attemptPercentage * 100).toFixed(
                1
              )}% attempts, color: ${zoneColor}`
            );

            court
              .append("path")
              .attr("d", zonePath)
              .attr("fill", zoneColor)
              .attr("stroke", "rgba(0, 100, 0, 0.3)")
              .attr("stroke-width", 1)
              .attr("fill-rule", "nonzero")
              .append("title")
              .text(
                `${zoneName}: ${(attemptPercentage * 100).toFixed(1)}% of ${selectedPlayer}'s shots`
              );
          }
        });

        // Add zone labels
        const zoneLabels = [
          { zone: 'Above the Break 3', displayName: 'Above the Break 3', x: width/2, y: height * 0.15, anchor: 'middle' },
          { zone: 'Mid-Range', displayName: 'Mid-Range', x: width/2, y: height * 0.6, anchor: 'middle' },
          { zone: 'In The Paint (Non-RA)', displayName: 'The Paint', x: width/2, y: height - 130, anchor: 'middle' },
          { zone: 'Restricted Area', displayName: 'Restricted Area', x: width/2, y: height - 60, anchor: 'middle' },
          { zone: 'Left Corner 3', displayName: 'Left Corner 3', x: 10, y: height - 235, anchor: 'start' },
          { zone: 'Right Corner 3', displayName: 'Right Corner 3', x: width - 10, y: height - 235, anchor: 'end' }
        ];
        
        zoneLabels.forEach(({ zone, displayName, x, y, anchor }) => {
          const attemptPercentage = selectedPlayerZones[zone] || 0;
          // Zone name label (above)
          court
            .append('text')
            .attr('x', x)
            .attr('y', y - 8)
            .attr('text-anchor', anchor)
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', '#000')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('paint-order', 'stroke')
            .text(displayName);
          
          // Percentage label (below)
          court
            .append('text')
            .attr('x', x)
            .attr('y', y + 8)
            .attr('text-anchor', anchor)
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', '#000')
            .attr('stroke', '#fff')
            .attr('stroke-width', 3)
            .attr('paint-order', 'stroke')
            .text(`${(attemptPercentage * 100).toFixed(1)}%`);
        });
      }
    } else {
      console.log("No shot data to plot");
    }
  };

  useEffect(() => {
    drawCourt();
  }, [shotData, visualizationMode, selectedPlayerZones]); // Redraw when shot data, mode, or zones change

  return (
    <div className="flex flex-col items-center gap-4">
      <h3 className="text-5xl sporting-outline">Basketball Shot Chart</h3>
      <div className="text-lg font-semibold">Learn more about your similar players!</div>
      
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
          onChange={(e) => setVisualizationMode(e.target.value as "points" | "zones")}
        >
          <option value="points">Shot Sample</option>
          <option value="zones">Shot Zones</option>
        </select>
      </div>

      {shotData.length > 0 && (
        <div className="text-lg font-semibold">
          {loading ? 'Loading shot data...' : 
           visualizationMode === 'zones' ? 
             `Percentage of Shots Attempted from Court Zones Across Career for ${selectedPlayer}` : 
             `Sample of 500 Made Shots Across Career for ${selectedPlayer}`}
        </div>
      )}

      {shotData.length === 0 && !loading && (
        <div className="text-sm text-gray-600">No shot data available for {selectedPlayer}</div>
      )}

      {/* Court SVG */}
      <svg
        ref={svgRef}
        width={800}
        height={660}
        className="border border-gray-300 rounded-lg bg-base-300 shadow-lg"
      />
    </div>
  );
};

export default ShotChart;
