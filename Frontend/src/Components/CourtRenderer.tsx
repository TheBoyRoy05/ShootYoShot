import * as d3 from "d3";
import { COURT_DIMENSIONS, SVG_DIMENSIONS } from '../Utils/courtConstants';

interface CourtRendererProps {
  svgRef: React.RefObject<SVGSVGElement | null>;
}

export const CourtRenderer = ({ svgRef }: CourtRendererProps) => {
  const drawCourtLines = () => {
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

    // Calculate center offset to position court in middle of SVG
    const horizontalOffset = (SVG_DIMENSIONS.width - width) / 2;
    const verticalOffset = (SVG_DIMENSIONS.height - height) / 2;

    // Create main group with centering transform
    const court = svg
      .append("g")
      .attr("transform", `translate(${horizontalOffset}, ${verticalOffset})`);

    // Court background
    court
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", height - 20)
      .attr("fill", "#f4e4bc")
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Half-court line
    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", width)
      .attr("y2", 0)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Three-point line calculations
    const arcIntersectionY =
      height -
      20 -
      Math.sqrt(
        threePointRadius * threePointRadius -
          (width / 2 - threePointStraightDistance) * (width / 2 - threePointStraightDistance)
      );

    // Three-point line left straight section
    court
      .append("line")
      .attr("x1", threePointStraightDistance)
      .attr("y1", height - 20)
      .attr("x2", threePointStraightDistance)
      .attr("y2", arcIntersectionY)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Three-point line right straight section
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

    // Paint area lines
    court
      .append("line")
      .attr("x1", (width - paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width - paintWidth) / 2)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    court
      .append("line")
      .attr("x1", (width + paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width + paintWidth) / 2)
      .attr("y2", height - 20)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Note: Paint area bottom line removed to avoid double line with court rectangle stroke

    // Free throw line
    court
      .append("line")
      .attr("x1", (width - paintWidth) / 2)
      .attr("y1", height - 20 - freeThrowLineDistance)
      .attr("x2", (width + paintWidth) / 2)
      .attr("y2", height - 20 - freeThrowLineDistance)
      .attr("stroke", "#000")
      .attr("stroke-width", 3);

    // Restricted area
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

    // Basket/Rim
    court
      .append("circle")
      .attr("cx", width / 2)
      .attr("cy", height - 5.25 * 14.4)
      .attr("r", basketRadius)
      .attr("fill", "none")
      .attr("stroke", "#d2691e")
      .attr("stroke-width", 3);

    // Backboard
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

    // Note: Baseline line removed to avoid double line - court rectangle stroke serves as boundary

    // Wing lines
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

    court
      .append("line")
      .attr("x1", 0)
      .attr("y1", wingLineY)
      .attr("x2", leftWingEndX)
      .attr("y2", wingLineY)
      .attr("stroke", "#666")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    court
      .append("line")
      .attr("x1", rightWingEndX)
      .attr("y1", wingLineY)
      .attr("x2", width)
      .attr("y2", wingLineY)
      .attr("stroke", "#666")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    return court;
  };

  return { drawCourtLines };
}; 